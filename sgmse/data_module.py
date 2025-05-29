
from os.path import join
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from torchaudio import load
import numpy as np
import torch.nn.functional as F

import sys
sys.path.append('/mnt/workspace/home/zhangjunan/masksr')
from dataset.audio_degradation_pipeline import process_from_audio_path, read_audio, process_live_performance_mix
import random
import json5

def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")


class Specs(Dataset):
    def __init__(self, data_dir, subset, dummy, shuffle_spec, num_frames,
            
            speech_list, noise_list, long_rir_list, short_rir_list, degradation_config,
            
            sr=48000,
            format='default', normalize="noisy", spec_transform=None,
            stft_kwargs=None, **kwargs 
            ):
        # # Read file paths according to file naming format.
        # if format == "default":
        #     self.clean_files = []
        #     self.clean_files += sorted(glob(join(data_dir, subset, "clean", "*.wav")))
        #     self.clean_files += sorted(glob(join(data_dir, subset, "clean", "**", "*.wav")))
        #     self.noisy_files = []
        #     self.noisy_files += sorted(glob(join(data_dir, subset, "noisy", "*.wav")))
        #     self.noisy_files += sorted(glob(join(data_dir, subset, "noisy", "**", "*.wav")))
        # elif format == "reverb":
        #     self.clean_files = []
        #     self.clean_files += sorted(glob(join(data_dir, subset, "anechoic", "*.wav")))
        #     self.clean_files += sorted(glob(join(data_dir, subset, "anechoic", "**", "*.wav")))
        #     self.noisy_files = []
        #     self.noisy_files += sorted(glob(join(data_dir, subset, "reverb", "*.wav")))
        #     self.noisy_files += sorted(glob(join(data_dir, subset, "reverb", "**", "*.wav")))
        # else:
        #     # Feel free to add your own directory format
        #     raise NotImplementedError(f"Directory format {format} unknown!")

        self.speech_list, self.seperation_speech_dataset_dict, self.noise_list, self.long_rir_list, self.short_rir_list = self.get_list(speech_list, noise_list, long_rir_list, short_rir_list)
        self.degradation_config = degradation_config
        self.sr = sr
        no_live_simulation = degradation_config["no_live_simulation"] if "no_live_simulation" in degradation_config else False
        self.reverb_v3 = degradation_config["reverb_v3"] if "reverb_v3" in degradation_config else False
        print("reverb_v3:", self.reverb_v3)
        print("no_live_simulation:", no_live_simulation)

        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"

    def get_list(self, speech_list, noise_list, long_rir_list, short_rir_list):
        # 对于speech，以\t前后的数据集为key，构造一个字典
        speech_paths = []
        speech_dataset_dict = {}
        for speech_file in speech_list:
            # speech_paths.extend([i.split('\t', 1)[1].strip() for i in open(speech_file, 'r').readlines()])
            if isinstance(speech_file, str):
                speech_file_ratio = 1
            elif isinstance(speech_file, list):
                speech_file, speech_file_ratio = speech_file[0], int(speech_file[1])
            
            speech_paths_temp = []
            for i in open(speech_file, 'r').readlines():
                key, path = i.split('\t', 1)
                key = key.strip()
                path = path.strip()
                if key not in speech_dataset_dict:
                    speech_dataset_dict[key] = []
                speech_dataset_dict[key].append(path)
                speech_paths_temp.append((key, path))
            speech_paths_temp = speech_file_ratio * speech_paths_temp
            speech_paths.extend(speech_paths_temp)
        
        # TODO: Emilia dataset可以都放进去
        seperation_speech_dataset_dict = {}
        for key in speech_dataset_dict.keys():
            seperation_speech_dataset_dict[key] = []
        for key, value in speech_dataset_dict.items():
            if "Emilia" in key or 'msst' in key:
                for key_sep in seperation_speech_dataset_dict.keys():
                    seperation_speech_dataset_dict[key_sep].extend(value)
            else:
                for key_sep in seperation_speech_dataset_dict.keys():
                    if key_sep != key:
                        seperation_speech_dataset_dict[key_sep].extend(value)
                
        
        # 找到第一个\t，之后的内容全部是路径
        noise_paths = []
        for noise_file in noise_list:
            noise_paths.extend([i.split('\t', 1)[1].strip() for i in open(noise_file, 'r').readlines()])
        
        long_rir_paths = []
        for rir_file in long_rir_list:
            long_rir_paths.extend([i.split('\t', 1)[1].strip() for i in open(rir_file, 'r').readlines()])
        
        short_rir_paths = []
        for rir_file in short_rir_list:
            short_rir_paths.extend([i.split('\t', 1)[1].strip() for i in open(rir_file, 'r').readlines()])

        print('datasets:', len(speech_dataset_dict))
        print('vocal:', len(speech_paths), speech_paths[:5])
        print('noise:', len(noise_paths), noise_paths[:5])
        print('long_rir:', len(long_rir_paths), long_rir_paths[:5])
        print('short_rir:', len(short_rir_paths), short_rir_paths[:5])

        return speech_paths, seperation_speech_dataset_dict, noise_paths, long_rir_paths, short_rir_paths


    def get_batch(self, speech_path, noise_path, long_rir_path, short_rir_path):
        speech_dataset_key, speech_path = speech_path
        to_seperate_vocal_paths = None
        # if random.random() < self.seperation_ratio:
        #     to_seperate_vocal_paths = random.sample(self.seperation_speech_dataset_dict[speech_dataset_key], random.randint(1, 2))
        #     use_prompt = True
        #     task_type = self.task_map["seperation"]
        #     degradation_config = self.seperation_degradation_config
        # else:
        #     use_prompt = random.random() < 0.5 # 50%使用prompt
        #     task_type = self.task_map["enhancement"]
        #     degradation_config = self.degradation_config
        degradation_config = self.degradation_config
        
        if "silence_ratio" in degradation_config and random.random() < degradation_config["silence_ratio"]:
            use_prompt = False
            noise, _ = read_audio(noise_path, force_1ch=True, fs=self.sr)
            noisy_speech = torch.from_numpy(noise).float()
            # -100db noise
            target_dB = -100
            amplitude = 10 ** (target_dB / 20)  # 计算线性幅度
            speech_sample = torch.randn_like(noisy_speech) * amplitude
            
            # assert speech_sample.shape == noisy_speech.shape
            # speech_sample, noisy_speech = self.pad_or_truncate(speech_sample, noisy_speech)
            # return speech_sample, noisy_speech, torch.tensor(use_prompt, dtype=torch.bool), torch.tensor(task_type, dtype=torch.long)
        else:
            no_live_simulation = degradation_config["no_live_simulation"] if "no_live_simulation" in degradation_config else False
            # 20%加入seperation speech
            if random.random() < 0.5 and not no_live_simulation:
                # Process the audio file using live performance simulation
                speech_sample, noise_sample, noisy_speech, fs = process_live_performance_mix(
                    vocal_path=speech_path, 
                    noise_path=noise_path, 
                    rir_path=long_rir_path,
                    to_seperate_vocal_paths=to_seperate_vocal_paths,
                    post_rir_paths=[short_rir_path],
                    fs=self.sr, 
                    force_1ch=True,
                    degradation_config=degradation_config,
                    reverb_v3=self.reverb_v3,
                )
            else:
                # Process the audio file with academic simulation
                speech_sample, noise_sample, noisy_speech, fs = process_from_audio_path(
                    vocal_path=speech_path, 
                    noise_path=noise_path, 
                    rir_path=random.choice([long_rir_path, short_rir_path]),
                    to_seperate_vocal_paths=to_seperate_vocal_paths,
                    fs=self.sr, 
                    force_1ch=True,
                    degradation_config=degradation_config,
                    reverb_v3=self.reverb_v3,
                )
            
            assert speech_sample.shape == noisy_speech.shape
            
            # Convert numpy arrays to torch tensors
            speech_sample = torch.from_numpy(speech_sample).float()
            noisy_speech = torch.from_numpy(noisy_speech).float()

        # x, _ = load(self.clean_files[i])
        # y, _ = load(self.noisy_files[i])
        x = speech_sample
        y = noisy_speech

        # formula applies for center=True
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len-target_len))
            else:
                start = int((current_len-target_len)/2)
            x = x[..., start:start+target_len]
            y = y[..., start:start+target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')
            y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant')

        # normalize w.r.t to the noisy or the clean signal or not at all
        # to ensure same clean signal power in x and y.
        if self.normalize == "noisy":
            normfac = y.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        elif self.normalize == "not":
            normfac = 1.0
        x = x / normfac
        y = y / normfac

        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)
        
        # import soundfile as sf
        # print(x.shape, y.shape)
        # sf.write('x.wav', x[0].numpy(), 48000)
        # sf.write('y.wav', y[0].numpy(), 48000)
        
        # print(X.shape, Y.shape)
        
        # import matplotlib.pyplot as plt
        # # 绘制 X 的 log spectrogram
        # plt.figure(figsize=(10, 6))
        # plt.imshow(torch.abs(X[0]).log1p().detach().numpy(), aspect='auto', origin='lower', cmap='inferno')
        # plt.title('STFT Spectrogram of x')
        # plt.xlabel('Time')
        # plt.ylabel('Frequency')
        # plt.colorbar(label='Magnitude')
        # plt.tight_layout()
        # plt.savefig('X_internal.png')

        # # 绘制 Y 的 spectrogram
        # plt.figure(figsize=(10, 6))
        # plt.imshow(torch.abs(Y[0]).log1p().detach().numpy(), aspect='auto', origin='lower', cmap='inferno')
        # plt.title('STFT Spectrogram of y')
        # plt.xlabel('Time')
        # plt.ylabel('Frequency')
        # plt.colorbar(label='Magnitude')
        # plt.tight_layout()
        # plt.savefig('y_internal.png')

        X, Y = self.spec_transform(X), self.spec_transform(Y)
        return X, Y
    
    def __getitem__(self, idx):
        while True:
            try:
                speech_path = self.speech_list[idx]
                noise_path = random.choice(self.noise_list)
                long_rir_path = random.choice(self.long_rir_list)
                short_rir_path = random.choice(self.short_rir_list)
                batch = self.get_batch(speech_path, noise_path, long_rir_path, short_rir_path)
                break
            except Exception as e:
                print(speech_path, noise_path, long_rir_path, short_rir_path)
                print(e)
                # import traceback
                # traceback.print_exc()
                idx = random.randint(0, len(self) - 1)
        return batch

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return int(len(self.speech_list)/200)
        else:
            return len(self.speech_list)


class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--config", type=str, help="Config file for the dataset.")
        parser.add_argument("--base_dir", type=str, help="The base directory of the dataset. Should contain `train`, `valid` and `test` subdirectories, each of which contain `clean` and `noisy` subdirectories.")
        parser.add_argument("--format", type=str, choices=("default", "reverb"), default="default", help="Read file paths according to file naming format.")
        parser.add_argument("--batch_size", type=int, default=8, help="The batch size. 8 by default.")
        parser.add_argument("--n_fft", type=int, default=510, help="Number of FFT bins. 510 by default.")   # to assure 256 freq bins
        parser.add_argument("--hop_length", type=int, default=128, help="Window hop length. 128 by default.")
        parser.add_argument("--num_frames", type=int, default=256, help="Number of frames for the dataset. 256 by default.")
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann", help="The window function to use for the STFT. 'hann' by default.")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use for DataLoaders. 4 by default.")
        parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
        parser.add_argument("--spec_factor", type=float, default=0.15, help="Factor to multiply complex STFT coefficients by. 0.15 by default.")
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5, help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). 0.5 by default.")
        parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy", help="Normalize the input waveforms by the clean signal, the noisy signal, or not at all.")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"), default="exponent", help="Spectogram transformation for input representation.")
        return parser

    def __init__(
        self, config, base_dir, format='default', batch_size=8,
        n_fft=510, hop_length=128, num_frames=256, window='hann',
        num_workers=4, dummy=False, spec_factor=0.15, spec_abs_exponent=0.5,
        gpu=True, normalize='noisy', transform_type="exponent", **kwargs
    ):
        super().__init__()
        self.base_dir = base_dir
        self.format = format
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.normalize = normalize
        self.transform_type = transform_type
        self.kwargs = kwargs
        
        with open(config) as f:
            config = json5.load(f)["dataset"]
        self.speech_list = config["train"]["speech_list"]
        self.noise_list = config["train"]["noise_list"]
        self.long_rir_list = config["train"]["long_rir_list"]
        self.short_rir_list = config["train"]["short_rir_list"]
        self.degradation_config = config["degradation_config"]
        self.sr = 48000

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs, num_frames=self.num_frames,
            spec_transform=self.spec_fwd, **self.kwargs
        )
        if stage == 'fit' or stage is None:
            self.train_set = Specs(data_dir=self.base_dir, subset='train',
                dummy=self.dummy, shuffle_spec=True, format=self.format,
                normalize=self.normalize, 
                speech_list=self.speech_list, noise_list=self.noise_list, long_rir_list=self.long_rir_list, short_rir_list=self.short_rir_list, degradation_config=self.degradation_config,
                **specs_kwargs)
            self.valid_set = self.train_set[:20]
        #     self.valid_set = Specs(data_dir=self.base_dir, subset='valid',
        #         dummy=self.dummy, shuffle_spec=False, format=self.format,
        #         normalize=self.normalize, **specs_kwargs)
        # if stage == 'test' or stage is None:
        #     self.test_set = Specs(data_dir=self.base_dir, subset='test',
        #         dummy=self.dummy, shuffle_spec=False, format=self.format,
        #         normalize=self.normalize, **specs_kwargs)

    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs()**e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True
        )

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )

    def test_dataloader(self):
        # return DataLoader(
        #     self.test_set, batch_size=self.batch_size,
        #     num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        # )
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser = SpecsDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    
    # nohup python train.py --base_dir ./ --backbone ncsnpp_48k --n_fft 1534 --hop_length 384 --spec_factor 0.065 --spec_abs_exponent 0.667 --sigma-min 0.1 --sigma-max 1.0 --theta 2.0 --num_eval_files 0 --config /mnt/workspace/home/zhangjunan/enhancement-baseline/sgmse/config/anyenhance-400M-reverb-v3-no-msst-44k.json >> train.log
    
    args.config = "/mnt/workspace/home/zhangjunan/enhancement-baseline/sgmse/config/anyenhance-400M-reverb-v3-no-msst-44k.json"
    args.base_dir = "/mnt/workspace/home/zhangjunan/enhancement-baseline/sgmse/dataset"
    args.format = "default"
    args.n_fft = 1534
    args.hop_length = 384
    args.spec_factor = 0.065
    args.spec_abs_exponent = 0.667
    args.batch_size = 1
    args.num_workers = 0
    
    
    print(args)
    
    import matplotlib.pyplot as plt    

    dm = SpecsDataModule(**vars(args))
    dm.setup()
    dl = dm.train_dataloader()
    for X, Y in dl:
        print(X.shape, Y.shape) # [1, 1, 768, 256], [1, 1, 768, 256]
        # save figure
        plt.figure()
        plt.imshow(X[0, 0].abs().log1p().cpu().numpy(), aspect='auto')
        plt.colorbar()
        plt.savefig('X.png')
        plt.close()
        plt.figure()
        plt.imshow(Y[0, 0].abs().log1p().cpu().numpy(), aspect='auto')
        plt.colorbar()
        plt.savefig('Y.png')
        plt.close()
        break