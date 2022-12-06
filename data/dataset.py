import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import json
import os
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torchvision import transforms as T
from PIL import Image


class SENDataset(Dataset):
    def __init__(
        self,
        json_file: str,
        img_path: str,
        audio_path: str,
        input_size=299,
        n_fft=512,
        n_mels=40,
        win_length=250,
        hop_length=100,
    ):
        super().__init__()
        data = json.load(open(json_file, "r", encoding="utf-8"))["data"]
        walker = [
            [
                dict(
                    label=datum["class"],
                    img=img_path + "/" + datum["img"],
                    audio=audio_path + "/" + wav,
                )
                for wav in datum["wav"]
            ]
            for datum in data
        ]
        self.walker = [j for i in walker for j in i]
        subset = json_file.rsplit(os.sep, 1)[-1].split("_", 1)[0]

        self.img_transform = {
            "train": T.Compose(
                [
                    T.RandomResizedCrop(input_size),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "test": T.Compose(
                [
                    T.Resize(input_size),
                    T.CenterCrop(input_size),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }[subset]

        sample_rate = 16000  # default value
        self.audio_transform = MelSpectrogram(
            sample_rate, n_fft, win_length, hop_length, n_mels=n_mels
        )

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        item = self.walker[idx]

        img = Image.open(item["img"])
        img = self.img_transform(img)

        wav, sr = torchaudio.load(item["audio"])
        mel_spec = self.audio_transform(wav)
        mel_spec = mel_spec.squeeze().permute(1, 0)  # (len, n_mels)

        return img, mel_spec, mel_spec.size(0), item["label"]
