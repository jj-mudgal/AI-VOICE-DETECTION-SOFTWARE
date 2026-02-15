"""
Audio dataset and preprocessing utilities
-----------------------------------------
Supports WAV / MP3 / FLAC / M4A
Automatically resamples to target SAMPLE_RATE.
"""

import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from .config import SAMPLE_RATE, DATA_DIR


class AudioDataset(Dataset):
    def __init__(self, data_dir: str = DATA_DIR, sample_rate: int = SAMPLE_RATE, augment: bool = False):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.augment = augment

        self.paths = []
        self.labels = []

        # Collect files
        for label_name, label in [("human", 0), ("synthetic", 1)]:
            folder = os.path.join(data_dir, label_name)
            if not os.path.isdir(folder):
                continue

            for f in os.listdir(folder):
                if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                    self.paths.append(os.path.join(folder, f))
                    self.labels.append(label)

        print(f"📁 Loaded dataset: {len(self.paths)} samples")

        # Optional audio augmentation
        self.augmenter = (
            Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.4),
                TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
                PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
                Shift(min_shift=-0.2, max_shift=0.2, p=0.3),
            ])
            if augment else None
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Load audio (librosa auto-detects input sample rate)
        waveform, sr = librosa.load(path, sr=None, mono=True)

        # Resample to model sample rate
        if sr != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)

        # Data augmentation
        if self.augmenter is not None:
            waveform = self.augmenter(samples=waveform, sample_rate=self.sample_rate)

        # Convert to tensor (1, T)
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        return waveform, label
