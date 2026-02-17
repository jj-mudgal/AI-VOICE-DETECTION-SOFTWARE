"""
Audio dataset and preprocessing utilities
-----------------------------------------
Supports WAV / MP3 / FLAC / M4A
Loads data from structured splits:
    data/train/
    data/val/
    data/test/

Converts audio → Log-Mel Spectrogram for CNN training.
"""

import os
import torch
import librosa
import torchaudio
from torch.utils.data import Dataset
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from .config import SAMPLE_RATE, DATA_DIR


class AudioDataset(Dataset):
    def __init__(self, split="train", sample_rate=SAMPLE_RATE, augment=False):
        """
        split: "train", "val", or "test"
        augment: only True for training set
        """

        self.sample_rate = sample_rate
        self.augment = augment
        self.data_dir = os.path.join(DATA_DIR, split)

        self.paths = []
        self.labels = []

        # -------------------------
        # Collect files from split
        # -------------------------
        for label_name, label in [("human", 0), ("synthetic", 1)]:
            folder = os.path.join(self.data_dir, label_name)
            if not os.path.isdir(folder):
                continue

            for f in os.listdir(folder):
                if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                    self.paths.append(os.path.join(folder, f))
                    self.labels.append(label)

        print(f"📁 Loaded {split} dataset: {len(self.paths)} samples")

        # -------------------------
        # Audio augmentation (train only)
        # -------------------------
        self.augmenter = (
            Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.3),
                TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2),
                PitchShift(min_semitones=-2, max_semitones=2, p=0.2),
                Shift(min_shift=-0.1, max_shift=0.1, p=0.2),
            ])
            if augment else None
        )

        # -------------------------
        # Mel Spectrogram Transform
        # -------------------------
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=128,
            n_fft=1024,
            hop_length=512
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Load audio
        waveform, sr = librosa.load(path, sr=None, mono=True)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=sr,
                target_sr=self.sample_rate
            )

        # Augmentation (train only)
        if self.augmenter is not None:
            waveform = self.augmenter(
                samples=waveform,
                sample_rate=self.sample_rate
            )

        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

        # Convert to Mel Spectrogram
        mel = self.mel_transform(waveform)
        mel = self.amplitude_to_db(mel)

        # Normalize per sample
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        return mel, label
