"""
Audio dataset and preprocessing utilities
-----------------------------------------
Handles loading, preprocessing, and augmenting audio data
for the AI Audio Detector project.

Supports:
- Multi-format audio (WAV, FLAC, MP3)
- Automatic resampling
- Optional audio augmentations
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
        """
        Args:
            data_dir (str): Directory containing subfolders 'human' and 'synthetic'.
            sample_rate (int): Target sample rate for audio loading.
            augment (bool): Whether to apply audio augmentations during training.
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.augment = augment
        self.paths, self.labels = [], []

        # Discover files
        for label_name, label_idx in [("human", 0), ("synthetic", 1)]:
            folder = os.path.join(data_dir, label_name)
            if not os.path.isdir(folder):
                continue
            for file in os.listdir(folder):
                if file.lower().endswith((".wav", ".flac", ".mp3", ".ogg", ".m4a")):
                    self.paths.a
