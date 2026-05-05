import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T

SUPPORTED_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
LABEL_MAP = {"human": 0, "synthetic": 1}


def load_audio(path, sr=16000):
    try:
        waveform, _ = librosa.load(path, sr=sr)
        if waveform is None or len(waveform) == 0:
            waveform = np.zeros(sr)
        return waveform
    except Exception:
        return np.zeros(sr)


def scan_files(root_dir):
    files = []
    for label in ["human", "synthetic"]:
        folder = os.path.join(root_dir, label)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.append((os.path.join(folder, fname), label))
    return files


class AudioDataset(Dataset):
    def __init__(self, root_dir, sr=16000, max_len=16000 * 5):
        self.sr = sr
        self.max_len = max_len
        self.samples = scan_files(root_dir)

        if len(self.samples) == 0:
            raise RuntimeError(f"No audio files found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform = load_audio(path, sr=self.sr)

        # Pad or trim to fixed length
        if len(waveform) < self.max_len:
            waveform = np.pad(waveform, (0, self.max_len - len(waveform)))
        else:
            waveform = waveform[:self.max_len]

        x = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(LABEL_MAP[label], dtype=torch.long)
        return x, y


class AudioAugmentation:
    def __init__(self):
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)

    def __call__(self, spec):
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)
        return spec
