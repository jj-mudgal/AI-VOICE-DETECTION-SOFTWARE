import os
import librosa
import numpy as np
import torch
import torchaudio.transforms as T

"""
AudioDataset

Expected folder structure:

data/
  train/
    human/
    synthetic/
  val/
    human/
    synthetic/
  test/
    human/
    synthetic/

Supported formats:
.wav, .mp3, .flac, .ogg, .m4a

Labels:
- human → 0
- synthetic → 1
"""

SUPPORTED_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]


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

        for fname in os.listdir(folder):
            ext = os.path.splitext(fname)[1].lower()

            if ext in SUPPORTED_EXTENSIONS:
                files.append((os.path.join(folder, fname), label))

    return files


class AudioAugmentation:
    def __init__(self):
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)

    def __call__(self, spec):
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)
        return spec
