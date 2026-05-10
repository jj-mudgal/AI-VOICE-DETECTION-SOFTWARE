import subprocess
import numpy as np
import torch

SAMPLE_RATE = 16000
MIN_AUDIO_BYTES = 100


def load_audio(path: str, sr: int = SAMPLE_RATE) -> tuple:
    """
    Load audio via ffmpeg. Forces mono, float32, target sample rate.
    Returns (waveform_tensor, sr) where waveform_tensor is shape (1, T).
    """
    cmd = [
        "ffmpeg", "-i", path,
        "-f", "f32le",
        "-ac", "1",          # force mono
        "-ar", str(sr),
        "-",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to load {path}: {e}")

    waveform = np.frombuffer(out, dtype=np.float32).copy()

    if len(waveform) == 0:
        waveform = np.zeros(sr, dtype=np.float32)

    # Normalise to [-1, 1]
    peak = np.abs(waveform).max()
    if peak > 0:
        waveform = waveform / peak

    tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)
    return tensor, sr


def validate_audio_bytes(file_bytes: bytes) -> bool:
    """Basic sanity check on raw audio bytes."""
    return isinstance(file_bytes, bytes) and len(file_bytes) > MIN_AUDIO_BYTES
