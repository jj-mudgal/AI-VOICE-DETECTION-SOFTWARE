import torch


def normalize_waveform(x: torch.Tensor) -> torch.Tensor:
    """Zero-mean, unit-variance normalization with epsilon guard."""
    return (x - x.mean()) / (x.std() + 1e-6)


def pad_or_trim(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad with zeros or trim to target_len along last axis."""
    T = x.shape[-1]
    if T < target_len:
        pad = target_len - T
        x = torch.nn.functional.pad(x, (0, pad))
    else:
        x = x[..., :target_len]
    return x


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_waveform_image(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Return an RGB numpy array of the waveform plot."""
    duration = len(waveform) / sr
    t = np.linspace(0, duration, num=len(waveform))

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(t, waveform, color="#4EC9B0", linewidth=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.set_xlim([0, duration])
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return buf


def load_and_preprocess(path: str, target_sr: int = 16000) -> "tuple[np.ndarray, torch.Tensor, int]":
    """
    Load audio from path, resample if needed, normalize.
    Returns (raw_waveform_np, tensor_for_model, sr)
    """
    import librosa, torch
    waveform, sr = librosa.load(path, sr=None, mono=True)

    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    tensor = normalize_waveform(tensor)
    return waveform, tensor, sr
