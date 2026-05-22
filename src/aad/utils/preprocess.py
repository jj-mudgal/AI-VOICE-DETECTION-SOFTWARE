import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def normalize_waveform(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + 1e-6)


def pad_or_trim(x: torch.Tensor, target_len: int) -> torch.Tensor:
    T = x.shape[-1]
    if T < target_len:
        x = torch.nn.functional.pad(x, (0, target_len - T))
    else:
        x = x[..., :target_len]
    return x


def compute_waveform_image(waveform: np.ndarray, sr: int) -> np.ndarray:
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


def load_and_preprocess(path: str, target_sr: int = 16000):
    import librosa
    waveform, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    tensor = normalize_waveform(tensor)
    return waveform, tensor, sr


def compute_prob_bar(human_prob: float, ai_prob: float) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(4, 2))
    bars = ax.barh(["Human", "AI Generated"], [human_prob, ai_prob],
                   color=["#28c840", "#ff5f57"])
    ax.set_xlim([0, 1])
    ax.set_xlabel("Probability")
    ax.set_title("Class Probabilities")
    for bar, val in zip(bars, [human_prob, ai_prob]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=9)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return buf
