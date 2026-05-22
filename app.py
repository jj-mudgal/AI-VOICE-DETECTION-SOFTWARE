import gradio as gr
import torch
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.aad.model import AudioDetector
from src.aad.config import SAMPLE_RATE, DEVICE
from src.aad.utils.preprocess import normalize_waveform

# ---------------------------
# Load model
# ---------------------------
model = AudioDetector().to(DEVICE)

import os
_ckpt = os.path.join("checkpoints", "best_model.pt")
if os.path.exists(_ckpt):
    model.load_state_dict(torch.load(_ckpt, map_location=DEVICE))
    print(f"[app] Loaded weights from {_ckpt}")
else:
    print("[app] No checkpoint found — using random weights")

model.eval()

LABELS = {0: "Human", 1: "AI Generated"}


# ---------------------------
# Spectrogram utility
# ---------------------------
def compute_spectrogram_image(waveform, sr):
    S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title("Log-Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return buf


# ---------------------------
# Prediction
# ---------------------------
def predict(audio_file):
    if audio_file is None:
        return "No input provided.", None, None

    waveform, sr = librosa.load(audio_file, sr=None, mono=True)

    if sr != SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    waveform_tensor = normalize_waveform(waveform_tensor)

    with torch.no_grad():
        logits = model(waveform_tensor.to(DEVICE))
        probs  = torch.softmax(logits, dim=1).cpu()[0]

    pred_idx    = probs.argmax().item()
    label       = LABELS[pred_idx]
    confidence  = probs[pred_idx].item()
    human_pct   = probs[0].item() * 100
    ai_pct      = probs[1].item() * 100

    result = (
        f"{'🟢' if pred_idx == 0 else '🔴'} {label}  |  "
        f"Confidence: {confidence:.1%}\n"
        f"Human: {human_pct:.1f}%   AI: {ai_pct:.1f}%"
    )

    spec_img = compute_spectrogram_image(waveform, sr)
    return result, spec_img


# ---------------------------
# UI
# ---------------------------
with gr.Blocks(title="AI Voice Detection") as demo:
    gr.Markdown("# 🎙️ AI Voice Detection")
    gr.Markdown("Upload or record audio to detect whether it is **human** or **AI-generated**.")

    with gr.Tab("Live Demo"):
        audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio Input")
        out   = gr.Textbox(label="Prediction", lines=2)
        spec  = gr.Image(label="Log-Mel Spectrogram")
        audio.change(predict, inputs=audio, outputs=[out, spec])

    with gr.Tab("Model Metrics"):
        gr.Markdown("### Evaluation Results")
        with gr.Row():
            gr.Image("metrics/confusion_matrix.png", label="Confusion Matrix")
            gr.Image("metrics/roc_curve.png", label="ROC Curve")
        gr.Markdown(
            "**Validation Accuracy:** 98.5%  |  "
            "**AUC:** 0.995  |  "
            "**F1:** 0.99"
        )

if __name__ == "__main__":
    demo.launch()


