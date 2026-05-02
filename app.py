import gradio as gr
import torch
import torchaudio
import librosa
import matplotlib.pyplot as plt
import numpy as np

from src.aad.model import AudioDetector
from src.aad.config import SAMPLE_RATE

# ---------------------------
# Device
# ---------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ---------------------------
# Load model
# ---------------------------
model = AudioDetector().to(DEVICE)
model.load_state_dict(
    torch.load("src/aad/checkpoints/best_model.pt", map_location=DEVICE)
)
model.eval()

# ---------------------------
# Transforms
# ---------------------------
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_mels=128,
    n_fft=1024,
    hop_length=512
)

amplitude_to_db = torchaudio.transforms.AmplitudeToDB()


# ---------------------------
# Spectrogram Utility (NEW)
# ---------------------------
def compute_spectrogram_image(waveform, sr):
    S = librosa.feature.melspectrogram(y=waveform, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure()
    plt.imshow(S_db, aspect="auto", origin="lower")
    plt.title("Log-Mel Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Mel")
    plt.tight_layout()

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return img


# ---------------------------
# Prediction
# ---------------------------
def predict(audio_file):
    if audio_file is None:
        return "No input", None

    waveform, sr = librosa.load(audio_file, sr=None, mono=True)

    if sr != SAMPLE_RATE:
        waveform = librosa.resample(
            waveform, orig_sr=sr, target_sr=SAMPLE_RATE
        )

    waveform_tensor = torch.tensor(waveform).float().unsqueeze(0)

    mel = mel_transform(waveform_tensor)
    mel = amplitude_to_db(mel)

    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    mel = mel.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(mel)
        probs = torch.softmax(logits, dim=1)
        ai_score = probs[0, 1].item()

    label = "AI Generated" if ai_score > 0.5 else "Human"
    confidence = ai_score if ai_score > 0.5 else 1 - ai_score

    spec_img = compute_spectrogram_image(waveform, sr)

    return f"{label} | Confidence: {confidence:.3f}", spec_img


# ---------------------------
# UI
# ---------------------------
with gr.Blocks() as demo:

    gr.Markdown("# 🎙️ AI Voice Detection")

    with gr.Tab("Live Demo"):
        audio = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath"
        )

        out = gr.Textbox(label="Prediction")
        spec = gr.Image(label="Spectrogram")  # ✅ NEW

        audio.change(predict, audio, [out, spec])  # ✅ UPDATED

    with gr.Tab("Model Metrics"):
        gr.Markdown("### Evaluation Results")

        gr.Image("confusion_matrix.png", label="Confusion Matrix")
        gr.Image("roc_curve.png", label="ROC Curve")

        gr.Markdown(
            """
            **Validation Accuracy:** 97.5%  
            **AUC:** 0.995  
            **Precision / Recall / F1:** 0.99  
            """
        )


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    demo.launch()