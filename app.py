import gradio as gr
import torch
import torchaudio
import librosa
from src.aad.model import AudioDetector
from src.aad.config import SAMPLE_RATE

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

model = AudioDetector().to(DEVICE)
model.load_state_dict(torch.load("src/aad/checkpoints/best_model.pt", map_location=DEVICE))
model.eval()

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_mels=128,
    n_fft=1024,
    hop_length=512
)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

def predict(audio_file):
    waveform, sr = librosa.load(audio_file, sr=None, mono=True)

    if sr != SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)

    waveform = torch.tensor(waveform).float().unsqueeze(0)
    mel = mel_transform(waveform)
    mel = amplitude_to_db(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    mel = mel.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(mel)
        probs = torch.softmax(logits, dim=1)
        ai_score = probs[0, 1].item()

    label = "AI Generated" if ai_score > 0.5 else "Human"
    confidence = ai_score if ai_score > 0.5 else 1 - ai_score

    return f"{label} | Confidence: {confidence:.3f}"

with gr.Blocks() as demo:

    gr.Markdown("# AI-Generated Voice Detector")
    gr.Markdown("""
    **Validation Accuracy:** 97.5%  
    **AUC:** 0.995  
    **Precision / Recall / F1:** 0.99  
    """)

    with gr.Tab("Live Demo"):
        audio_input = gr.Audio(type="filepath")
        output = gr.Textbox(label="Prediction")
        audio_input.change(predict, audio_input, output)

    with gr.Tab("Model Metrics"):
        gr.Image("confusion_matrix.png", label="Confusion Matrix")
        gr.Image("roc_curve.png", label="ROC Curve")
        gr.Markdown("""
        Dataset Size: 2000 samples  
        Architecture: CNN + Log-Mel Spectrogram  
        Training Device: Apple MPS  
        """)

if __name__ == "__main__":
    demo.launch()
