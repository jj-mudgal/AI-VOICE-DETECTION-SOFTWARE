import torch
import gradio as gr
from .model_whisper import WhisperClassifier
from .audio_loader import load_audio

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "src/aad/checkpoints/best_whisper_epoch3.pt"

model = WhisperClassifier().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def predict(audio_file):
    if audio_file is None:
        return "No audio provided."

    audio_path = audio_file
    waveform, sr = load_audio(audio_path)
    waveform = waveform.to(DEVICE)

    with torch.no_grad():
        logits = model(waveform)
        probs = torch.softmax(logits, 1)
        ai_score = probs[0, 1].item()

    if ai_score > 0.5:
        return f"AI Generated | Confidence: {ai_score:.3f}"
    else:
        return f"Human | Confidence: {(1 - ai_score):.3f}"

ui = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload Audio"),
    outputs=gr.Textbox(label="Result"),
    title="Synthetic Voice Detector",
    description="Built by Janmejai Mudgal.",
)

if __name__ == "__main__":
    ui.launch()
