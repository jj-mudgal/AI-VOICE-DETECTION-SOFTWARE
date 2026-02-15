import torch
import torchaudio
from .model_whisper import WhisperClassifier
from .config import SAMPLE_RATE, NUM_CLASSES

LABELS = ["Human Voice", "AI-Generated Voice"][:NUM_CLASSES]

def load_model(model_path, device=torch.device("cpu")):
    """Load the WhisperClassifier with given checkpoint."""
    model = WhisperClassifier()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def infer(file_path, model, device=torch.device("cpu")):
    """
    Infer if an audio file is human or AI-generated.
    
    Args:
        file_path: str, path to audio file
        model: WhisperClassifier
        device: torch.device
    Returns:
        str: Prediction label
    """
    try:
        waveform, sr = torchaudio.load(file_path)
    except Exception as e:
        return f"Error loading audio: {e}"

    # Resample if necessary
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    # Ensure waveform has shape (1, T)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Move to device
    waveform = waveform.to(device)

    # Model inference
    try:
        with torch.no_grad():
            logits = model(waveform)
            pred = torch.argmax(logits, dim=1).item()
        return LABELS[pred]
    except Exception as e:
        return f"Error during model inference: {e}"
