"""
FastAPI interface for AI Audio Detector
---------------------------------------
Exposes REST endpoints for detecting whether
an uploaded audio file is AI-generated or human.

Run:
    uvicorn src.aad.api:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import torch
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torch.nn import functional as F
from .model import AudioDetector
from .config import MODEL_NAME, SAMPLE_RATE, DEVICE, OUTPUT_DIR
from .infer import InferenceEngine

# ------------------------------------------------------------
# App Setup
# ------------------------------------------------------------
app = FastAPI(
    title="AI Audio Detector API",
    description="Detect whether an audio clip is AI-generated or human.",
    version="1.0.0",
)

# Enable CORS (so a web frontend can connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load inference engine at startup
engine = InferenceEngine()

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.get("/")
def home():
    """Root endpoint for API health check."""
    return {"message": "AI Audio Detector API is running!"}


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """
    Predict whether uploaded audio is AI-generated or human.
    Accepts: .wav, .flac, .mp3, .ogg, .m4a
    Returns: JSON with probabilities and label
    """
    if not file.filename.lower().endswith((".wav", ".flac", ".mp3", ".ogg", ".m4a")):
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    # Read file into memory
    contents = await file.read()
    audio_bytes = io.BytesIO(contents)

    # Load and preprocess
    try:
        waveform, sr = librosa.load(audio_bytes, sr=SAMPLE_RATE)
        waveform = librosa.util.normalize(waveform)
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading audio: {e}")

    # Model inference
    with torch.no_grad():
        logits = engine.model(waveform)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    result = {
        "filename": file.filename,
        "human_prob": float(probs[0]),
        "ai_prob": float(probs[1]),
        "predicted_label": "synthetic" if probs[1] >= engine.threshold else "human",
    }

    return result


# ------------------------------------------------------------
# Run with: uvicorn src.aad.api:app --port 8000
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.aad.api:app", host="0.0.0.0", port=8000, reload=True)
