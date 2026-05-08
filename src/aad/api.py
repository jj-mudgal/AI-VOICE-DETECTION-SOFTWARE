"""
FastAPI interface for AI Audio Detector
---------------------------------------
Run:
    uvicorn src.aad.api:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import os
import torch
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torch.nn import functional as F

from .infer import InferenceEngine
from .config import SAMPLE_RATE, DEVICE

# ------------------------------------------------------------
# App Setup
# ------------------------------------------------------------
app = FastAPI(
    title="AI Audio Detector API",
    description="Detect whether an audio clip is AI-generated or human.",
    version="1.0.0",
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load inference engine once at startup
engine = InferenceEngine(
    checkpoint_path=os.path.join("checkpoints", "best_model.pt")
    if os.path.exists(os.path.join("checkpoints", "best_model.pt"))
    else None
)


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.get("/")
def home():
    return {"message": "AI Audio Detector API is running!"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "AudioDetector",
        "device": DEVICE,
    }


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".flac", ".mp3", ".ogg", ".m4a")):
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    contents = await file.read()
    audio_bytes = io.BytesIO(contents)

    try:
        waveform, sr = librosa.load(audio_bytes, sr=SAMPLE_RATE, mono=True)
        waveform = librosa.util.normalize(waveform)
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading audio: {e}")

    probs = engine.predict(waveform).cpu().numpy()[0]

    return {
        "filename": file.filename,
        "human_prob": float(probs[0]),
        "ai_prob": float(probs[1]),
        "predicted_label": "synthetic" if probs[1] >= engine.threshold else "human",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.aad.api:app", host="0.0.0.0", port=8000, reload=True)
