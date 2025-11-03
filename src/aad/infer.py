"""
Inference module for AI Audio Detector
--------------------------------------
Loads a trained model and predicts whether an audio
clip is human or AI-generated.

Supports:
- Command-line usage
- Integration into FastAPI or web apps
"""

import os
import torch
import librosa
import numpy as np
from torch.nn import functional as F
from .model import AudioDetector
from .config import MODEL_NAME, SAMPLE_RATE, DEVICE, OUTPUT_DIR


class InferenceEngine:
    """Encapsulates model loading and single-file prediction."""

    def __init__(self, model_path: str = None, threshold: float = 0.5):
        """
        Args:
            model_path (str): Path to a saved model checkpoint (.pt)
            threshold (float): Probability threshold for classification
        """
        self.model_path = model_path or self._get_latest_checkpoint()
        self.threshold = threshold
        self.model = AudioDetector(pretrained_model=MODEL_NAME)
        self.model.to(DEVICE)
        self.model.eval()

        if self.model_path and os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
            print(f"✅ Loaded model weights from {self.model_path}")
        else:
            print("⚠️ No model checkpoint found — using base pretrained encoder.")

    def _get_latest_checkpoint(self):
        """Finds the most recent model checkpoint in OUTPUT_DIR."""
        if not os.path.exists(OUTPUT_DIR):
            return None
        files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".pt")]
        if not files:
            return None
        files.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
        return os.path.join(OUTPUT_DIR, files[0])

    def predict(self, filepath: str):
        """
        Predicts whether an audio file is AI-generated or human.

        Returns:
            dict: {
                "file": <filename>,
                "human_prob": float,
                "ai_prob": float,
                "predicted_label": "human" | "synthetic"
            }
        """
        waveform, sr = librosa.load(filepath, sr=SAMPLE_RATE)
        waveform = librosa.util.normalize(waveform)
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(waveform)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        result = {
            "file": os.path.basename(filepath),
            "human_prob": float(probs[0]),
            "ai_prob": float(probs[1]),
            "predicted_label": "synthetic" if probs[1] >= self.threshold else "human",
        }

        print(
            f"🎧 {result['file']}: Human={result['human_prob']:.3f} | AI={result['ai_prob']:.3f} → {result['predicted_label']}"
        )
        return result


# ------------------------------------------------------------
# Command-line usage
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Audio Detector Inference")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (.wav, .flac, .mp3)")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    args = parser.parse_args()

    engine = InferenceEngine(model_path=args.model, threshold=args.threshold)
    engine.predict(args.audio)
