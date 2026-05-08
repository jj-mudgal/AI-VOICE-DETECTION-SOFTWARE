import os
import torch
from .model import AudioDetector
from .config import DEVICE


class InferenceEngine:
    def __init__(self, checkpoint_path: str = None, threshold: float = 0.5):
        self.model     = AudioDetector().to(DEVICE)
        self.device    = DEVICE
        self.threshold = threshold

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=DEVICE)
            )
            print(f"[InferenceEngine] Loaded weights from {checkpoint_path}")
        else:
            print("[InferenceEngine] No checkpoint found — using random weights")

        self.model.eval()

    @classmethod
    def from_checkpoint(cls, path: str, threshold: float = 0.5) -> "InferenceEngine":
        """Convenience constructor: InferenceEngine.from_checkpoint('checkpoints/best_model.pt')"""
        return cls(checkpoint_path=path, threshold=threshold)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        logits = self.model(x)
        return torch.softmax(logits, dim=-1)
