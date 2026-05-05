import torch
from .model import AudioDetector
from .config import DEVICE


class InferenceEngine:
    def __init__(self, checkpoint_path: str = None, threshold: float = 0.5):
        self.model = AudioDetector().to(DEVICE)
        self.device = DEVICE
        self.threshold = threshold

        if checkpoint_path:
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=DEVICE)
            )

        self.model.eval()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        logits = self.model(x)
        return torch.softmax(logits, dim=-1)
