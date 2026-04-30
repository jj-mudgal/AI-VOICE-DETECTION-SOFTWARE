import torch

class InferenceEngine:
    def __init__(self, model, device="cpu", threshold=0.5):
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold
        self.model.eval()

    @torch.no_grad()
    def predict(self, x):
        x = x.to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)
        return probs
