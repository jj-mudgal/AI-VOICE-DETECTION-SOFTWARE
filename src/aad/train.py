import torch
from src.aad.model import AudioDetector
from src.aad.config import Config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    model = AudioDetector(dropout=Config.DROPOUT)

    print(f"Model parameters: {count_parameters(model):,}")

    # dummy forward (sanity check)
    x = torch.randn(2, 16000)
    y = model(x)

    print("Output shape:", y.shape)


if __name__ == "__main__":
    main()
