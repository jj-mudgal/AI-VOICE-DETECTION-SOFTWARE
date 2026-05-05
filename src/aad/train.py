import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.aad.model import AudioDetector
from src.aad.config import Config, OUTPUT_DIR
from src.aad.data import get_dataloaders
from src.aad.eval import evaluate
from src.aad.threshold import find_best_threshold


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += x.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += x.size(0)

    return total_loss / total, correct / total


def main():
    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device: {device}")

    train_loader, val_loader = get_dataloaders(
        data_dir="data/train",
        batch_size=Config.BATCH_SIZE
    )

    model     = AudioDetector(dropout=Config.DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    print(f"Parameters: {count_parameters(model):,}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            print(f"  ✓ Saved best model (val_acc={val_acc:.3f})")


if __name__ == "__main__":
    main()


# --- early stopping re-export so the import resolves cleanly ---
from src.aad.utils.early_stopping import EarlyStopping  # noqa: F401
