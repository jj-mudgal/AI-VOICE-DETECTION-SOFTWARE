"""
Training script for AI Audio Detector
-------------------------------------
Handles data loading, training loop, validation,
and checkpoint saving for the AudioDetector model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .utils.collate import pad_sequence
from .config import EPOCHS, BATCH_SIZE, LEARNING_RATE, SEED
from .data import AudioDataset
from .model import AudioDetector

# ------------------------------------------------------------
# Device Setup (Apple Silicon M1/M2/M3/M4)
# ------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Using MPS (Apple Silicon GPU)")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ MPS not available, using CPU")

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
torch.manual_seed(SEED)

# ------------------------------------------------------------
# Ensure checkpoint directory exists
# ------------------------------------------------------------
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Training function
# ------------------------------------------------------------
def train_model():
    print(f"🚀 Starting training on device: {DEVICE}")

    # Load dataset
    dataset = AudioDataset(augment=True)
    total_len = len(dataset)
    if total_len == 0:
        raise RuntimeError("No audio samples found in your dataset.")
    val_len = int(0.2 * total_len)
    train_len = total_len - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    # DataLoaders with collate_fn
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=pad_sequence,
        num_workers=2,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_sequence,
        num_workers=1,
        pin_memory=False,
    )

    # Model, loss, optimizer
    model = AudioDetector().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Mixed precision for MPS
    scaler = torch.amp.GradScaler() if DEVICE.type == "mps" else None

    best_val_acc = 0.0

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, train_correct, train_total = 0.0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]", leave=False)

        for batch in progress_bar:
            waveforms, labels, lengths = batch
            waveforms, labels = waveforms.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast(device_type=DEVICE.type):
                    outputs = model(waveforms)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item()) * waveforms.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += int((preds == labels).sum().item())
            train_total += int(labels.size(0))
            progress_bar.set_postfix(loss=float(loss.item()))

        train_loss = running_loss / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                waveforms, labels, lengths = batch
                waveforms, labels = waveforms.to(DEVICE), labels.to(DEVICE)
                outputs = model(waveforms)
                _, preds = torch.max(outputs, 1)
                val_correct += int((preds == labels).sum().item())
                val_total += int(labels.size(0))

        val_acc = val_correct / max(1, val_total)
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(CHECKPOINT_DIR, f"best_model_epoch{epoch}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved new best model to {save_path}")

    print("✅ Training complete.")


if __name__ == "__main__":
    train_model()
