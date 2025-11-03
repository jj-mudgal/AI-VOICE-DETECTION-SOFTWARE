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
from .config import (
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    DEVICE,
    OUTPUT_DIR,
    SEED,
)
from .data import AudioDataset
from .model import AudioDetector

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def train_model():
    print(f"🚀 Starting training on device: {DEVICE}")

    # Dataset
    dataset = AudioDataset(augment=True)
    total_len = len(dataset)
    val_len = int(0.2 * total_len)
    train_len = total_len - val_len

    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # Model, loss, optimizer
    model = AudioDetector().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    # ------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]", leave=False)

        for waveforms, labels in progress_bar:
            waveforms, labels = waveforms.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * waveforms.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            progress_bar.set_postfix(loss=loss.item())

        train_loss = running_loss / train_total
        train_acc = train_correct / train_total

        # ------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms, labels = waveforms.to(DEVICE), labels.to(DEVICE)
                outputs = model(waveforms)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}")

        # ------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(OUTPUT_DIR, f"best_model_epoch{epoch}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved new best model to {save_path}")

    print("✅ Training complete.")


if __name__ == "__main__":
    train_model()
