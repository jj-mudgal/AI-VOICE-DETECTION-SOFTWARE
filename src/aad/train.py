"""
Training script for AI Audio Detector
Uses explicit train / val / test folder splits.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils.collate import pad_sequence
from .config import EPOCHS, BATCH_SIZE, LEARNING_RATE, SEED
from .data import AudioDataset
from .model import AudioDetector

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

torch.manual_seed(SEED)

# ------------------------------------------------------------
# Checkpoints
# ------------------------------------------------------------
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def train_model():

    print("Starting training on:", DEVICE)

    # -------------------------
    # Load datasets
    # -------------------------
    train_dataset = AudioDataset(split="train", augment=True)
    val_dataset = AudioDataset(split="val", augment=False)
    test_dataset = AudioDataset(split="test", augment=False)

    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset empty.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pad_sequence,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_sequence,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_sequence,
    )

    # -------------------------
    # Model
    # -------------------------
    model = AudioDetector().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_val_acc = 0.0

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(EPOCHS):

        # -------- TRAIN --------
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for waveforms, labels, lengths in tqdm(train_loader):

            waveforms = waveforms.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * waveforms.size(0)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_acc = correct / total

        # -------- VALIDATION --------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for waveforms, labels, lengths in val_loader:
                waveforms = waveforms.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(waveforms)
                preds = torch.argmax(outputs, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print("Saved best model →", save_path)

    print("\nTraining complete.")
    print("Best validation accuracy:", best_val_acc)

    # -------------------------
    # Final Test Evaluation
    # -------------------------
    print("\nEvaluating on Test Set...")

    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt")))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for waveforms, labels, lengths in test_loader:
            waveforms = waveforms.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(waveforms)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    print("Final Test Accuracy:", test_acc)


if __name__ == "__main__":
    train_model()
