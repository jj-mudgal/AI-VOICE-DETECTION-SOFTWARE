"""
Model definition for AI Audio Detector
--------------------------------------
Implements a transformer-based classifier using a pre-trained
Wav2Vec2 encoder (from Hugging Face Transformers).
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from .config import MODEL_NAME, NUM_CLASSES


class AudioDetector(nn.Module):
    """
    AudioDetector — Wav2Vec2-based classification model.

    Architecture:
    -------------
    - Pretrained Wav2Vec2 encoder (frozen or fine-tuned)
    - Linear projection + small Transformer Encoder for temporal modeling
    - Classification head (binary or multi-class)
    """

    def __init__(self, pretrained_model: str = MODEL_NAME, num_classes: int = NUM_CLASSES, freeze_encoder: bool = False):
        super().__init__()

        # Load pretrained encoder
        self.encoder = Wav2Vec2Model.from_pretrained(pretrained_model)

        # Optionally freeze encoder layers (useful for small datasets)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size

        # Projection + Transformer block
        self.project = nn.Linear(hidden_size, 256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Audio waveform tensor (batch_size, time)

        Returns:
            Tensor: Logits (batch_size, num_classes)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Pass through Wav2Vec2 encoder
        outputs = self.encoder(x, attention_mask=None)
        hidden_states = outputs.last_hidden_state  # (B, T, H)

        # Project + transformer
        projected = self.project(hidden_states)
        transformed = self.transformer(projected)

        # Mean pooling across time
        pooled = transformed.mean(dim=1)

        # Classification
        logits = self.classifier(pooled)
        return logits


# ------------------------------------------------------------
# Debug utility (run independently)
# ------------------------------------------------------------
if __name__ == "__main__":
    model = AudioDetector()
    dummy_audio = torch.randn(1, 16000 * 3)  # 3-second clip
    logits = model(dummy_audio)
    print(f"Output shape: {logits.shape}")
