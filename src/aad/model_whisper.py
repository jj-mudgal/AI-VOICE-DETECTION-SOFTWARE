# src/aad/model_whisper.py

import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperFeatureExtractor
from .config import NUM_CLASSES, SAMPLE_RATE

WHISPER_NAME = "openai/whisper-small"  # or "openai/whisper-base"
TARGET_MEL_FRAMES = 3000  # whisper expects exactly 3000 time frames


class WhisperClassifier(nn.Module):
    def __init__(self, model_name: str = WHISPER_NAME, num_classes: int = NUM_CLASSES, freeze_encoder: bool = False):
        super().__init__()

        # Raw audio → log-mel processor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_name, sampling_rate=SAMPLE_RATE
        )

        # Use encoder only (no decoder required)
        self.encoder = WhisperModel.from_pretrained(model_name).encoder
        hidden_dim = self.encoder.config.d_model

        # Optional encoder freezing (useful on small GPUs)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Classification head
        self.project = nn.Linear(hidden_dim, 256)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, waveforms, lengths=None):
        """
        waveforms: Tensor (B, 1, T) or (B, T)
        """
        # Make shape (B, T)
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)

        # Convert to CPU numpy list (WhisperFeatureExtractor requirement)
        inputs = [w.cpu().numpy().astype("float32") for w in waveforms]

        # Extract log-mel features (may be shorter or longer than 3000)
        features = self.feature_extractor(
            inputs,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        mel = features["input_features"].to(waveforms.device)  # (B, 80, T)

        # ✅ Force mel frames exactly = 3000
        B, C, T = mel.shape
        if T < TARGET_MEL_FRAMES:
            pad = TARGET_MEL_FRAMES - T
            mel = torch.nn.functional.pad(mel, (0, pad), value=0.0)
        elif T > TARGET_MEL_FRAMES:
            mel = mel[:, :, :TARGET_MEL_FRAMES]

        # Encoder forward -> (B, 3000, H)
        enc_out = self.encoder(mel)
        hidden_states = enc_out.last_hidden_state

        # Mean-pool across time → (B, H)
        pooled = hidden_states.mean(dim=1)

        # Classification head
        x = self.project(pooled)
        logits = self.classifier(x)
        return logits
