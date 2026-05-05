import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperFeatureExtractor
from .config import NUM_CLASSES, SAMPLE_RATE

WHISPER_NAME       = "openai/whisper-small"
TARGET_MEL_FRAMES  = 3000


class WhisperClassifier(nn.Module):
    def __init__(
        self,
        model_name: str  = WHISPER_NAME,
        num_classes: int = NUM_CLASSES,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_name, sampling_rate=SAMPLE_RATE
        )

        self.encoder = WhisperModel.from_pretrained(model_name).encoder
        hidden_dim   = self.encoder.config.d_model

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.project    = nn.Linear(hidden_dim, 256)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)

        inputs = [w.cpu().numpy().astype("float32") for w in waveforms]

        features = self.feature_extractor(
            inputs,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        mel = features["input_features"].to(waveforms.device)

        B, C, T = mel.shape
        if T < TARGET_MEL_FRAMES:
            mel = torch.nn.functional.pad(mel, (0, TARGET_MEL_FRAMES - T))
        else:
            mel = mel[:, :, :TARGET_MEL_FRAMES]

        hidden_states = self.encoder(mel).last_hidden_state
        pooled        = hidden_states.mean(dim=1)
        return self.classifier(self.project(pooled))
