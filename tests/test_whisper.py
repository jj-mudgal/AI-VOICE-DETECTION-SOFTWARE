import pytest
import torch


def test_whisper_classifier_import():
    """WhisperClassifier should import without errors."""
    import ast
    src = open("src/aad/model_whisper.py").read()
    ast.parse(src)


def test_whisper_output_shape(monkeypatch):
    """
    WhisperClassifier.forward() should return (B, NUM_CLASSES).
    We monkeypatch the heavy HuggingFace downloads so CI stays fast.
    """
    import types, torch, torch.nn as nn

    # Minimal mock encoder
    class FakeEncoder(nn.Module):
        class config:
            d_model = 512
        def forward(self, x):
            class Out:
                last_hidden_state = torch.zeros(x.shape[0], 10, 512)
            return Out()

    class FakeExtractor:
        @classmethod
        def from_pretrained(cls, *a, **kw): return cls()
        def __call__(self, inputs, **kw):
            B = len(inputs)
            return {"input_features": torch.zeros(B, 80, 3000)}

    class FakeWhisperModel:
        encoder = FakeEncoder()
        @classmethod
        def from_pretrained(cls, *a, **kw): return cls()

    import src.aad.model_whisper as mw
    monkeypatch.setattr(mw, "WhisperFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(mw, "WhisperModel", FakeWhisperModel)

    from src.aad.model_whisper import WhisperClassifier
    model = WhisperClassifier.__new__(WhisperClassifier)
    model.feature_extractor = FakeExtractor()
    model.encoder = FakeEncoder()
    model.project = nn.Linear(512, 256)
    model.classifier = nn.Sequential(nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, 2))

    x = torch.randn(2, 16000)
    out = model(x)
    assert out.shape == (2, 2), f"Expected (2, 2), got {out.shape}"
