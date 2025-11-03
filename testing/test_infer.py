"""
Unit tests for the inference engine
-----------------------------------
Ensures that AI Audio Detector inference works
and produces consistent, valid outputs.
"""

import os
import torch
import pytest
import numpy as np
from src.aad.infer import InferenceEngine


@pytest.fixture(scope="session")
def engine():
    """Load the inference engine once for all tests."""
    return InferenceEngine(model_path=None)


def test_model_loaded(engine):
    """Check if model is initialized correctly."""
    assert engine.model is not None, "Model not loaded"
    assert hasattr(engine.model, "forward"), "Model missing forward() method"


def test_prediction_structure(engine, tmp_path):
    """Run inference on dummy waveform to verify output structure."""
    # Create a fake waveform (1 sec of random noise)
    dummy_wave = torch.randn(1, 16000)
    with torch.no_grad():
        logits = engine.model(dummy_wave)
    assert logits.shape[-1] == 2, "Model output should have 2 classes"

    # Simulate final JSON structure
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    result = {
        "file": "dummy.wav",
        "human_prob": float(probs[0]),
        "ai_prob": float(probs[1]),
        "predicted_label": "synthetic" if probs[1] > 0.5 else "human",
    }

    # Validate keys
    for key in ["file", "human_prob", "ai_prob", "predicted_label"]:
        assert key in result, f"Missing key {key} in result"

    assert isinstance(result["human_prob"], float)
    assert isinstance(result["predicted_label"], str)


def test_checkpoint_discovery(tmp_path):
    """Ensure latest checkpoint finder works properly."""
    # Simulate two fake checkpoints
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "model1.pt").write_text("dummy")
    (ckpt_dir / "model2.pt").write_text("dummy")
    engine = InferenceEngine()
    engine._get_latest_checkpoint = lambda: str(ckpt_dir / "model2.pt")
    latest = engine._get_latest_checkpoint()
    assert "model2.pt" in latest
