import torch
import pytest
from src.aad.infer import InferenceEngine


def test_engine_loads_without_checkpoint():
    """Engine should initialise cleanly with no checkpoint."""
    engine = InferenceEngine(checkpoint_path=None)
    assert engine.model is not None
    assert engine.threshold == 0.5


def test_predict_output_shape():
    """predict() should return (B, 2) probability tensor."""
    engine = InferenceEngine()
    x = torch.randn(1, 16000)
    probs = engine.predict(x)
    assert probs.shape == (1, 2)


def test_predict_sums_to_one():
    """Softmax output should sum to ~1.0 per sample."""
    engine = InferenceEngine()
    x = torch.randn(3, 16000)
    probs = engine.predict(x)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(3), atol=1e-5)


def test_from_checkpoint_classmethod(tmp_path):
    """from_checkpoint() with missing file should still load (random weights)."""
    fake_path = str(tmp_path / "no_model.pt")
    engine = InferenceEngine.from_checkpoint(fake_path)
    assert engine.model is not None
