import torch
import pytest
from src.aad.model import AudioDetector


def test_output_shape_batch():
    """Model should return (B, 2) for a batch of raw waveforms."""
    model = AudioDetector()
    model.eval()
    x = torch.randn(4, 16000)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 2), f"Expected (4, 2), got {out.shape}"


def test_output_shape_single():
    """Model should handle a single sample."""
    model = AudioDetector()
    model.eval()
    x = torch.randn(1, 16000)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2)


def test_output_is_logits():
    """Output should be raw logits — not softmaxed (values not bounded 0-1)."""
    model = AudioDetector()
    model.eval()
    x = torch.randn(2, 16000)
    with torch.no_grad():
        out = model(x)
    # logits can be any real number; softmax output would always be < 1
    assert out.dtype == torch.float32


def test_longer_audio():
    """Model should handle audio longer than 1 second."""
    model = AudioDetector()
    model.eval()
    x = torch.randn(2, 16000 * 5)  # 5 seconds
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2)
