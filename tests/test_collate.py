import torch
import pytest
from src.aad.utils.collate import pad_sequence


def make_batch(lengths):
    """Helper — build a fake batch with variable-length waveforms."""
    return [
        (torch.randn(length), torch.tensor(i % 2))
        for i, length in enumerate(lengths)
    ]


def test_output_shape():
    """Padded batch should be (B, 1, T_max)."""
    batch = make_batch([8000, 12000, 16000])
    waveforms, labels, lengths = pad_sequence(batch)
    assert waveforms.shape == (3, 1, 16000)


def test_labels_dtype():
    """Labels tensor should be long (int64)."""
    batch = make_batch([8000, 8000])
    _, labels, _ = pad_sequence(batch)
    assert labels.dtype == torch.long


def test_lengths_correct():
    """Lengths should match original waveform sizes."""
    sizes = [4000, 8000, 16000]
    batch = make_batch(sizes)
    _, _, lengths = pad_sequence(batch)
    assert lengths.tolist() == sizes


def test_padding_is_zero():
    """Shorter sequences should be zero-padded at the end."""
    batch = make_batch([4000, 8000])
    waveforms, _, _ = pad_sequence(batch)
    # last 4000 samples of the first (shorter) waveform should be zero
    assert waveforms[0, 0, 4000:].sum().item() == 0.0
