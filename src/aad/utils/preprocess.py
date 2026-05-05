import torch


def normalize_waveform(x: torch.Tensor) -> torch.Tensor:
    """Zero-mean, unit-variance normalization with epsilon guard."""
    return (x - x.mean()) / (x.std() + 1e-6)


def pad_or_trim(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad with zeros or trim to target_len along last axis."""
    T = x.shape[-1]
    if T < target_len:
        pad = target_len - T
        x = torch.nn.functional.pad(x, (0, pad))
    else:
        x = x[..., :target_len]
    return x
