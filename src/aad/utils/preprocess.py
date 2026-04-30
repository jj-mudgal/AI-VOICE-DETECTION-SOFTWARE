import torch

def normalize_waveform(x: torch.Tensor):
    return (x - x.mean()) / (x.std() + 1e-6)
