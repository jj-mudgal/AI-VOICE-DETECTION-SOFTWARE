# src/aad/utils/collate.py
import torch
from typing import List, Tuple

def pad_sequence(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_value: float = 0.0):
    """
    Collate fn for DataLoader.

    batch: list of (waveform, label) where waveform is Tensor (1, T) or (T,)
    Returns:
        waveforms: Tensor of shape (B, 1, T_max)
        labels: Tensor of shape (B,)
        lengths: Tensor of original lengths (B,)
    """
    # normalize shapes: make each waveform a 1-D tensor
    waveforms = []
    for b in batch:
        w = b[0]
        if w.dim() == 2 and w.size(0) == 1:
            w = w.squeeze(0)         # (T,)
        elif w.dim() == 3 and w.size(1) == 1:
            w = w.squeeze(1).squeeze(0)
        elif w.dim() == 1:
            pass
        else:
            # fallback: flatten leading dims
            w = w.view(-1)
        waveforms.append(w)

    labels = torch.stack([b[1] for b in batch]).long()
    lengths = torch.tensor([w.shape[-1] for w in waveforms], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded = torch.stack(
        [torch.nn.functional.pad(w, (0, max_len - w.shape[-1]), value=pad_value) for w in waveforms]
    )  # (B, T_max)

    # model code expects (B, 1, T_max)
    padded = padded.unsqueeze(1)

    return padded, labels, lengths
