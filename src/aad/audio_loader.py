import subprocess
import numpy as np
import torch

SAMPLE_RATE = 16000

def load_audio(path, sr=SAMPLE_RATE):
    cmd = [
        "ffmpeg", "-i", path,
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(sr),
        "-"
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    waveform = np.frombuffer(out, dtype=np.float32)
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    return waveform, sr

def validate_audio_bytes(file_bytes: bytes):
    # Basic header check (not extension-based)
    return len(file_bytes) > 100
