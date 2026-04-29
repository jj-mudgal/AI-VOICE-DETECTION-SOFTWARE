import torch
from src.aad.model_whisper import WhisperClassifier

def test_output_shape():
    model = WhisperClassifier()
    x = torch.randn(2, 80, 3000)
    out = model(x)
    assert out.shape[0] == 2
