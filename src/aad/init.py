"""
AI Audio Detector — Next-Gen Edition
====================================

A modular pipeline for detecting AI-generated (synthetic) vs. human speech.
Built with PyTorch, Hugging Face Transformers, and FastAPI.

Modules:
--------
- config.py : Global configuration & environment variables.
- data.py   : Audio dataset loader with augmentation.
- model.py  : Deep neural network (Wav2Vec2 / CLAP / Whisper backbone).
- train.py  : Model training loop with checkpointing.
- infer.py  : Inference utilities for evaluating audio samples.
- api.py    : FastAPI-based web API for real-time prediction.

Author: Your Name
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "youremail@example.com"

# Export key components for convenience
from .model import AudioDetector
from .infer import infer
