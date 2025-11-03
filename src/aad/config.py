"""
Configuration module for the AI Audio Detector.
------------------------------------------------
Handles environment variables, default parameters,
and runtime device selection.
"""

import os
from dotenv import load_dotenv
import torch

# ------------------------------------------------------------
# Load environment variables from .env file (if present)
# ------------------------------------------------------------
load_dotenv()

# ------------------------------------------------------------
# Core Configuration
# ------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "facebook/wav2vec2-large-xlsr-53")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 2))

# Directories
DATA_DIR = os.getenv("DATA_DIR", "data/")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "checkpoints/")
LOG_DIR = os.getenv("LOG_DIR", "logs/")

# Training Hyperparameters
EPOCHS = int(os.getenv("EPOCHS", 10))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-5))

# ------------------------------------------------------------
# Device Selection (GPU / CPU)
# ------------------------------------------------------------
if torch.cuda.is_available() and os.getenv("USE_CUDA", "1") == "1":
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# ------------------------------------------------------------
# Miscellaneous
# ------------------------------------------------------------
SEED = int(os.getenv("SEED", 42))
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------
