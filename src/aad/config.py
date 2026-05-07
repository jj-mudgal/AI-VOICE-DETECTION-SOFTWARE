from dotenv import load_dotenv
import os
import torch

load_dotenv()

SAMPLE_RATE   = int(os.getenv("SAMPLE_RATE", 16000))
NUM_CLASSES   = int(os.getenv("NUM_CLASSES", 2))
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", 16))
EPOCHS        = int(os.getenv("EPOCHS", 35))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-4))
OUTPUT_DIR    = os.getenv("OUTPUT_DIR", "checkpoints/")
MODEL_NAME    = os.getenv("MODEL_NAME", "cnn_mel")

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)


class Config:
    SAMPLE_RATE = SAMPLE_RATE
    BATCH_SIZE  = BATCH_SIZE
    LR          = LEARNING_RATE
    EPOCHS      = EPOCHS
    DROPOUT     = 0.4
