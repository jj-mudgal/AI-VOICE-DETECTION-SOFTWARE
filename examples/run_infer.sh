#!/usr/bin/env bash
# ============================================================
# AI Audio Detector - Example Inference Script
# ------------------------------------------------------------
# Run this script to test your trained model on an audio file.
#
# Usage:
#   bash examples/run_infer.sh path/to/audio.wav
# ============================================================

# Exit if any command fails
set -e

# Check argument
if [ $# -lt 1 ]; then
  echo "Usage: $0 <path_to_audio>"
  echo "Example: bash examples/run_infer.sh data/sample.wav"
  exit 1
fi

AUDIO_FILE=$1

# Optional: specify custom checkpoint
MODEL_PATH="checkpoints/best_model.pt"

# Check for Python module
if ! command -v python &> /dev/null
then
    echo "Python not found. Please install Python 3.10+ first."
    exit 1
fi

# Run inference
echo "🎧 Running inference on: $AUDIO_FILE"
echo "-------------------------------------------------"

python -m src.aad.infer \
  --audio "$AUDIO_FILE" \
  --model "$MODEL_PATH" \
  --threshold 0.5

echo "-------------------------------------------------"
echo "✅ Inference complete."
