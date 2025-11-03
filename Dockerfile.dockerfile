# ============================================================
# 1️⃣ Base Image — lightweight CUDA-enabled PyTorch runtime
# ============================================================
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime as base

# Set working directory
WORKDIR /app

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for audio + ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# ============================================================
# 2️⃣ Install dependencies
# ============================================================
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ============================================================
# 3️⃣ Copy application code
# ============================================================
COPY . .

# Expose FastAPI port
EXPOSE 8000

# ============================================================
# 4️⃣ Runtime environment variables
# ============================================================
ENV MODEL_NAME="facebook/wav2vec2-large-xlsr-53"
ENV APP_ENV="production"
ENV PYTHONPATH=/app

# ============================================================
# 5️⃣ Default command — run FastAPI app
# ============================================================
CMD ["uvicorn", "src.aad.api:app", "--host", "0.0.0.0", "--port", "8000"]
