# AI Audio Detector — Next-Gen Edition

> Detect whether a voice is human or AI-generated — powered by **Wav2Vec2**, **CLAP embeddings**, and a **CNN + Log-Mel Spectrogram** architecture trained to distinguish synthetic (TTS/VC) from real human speech.

{paste image — screenshot of the Gradio web UI open in a browser, showing an audio upload widget on the left and a prediction result like "AI Generated | Confidence: 0.981" on the right}

---

## ⏣ Highlights

- **Transformer-based Audio Detection** — built on `torch`, `transformers`, and `torchaudio`
- **Pretrained Embedding Support** — integrates `facebook/wav2vec2-base`, `openai/whisper-base`, or `laion/clap`
- **CNN + Log-Mel Spectrogram Pipeline** — lightweight, fast, and accurate on 2000-sample datasets
- **Whisper Classifier variant** — encoder-only Whisper with attention pooling + classification head
- **Inference CLI & Web API** — run as Gradio demo, FastAPI REST service, or containerized microservice
- **Audio Augmentation** — Gaussian noise, time stretch, pitch shift, and random shifts via `audiomentations`
- **Continuous Integration** — preconfigured GitHub Actions for tests, linting, and Docker build
- **Validation Accuracy: 97.5% | AUC: 0.995 | F1: 0.99**

---

## ⏣ Model Overview

| Component | Description |
|---|---|
| **Feature Extractor** | Log-Mel Spectrogram (128 mels, 1024 FFT) |
| **Primary Backbone** | 4-layer CNN with BatchNorm + AdaptiveAvgPool |
| **Alt Backbone** | Whisper encoder (openai/whisper-small) with mean pooling |
| **Classifier Head** | Linear → ReLU → Dropout → Linear (2 classes) |
| **Loss** | Cross-entropy (weighted) |
| **Optimizer** | AdamW with weight decay 1e-4 |
| **Training Device** | Apple MPS / CUDA / CPU (auto-detected) |

---

## ⏣ Architecture

```
Raw Audio (.wav / .mp3 / .flac / .ogg / .m4a)
    |
    |  librosa / torchaudio load + resample → 16kHz mono
    v
Preprocessing
    |-- Log-Mel Spectrogram  (128 mels, n_fft=1024, hop=512)
    |-- AmplitudeToDB
    `-- Per-sample normalization  (mean=0, std=1)
    |
    v
CNN Backbone (AudioDetector)
    |-- Conv2d(1 → 64)  + BN + ReLU + MaxPool2d
    |-- Conv2d(64 → 128) + BN + ReLU + MaxPool2d
    |-- Conv2d(128 → 256) + BN + ReLU + MaxPool2d
    |-- Conv2d(256 → 256) + BN + ReLU + MaxPool2d
    `-- AdaptiveAvgPool2d(1,1) → Flatten
    |
    v
Classifier Head
    |-- Linear(256 → 128) + ReLU + Dropout(0.4)
    `-- Linear(128 → 2)
    |
    v
Softmax → { human_prob, ai_prob }
    `-- Label: "Human" or "AI Generated"
```

{paste image — architecture diagram showing the CNN pipeline from raw audio → mel spectrogram → convolutional layers → classifier head → output label, OR a screenshot of your terminal showing training logs with epoch/loss/accuracy printed}

---

## ⏣ Whisper Variant Architecture

An alternate backbone using OpenAI's Whisper encoder for richer audio representations:

```
Raw Audio (B, T)
    |
    v
WhisperFeatureExtractor  →  Log-Mel (B, 80, 3000)
    |
    v
Whisper Encoder (openai/whisper-small)
    |
    v
last_hidden_state  →  Mean Pool across time  →  (B, H)
    |
    v
Linear(H → 256) + ReLU + Dropout(0.25) + Linear(256 → 2)
    |
    v
Logits → Predicted Label
```

---

## ⏣ Tech Stack

| Layer | Technology |
|---|---|
| Core ML | PyTorch, torchaudio, transformers |
| Feature Extraction | librosa, Wav2Vec2, Whisper, CLAP |
| Augmentation | audiomentations |
| Web Demo | Gradio |
| REST API | FastAPI + Uvicorn |
| Data & Metrics | scikit-learn, matplotlib, numpy |
| CI/CD | GitHub Actions |
| Containerization | Docker |

---

## ⏣ Project Structure

```
|-- app.py                          # Gradio web demo
|-- requirements.txt
|-- confusion_matrix.png
|-- roc_curve.png
|
|-- src/
|   |-- __init__.py
|   `-- aad/
|       |-- config.py               # Env vars, hyperparams, device selection
|       |-- data.py                 # AudioDataset: load, augment, mel transform
|       |-- model.py                # CNN-based AudioDetector
|       |-- model_whisper.py        # WhisperClassifier (encoder-only)
|       |-- train.py                # Training loop (train/val/test splits)
|       |-- infer.py                # Inference engine: load model + predict
|       |-- api.py                  # FastAPI REST endpoint (/predict)
|       |-- audio_loader.py         # ffmpeg-based raw audio loader
|       `-- utils/
|           `-- collate.py          # DataLoader collate fn (variable-length padding)
|
|-- data/
|   |-- train/
|   |   |-- human/
|   |   `-- synthetic/
|   |-- val/
|   `-- test/
|
|-- checkpoints/
|   `-- best_model.pt
|
`-- .github/
    |-- workflows/ci.yaml
    |-- PULL_REQUEST.md
    `-- ISSUE_TEMPLATE/
        |-- bugreport.yaml
        `-- FEATURE_REQUEST.md
```

---

## ⏣ Getting Started

### Prerequisites

- Python 3.10+
- `ffmpeg` installed on system (for `audio_loader.py`)
- Apple Silicon (MPS), CUDA GPU, or CPU

### Install

```bash
# Clone the repo
git clone https://github.com/your-username/ai-audio-detector.git
cd ai-audio-detector

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Prepare Data

Organise your audio files like this:

```
data/
  train/
    human/      ← real human speech clips
    synthetic/  ← TTS / voice-cloned clips
  val/
    human/
    synthetic/
  test/
    human/
    synthetic/
```

Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`

### Train

```bash
python -m src.aad.train
```

Best checkpoint is automatically saved to `checkpoints/best_model.pt`.

{paste image — terminal screenshot showing training output: epoch number, loss, train accuracy, val accuracy printed per epoch, and the "Saved best model" line}

### Run Gradio Demo

```bash
python app.py
```

Open `http://localhost:7860` — upload any audio file and get an instant prediction.

{paste image — two-tab Gradio UI: first tab showing the audio upload widget with a prediction result, second tab showing the confusion matrix and ROC curve images}

### Run FastAPI Server

```bash
uvicorn src.aad.api:app --host 0.0.0.0 --port 8000 --reload
```

#### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample.wav"
```

#### Example response

```json
{
  "filename": "sample.wav",
  "human_prob": 0.031,
  "ai_prob": 0.969,
  "predicted_label": "synthetic"
}
```

---

## ⏣ API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Upload audio file → returns label + probabilities |

**Accepted formats:** `.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`

**Response schema:**

```json
{
  "filename": "string",
  "human_prob": 0.0,
  "ai_prob": 0.0,
  "predicted_label": "human | synthetic"
}
```

---

## ⏣ Model Metrics

| Metric | Score |
|---|---|
| Validation Accuracy | **97.5%** |
| AUC | **0.995** |
| Precision | **0.99** |
| Recall | **0.99** |
| F1 Score | **0.99** |
| Dataset Size | 2000 samples |
| Training Device | Apple MPS |

{paste image — confusion matrix plot (confusion_matrix.png) showing true vs predicted labels for Human and Synthetic classes}

{paste image — ROC curve plot (roc_curve.png) showing AUC = 0.995}

---

## ⏣ Configuration

All parameters are configurable via environment variables or a `.env` file:

```env
MODEL_NAME=cnn_raw_waveform
SAMPLE_RATE=16000
NUM_CLASSES=2
DATA_DIR=data/
OUTPUT_DIR=checkpoints/
LOG_DIR=logs/
EPOCHS=35
BATCH_SIZE=16
LEARNING_RATE=1e-4
SEED=42
USE_CUDA=1
```

Device is auto-selected: **MPS → CUDA → CPU** (in that priority order).

---

## ⏣ Run with Docker

```bash
docker build -t ai-audio-detector:latest .
docker run -p 8000:8000 ai-audio-detector:latest
```

CI also auto-builds Docker on every push to `main` via GitHub Actions.

---

## ⏣ CI / CD

GitHub Actions pipeline runs on every push and pull request to `main` / `dev`:

| Step | Tool |
|---|---|
| Linting | `flake8` |
| Formatting | `black`, `isort` |
| Unit Tests | `pytest` |
| Docker Build | Runs on `main` only, after tests pass |

---

## ⏣ Roadmap

- [x] CNN + Log-Mel Spectrogram classifier
- [x] Whisper encoder classifier
- [x] Audio augmentation pipeline
- [x] Gradio web demo
- [x] FastAPI REST API
- [x] CI/CD with GitHub Actions
- [ ] Wav2Vec2 / CLAP embedding fine-tuning
- [ ] Ensemble: CNN + Whisper + Wav2Vec2 fusion
- [ ] Docker sandbox with GPU support
- [ ] Dataset expansion beyond 2000 samples
- [ ] Real-time streaming inference
- [ ] Hugging Face Hub model card + deployment

---

## ⏣ Security Notes

- CORS is currently set to `origin: "*"` — replace with your actual frontend domain before deploying
- Code execution sandbox not applicable here, but model inference has no external side effects
- **TODO:** Add authentication for the `/predict` endpoint in production deployments

---

## ⏣ Contributing

Contributions are welcome — bugs, features, or documentation improvements.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Format your code (`black src tests`, `isort src tests`)
4. Run tests (`pytest -q`)
5. Open a pull request using the provided PR template

See `CONTRIBUTING.md` for full guidelines.

---

## ⏣ License

MIT © 2025 Janmejai Mudgal
