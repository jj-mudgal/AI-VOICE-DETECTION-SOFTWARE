# AI Audio Detector — Next-Gen Edition

An **AI-generated speech detector** built using state-of-the-art deep learning technologies — leveraging **Wav2Vec2**, **CLAP embeddings**, and **transformer-based audio encoders** to distinguish between **synthetic (TTS/VC)** and **human** speech.

---

## Highlights

- **Transformer-based Audio Detection** — built on `torch`, `transformers`, and `torchaudio`.
- **Pretrained Embedding Support** — integrates `facebook/wav2vec2-base`, `openai/whisper-base`, or `laion/clap`.
- **Spectral & Temporal Features** — uses hybrid CNN + Transformer fusion architecture.
- **Inference CLI & Web API** — run as CLI, REST API, or containerized microservice.
- **Continuous Integration** — preconfigured GitHub Actions for tests, linting, and Docker build.

---

## Model Overview

| Component | Description |
|------------|-------------|
| **Feature Extractor** | Wav2Vec2 / CLAP embeddings |
| **Classifier Head** | Transformer + attention pooling |
| **Loss** | Weighted cross-entropy |
| **Optimizer** | AdamW with cosine decay |

---

## 🔧 Installation

```bash
# Clone repo
git clone https://github.com/yourname/ai-audio-detector.git
cd ai-audio-detector

# Create environment
python -m venv venv
source venv/bin/activate

# Install deps
pip install -r requirements.txt
