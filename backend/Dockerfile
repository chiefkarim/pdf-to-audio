FROM python:3.11-slim

# Install uv (fast pip replacement)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# System deps: tesseract, poppler (pdf2image), ffmpeg (pydub), libsndfile (soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Cache paths for HuggingFace models (set before any pre-bake RUN so both
# build-time downloads and runtime inference use the same location)
ENV TRANSFORMERS_CACHE=/opt/hf_cache
ENV HF_HOME=/opt/hf_cache

# Pre-download Coqui TTS model at build time so runtime is fully offline
RUN python -c "from TTS.api import TTS; TTS('tts_models/en/ljspeech/tacotron2-DDC', gpu=False)"

# Pre-download MMS-TTS model for fast-mode inference
RUN python -c "\
from transformers import VitsModel, AutoTokenizer; \
VitsModel.from_pretrained('facebook/mms-tts-eng'); \
AutoTokenizer.from_pretrained('facebook/mms-tts-eng')"

# Copy app source
COPY app/ ./app/

ENV PYTHONPATH=/app
ENV MAX_UPLOAD_MB=50

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
