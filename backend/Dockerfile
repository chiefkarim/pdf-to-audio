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

# Prevent glibc per-thread arena fragmentation from PyTorch CPU tensor allocs.
# MALLOC_ARENA_MAX=2: cap arenas at 2 instead of default 8*ncpu so malloc_trim
# can reclaim memory from worker thread arenas.
# MALLOC_MMAP_THRESHOLD_=65536: tensors >=64KB use mmap and are immediately
# returned to OS on free() rather than held in arena free-lists.
# MALLOC_TRIM_THRESHOLD_=65536: trigger trim automatically after 64KB is freed.
ENV MALLOC_ARENA_MAX=2
ENV MALLOC_MMAP_THRESHOLD_=65536
ENV MALLOC_TRIM_THRESHOLD_=65536

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
