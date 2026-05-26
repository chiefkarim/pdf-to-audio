FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN uv pip install --system --no-cache \
    torch==2.5.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --system --no-cache -r requirements.txt

# Cache paths for HuggingFace models (set before pre-bake RUNs)
ENV TRANSFORMERS_CACHE=/opt/hf_cache
ENV HF_HOME=/opt/hf_cache

# Pre-download MMS-TTS model (fast mode)
RUN python -c "\
from transformers import VitsModel, AutoTokenizer; \
VitsModel.from_pretrained('facebook/mms-tts-eng'); \
AutoTokenizer.from_pretrained('facebook/mms-tts-eng')"

# Pre-download quality male VITS model
RUN python -c "\
from transformers import VitsModel, AutoTokenizer; \
VitsModel.from_pretrained('kakao-enterprise/vits-vctk'); \
AutoTokenizer.from_pretrained('kakao-enterprise/vits-vctk')"

COPY backend/app/ ./app/
COPY frontend/ ./frontend/

ENV PYTHONPATH=/app
ENV MAX_UPLOAD_MB=50
ENV DB_PATH=/app/data/jobs.db

ENV MALLOC_ARENA_MAX=2
ENV MALLOC_MMAP_THRESHOLD_=65536
ENV MALLOC_TRIM_THRESHOLD_=65536

RUN mkdir -p /app/data

# HuggingFace Spaces runs containers as uid 1000
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
