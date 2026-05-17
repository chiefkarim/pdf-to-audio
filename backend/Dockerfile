FROM python:3.11-slim

# Install uv (fast pip replacement)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# System deps: tesseract, poppler (pdf2image), ffmpeg (pydub), libsndfile (soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Pre-download Coqui TTS model at build time so runtime is fully offline
RUN python -c "from TTS.api import TTS; TTS('tts_models/en/ljspeech/tacotron2-DDC', gpu=False)"

# Copy app source
COPY app/ ./app/

ENV PYTHONPATH=/app
ENV MAX_UPLOAD_MB=50

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
