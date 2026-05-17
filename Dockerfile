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

COPY backend/app/ ./app/
COPY frontend/ ./frontend/

ENV PYTHONPATH=/app
ENV MAX_UPLOAD_MB=50
ENV DB_PATH=/app/data/jobs.db

RUN mkdir -p /app/data

# HuggingFace Spaces runs containers as uid 1000
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

RUN python -c "from TTS.api import TTS; TTS('tts_models/en/ljspeech/tacotron2-DDC', gpu=False)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
