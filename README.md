# PDF-to-Audio

Convert PDF files to broadcast-quality audio via OCR + neural TTS. Runs fully offline.

## Prerequisites

**Option A — Docker (recommended)**
- Docker Engine 24+
- docker compose v2 (bundled with Docker Desktop or installed as a plugin)

**Option B — Local Python**
- Python 3.11+
- System packages: `tesseract-ocr`, `poppler-utils`, `ffmpeg`, `libsndfile1`

## Quick start (Docker)

```bash
docker compose up --build
```

Open http://localhost:8000 once the container is ready.
The first build is slow (~5 min) because it pre-downloads the Coqui TTS model.
Subsequent builds use the cached layer.

## Local dev (no Docker)

```bash
cd backend
uv pip install -r requirements.txt
PYTHONPATH=. uvicorn app.main:app --reload
```

The frontend is served from `../frontend` via FastAPI's `StaticFiles` mount.

## Running tests

```bash
cd backend
PYTHONPATH=. pytest tests/ -v --ignore=tests/test_tts.py
```

TTS tests are slow and require the model to be downloaded. Include them with:

```bash
PYTHONPATH=. pytest tests/ -v -m slow
```
