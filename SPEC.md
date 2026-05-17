# SPEC: PDF-to-Audio Converter

## 1. Objective

Convert uploaded PDF files into broadcast-quality audio using OCR + neural TTS + a deterministic audio processing chain. Target users: individuals who prefer listening to documents (accessibility, multitasking).

**Success criteria:**
- PDF uploaded via web UI → downloadable MP3/WAV within a reasonable time
- Audio passes through the full processing chain without artifacts
- Zero inter-sentence gaps in concatenated output
- Runs fully offline with no paid third-party APIs

---

## 2. Architecture

```
Browser (Upload UI)
     │  POST /upload (multipart PDF)
     ▼
FastAPI backend
     ├── OCR service      pdf → page images → raw text (Tesseract)
     ├── TTS service      sentences → raw audio segments (Coqui TTS)
     ├── Audio chain      raw segments → processed audio (pedalboard)
     │     HPF 80 Hz → Voice EQ → De-esser 7 kHz → Gain → Compressor 2.5:1 → Analyser
     └── Concat + export  zero-gap concatenation → MP3 / WAV
     │  GET /jobs/{id}/download
     ▼
Browser (Download)
```

---

## 3. Tech Stack

| Layer            | Technology                        | Rationale                                      |
|------------------|-----------------------------------|------------------------------------------------|
| API framework    | FastAPI                           | Async, OpenAPI docs, simple background tasks   |
| OCR              | pytesseract + pdf2image           | Tesseract wrapping; poppler for PDF→image      |
| TTS              | Coqui TTS (`TTS` package)         | Offline neural voices, no API key              |
| Audio processing | pedalboard (Spotify)              | VST-style DSP chain in pure Python             |
| Audio I/O        | soundfile + pydub                 | WAV/MP3 encode & decode                        |
| Frontend         | Vanilla HTML/CSS/JS (single page) | No build step, easy to serve from FastAPI      |
| Containerisation | Docker + docker-compose           | Single `docker compose up` startup             |

---

## 4. Project Structure

```
pdf-to-audio/
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app, mounts static frontend
│   │   ├── routers/
│   │   │   ├── upload.py          # POST /upload → job_id
│   │   │   └── jobs.py            # GET /jobs/{id}/status, /jobs/{id}/download
│   │   ├── services/
│   │   │   ├── ocr.py             # pdf → list[str] sentences via Tesseract
│   │   │   ├── tts.py             # sentences → list[np.ndarray] audio segments
│   │   │   └── audio_chain.py     # process + concatenate segments → bytes
│   │   ├── models/
│   │   │   └── schemas.py         # Pydantic models: Job, JobStatus, ExportFormat
│   │   └── storage.py             # In-memory job store (dict keyed by UUID)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── docker-compose.yml
└── SPEC.md
```

---

## 5. API Contract

### `POST /upload`
- Body: `multipart/form-data` — `file` (PDF), `format` (mp3|wav)
- Response `202`: `{ "job_id": "<uuid>", "status": "queued" }`

### `GET /jobs/{job_id}/status`
- Response `200`: `{ "job_id": "...", "status": "queued|processing|done|error", "progress": 0-100 }`

### `GET /jobs/{job_id}/download`
- Response `200`: audio file stream (Content-Type: audio/mpeg or audio/wav)
- Response `404` if job not found; `409` if not yet done

---

## 6. Audio Processing Chain

Each TTS segment is processed independently, then all segments are concatenated with **zero padding** (no silence inserted between sentences).

```
segment (np.ndarray, float32, 22050 Hz)
  → HighpassFilter(cutoff_hz=80)
  → PeakFilter(cutoff_hz=3000, gain_db=+2.5, q=0.9)   # presence lift
  → PeakFilter(cutoff_hz=300,  gain_db=-1.5, q=1.2)   # mud cut
  → HighShelfFilter(cutoff_hz=7000, gain_db=-4.0)      # de-esser approximation
  → Gain(gain_db=+1.0)
  → Compressor(threshold_db=-18, ratio=2.5, attack_ms=10, release_ms=100)
  → [level analysis via numpy RMS — logged, not applied]
  → concatenate all segments (np.concatenate, axis=0)
  → export as MP3 (pydub, 192 kbps) or WAV (soundfile, 16-bit PCM)
```

> **Note:** A full multiband de-esser requires sidechain DSP not natively in pedalboard.
> The high-shelf cut at 7 kHz is an effective broadband approximation suitable for neural TTS output.

---

## 7. Code Style

- Python 3.11+
- Black formatter, isort imports
- Type hints on all function signatures
- Services are pure functions (no side effects except I/O); stateful job tracking lives in `storage.py`
- No global mutable state outside `storage.py`
- Async FastAPI routes; blocking work (TTS, OCR) offloaded to `asyncio.run_in_executor`

---

## 8. Testing Strategy

| Layer              | Approach                                              |
|--------------------|-------------------------------------------------------|
| OCR service        | Unit test with a known fixture PDF → assert text      |
| TTS service        | Unit test: assert output is `np.ndarray`, correct SR  |
| Audio chain        | Unit test: assert output shape, RMS within bounds     |
| API integration    | `httpx.AsyncClient` against live FastAPI test app     |
| End-to-end         | Upload fixture PDF, poll status, assert download 200  |

Test framework: `pytest` + `pytest-asyncio`.

---

## 9. Boundaries

### Always do
- Validate uploaded file is a PDF (magic bytes check, not just extension)
- Sanitize filenames before writing to disk
- Clean up temp files after job completes or errors
- Log chain step RMS values for the Analyser stage

### Ask first about
- Changing the audio chain parameters (ratios, frequencies)
- Swapping the TTS voice or model

### Never do
- Make outbound network requests at runtime
- Store uploaded files permanently (temp dir only, deleted post-job)
- Accept files larger than 50 MB without a configurable limit check

---

## 10. Docker

```yaml
# docker-compose.yml (outline)
services:
  app:
    build: ./backend
    ports: ["8000:8000"]
    volumes:
      - ./frontend:/app/frontend:ro   # served as static by FastAPI
    environment:
      - MAX_UPLOAD_MB=50
```

Single container; no database. Jobs live in memory — restart clears them (acceptable for v1).
