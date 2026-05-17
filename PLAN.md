# Implementation Plan: PDF-to-Audio Converter

## Overview

Build a self-contained web app that converts PDF files to broadcast-quality audio. The pipeline is:
PDF → Tesseract OCR → Coqui TTS → pedalboard audio chain → MP3/WAV download.
Fully offline, Dockerised, no paid APIs.

## Architecture Decisions

- **In-memory job store**: No database for v1. Jobs are UUIDs mapped to status + output bytes. Acceptable since Docker restarts are rare in dev use.
- **Background tasks via FastAPI `BackgroundTasks`**: Simpler than Celery for a single-worker use case. Blocking work (OCR, TTS) is wrapped in `run_in_executor`.
- **pedalboard for DSP**: Spotify's library gives us typed VST-style nodes. De-esser is approximated as a high-shelf cut (true sidechain not available).
- **Frontend served by FastAPI**: No build step; static files mounted at `/`. Simplifies Docker.

---

## Dependency Graph

```
schemas.py (Pydantic models)
    │
    ├── storage.py (job store)
    │       │
    │       ├── routers/upload.py
    │       └── routers/jobs.py
    │
    └── services/
            ├── ocr.py          (no internal deps)
            ├── tts.py          (no internal deps)
            └── audio_chain.py  (no internal deps)
                    │
                    └── routers/upload.py (orchestrates all three)
```

Services are independent — they can be built and tested in parallel.

---

## Phase 1: Project Scaffold

### Task 1: Directory structure + requirements

**Description:** Create all directories, `requirements.txt`, and empty `__init__.py` files. No logic yet.

**Acceptance criteria:**
- [ ] `backend/app/` tree matches spec structure
- [ ] `requirements.txt` lists all dependencies with pinned major versions
- [ ] `frontend/` directory exists with placeholder files

**Verification:**
- [ ] `find backend frontend -type f` shows expected files
- [ ] `pip install -r backend/requirements.txt` succeeds (in Docker or venv)

**Dependencies:** None

**Files:**
- `backend/requirements.txt`
- `backend/app/__init__.py`, `routers/__init__.py`, `services/__init__.py`, `models/__init__.py`
- `frontend/index.html`, `frontend/app.js`, `frontend/styles.css` (stubs)

**Scope:** S

---

### Task 2: Pydantic schemas + storage

**Description:** Define `JobStatus` enum, `Job` model, and `ExportFormat` enum in `schemas.py`. Implement the in-memory job store in `storage.py` (create, get, update).

**Acceptance criteria:**
- [ ] `Job` has fields: `job_id`, `status`, `progress`, `format`, `result_bytes`, `error`
- [ ] `storage.create_job()` returns a UUID string
- [ ] `storage.get_job()` returns `None` for unknown IDs
- [ ] `storage.update_job()` mutates in place

**Verification:**
- [ ] `pytest tests/test_storage.py` passes

**Dependencies:** Task 1

**Files:**
- `backend/app/models/schemas.py`
- `backend/app/storage.py`
- `backend/tests/test_storage.py`

**Scope:** S

---

### Checkpoint: Phase 1
- [ ] All tests pass
- [ ] Project imports cleanly (`python -c "from app.storage import create_job"`)

---

## Phase 2: Core Services (parallelisable)

### Task 3: OCR service

**Description:** Implement `ocr.py` — takes a PDF file path, converts pages to images via `pdf2image`, runs Tesseract, returns a flat list of sentences (split on `.` / `?` / `!`, stripped of whitespace, filtered for empty strings).

**Acceptance criteria:**
- [ ] `extract_sentences(pdf_path: Path) -> list[str]`
- [ ] Returns at least one sentence for any valid PDF
- [ ] Empty pages produce no output (not empty strings)
- [ ] Raises `ValueError` on non-PDF input

**Verification:**
- [ ] `pytest tests/test_ocr.py` with fixture PDF

**Dependencies:** Task 1

**Files:**
- `backend/app/services/ocr.py`
- `backend/tests/test_ocr.py`
- `backend/tests/fixtures/sample.pdf`

**Scope:** S

---

### Task 4: TTS service

**Description:** Implement `tts.py` — loads Coqui TTS model once at startup (module-level singleton), exposes `synthesise(sentence: str) -> np.ndarray` returning float32 audio at the model's native sample rate (22050 Hz). Returns empty array for blank input.

**Acceptance criteria:**
- [ ] Output is `np.ndarray`, dtype `float32`
- [ ] Output length > 0 for non-empty input
- [ ] Sample rate constant exported: `SAMPLE_RATE = 22050`
- [ ] Empty/whitespace input returns zero-length array without error

**Verification:**
- [ ] `pytest tests/test_tts.py` (may be slow first run due to model download)

**Dependencies:** Task 1

**Files:**
- `backend/app/services/tts.py`
- `backend/tests/test_tts.py`

**Scope:** S

---

### Task 5: Audio chain service

**Description:** Implement `audio_chain.py` — `process_and_export(segments: list[np.ndarray], sample_rate: int, fmt: ExportFormat) -> bytes`. Applies the pedalboard chain to each segment, concatenates with zero gaps, exports to MP3 or WAV bytes.

Chain: `HighpassFilter(80) → PeakFilter(3000, +2.5) → PeakFilter(300, -1.5) → HighShelfFilter(7000, -4.0) → Gain(+1.0) → Compressor(−18 dB, 2.5:1)`

After concat: log RMS (the Analyser step). Export via pydub (MP3, 192 kbps) or soundfile (WAV, 16-bit PCM).

**Acceptance criteria:**
- [ ] Output is `bytes`, non-empty for non-empty input list
- [ ] MP3 output starts with `ID3` or `\xff\xfb` magic bytes
- [ ] WAV output starts with `RIFF` magic bytes
- [ ] RMS value is logged at INFO level
- [ ] Empty segment list returns empty bytes

**Verification:**
- [ ] `pytest tests/test_audio_chain.py`

**Dependencies:** Task 1

**Files:**
- `backend/app/services/audio_chain.py`
- `backend/tests/test_audio_chain.py`

**Scope:** M

---

### Checkpoint: Phase 2
- [ ] All service unit tests pass
- [ ] Each service importable standalone

---

## Phase 3: API Layer

### Task 6: FastAPI app skeleton + upload router

**Description:** Wire `main.py` (FastAPI instance, static file mount, router registration). Implement `routers/upload.py`: validate PDF magic bytes, enforce `MAX_UPLOAD_MB`, write to temp file, create job, enqueue background task that calls OCR → TTS → audio chain → `storage.update_job(done)`.

**Acceptance criteria:**
- [ ] `POST /upload` with valid PDF returns `202` with `job_id`
- [ ] Non-PDF file returns `400`
- [ ] File exceeding size limit returns `413`
- [ ] Background task progresses job through `queued → processing → done`
- [ ] Temp file deleted after job finishes (success or error)

**Verification:**
- [ ] `pytest tests/test_upload.py` using `httpx.AsyncClient`
- [ ] Manual: `curl -F file=@sample.pdf -F format=mp3 http://localhost:8000/upload`

**Dependencies:** Tasks 2, 3, 4, 5

**Files:**
- `backend/app/main.py`
- `backend/app/routers/upload.py`
- `backend/tests/test_upload.py`

**Scope:** M

---

### Task 7: Jobs router

**Description:** Implement `routers/jobs.py`: `GET /jobs/{job_id}/status` and `GET /jobs/{job_id}/download`. Status returns `JobStatus` JSON. Download streams `result_bytes` with correct Content-Type; returns `404` if unknown, `409` if not done yet.

**Acceptance criteria:**
- [ ] Status returns correct `progress` value (0, 50, or 100)
- [ ] Download returns `200` with audio bytes when status is `done`
- [ ] Download returns `409` when status is `processing`
- [ ] Download returns `404` for unknown job ID

**Verification:**
- [ ] `pytest tests/test_jobs.py`

**Dependencies:** Task 6

**Files:**
- `backend/app/routers/jobs.py`
- `backend/tests/test_jobs.py`

**Scope:** S

---

### Checkpoint: Phase 3
- [ ] All API tests pass
- [ ] Full pipeline runs: `POST /upload` → poll status → `GET /download`
- [ ] Manual end-to-end with a real PDF

---

## Phase 4: Frontend

### Task 8: Web UI

**Description:** Build `frontend/index.html` + `app.js` + `styles.css`. Single-page flow: file picker (PDF only) → format selector (MP3/WAV) → upload button → progress polling (1s interval) → download link when done → error message on failure.

**Acceptance criteria:**
- [ ] File picker filters to `.pdf` only
- [ ] Upload button disabled until file selected
- [ ] Progress bar updates as status polling returns `progress`
- [ ] Download link appears when status is `done`
- [ ] Error state shown if status is `error`
- [ ] No external CDN dependencies (fully offline)

**Verification:**
- [ ] Manual browser test: upload sample PDF, observe progress, download audio

**Dependencies:** Task 7

**Files:**
- `frontend/index.html`
- `frontend/app.js`
- `frontend/styles.css`

**Scope:** M

---

### Checkpoint: Phase 4
- [ ] Full user flow works in browser
- [ ] No console errors

---

## Phase 5: Docker

### Task 9: Dockerfile + docker-compose

**Description:** Write `backend/Dockerfile` — install system deps (tesseract-ocr, poppler-utils, ffmpeg), copy app, install Python deps. Write `docker-compose.yml` with volume mount for frontend, `MAX_UPLOAD_MB` env var.

**Acceptance criteria:**
- [ ] `docker compose up --build` starts without errors
- [ ] `POST /upload` works from browser against the container
- [ ] No internet access required at runtime (model downloaded during build)

**Verification:**
- [ ] `docker compose up --build` succeeds
- [ ] End-to-end flow works inside container

**Dependencies:** Task 8

**Files:**
- `backend/Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

**Scope:** S

---

### Final Checkpoint
- [ ] All tests pass: `pytest backend/tests/`
- [ ] Docker build succeeds
- [ ] End-to-end in container: upload PDF → download audio
- [ ] No outbound network calls at runtime

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Coqui TTS model download (~500 MB) slow in CI | Medium | Cache model in Docker layer; document first-run delay |
| pedalboard `HighShelfFilter` API differs by version | Medium | Pin `pedalboard==0.9.*`; add version check in tests |
| pdf2image requires poppler system dep | Medium | Install in Dockerfile; document for local dev |
| Coqui TTS CPU inference slow for long PDFs | Low | Out of scope for v1; note in README |

## Open Questions

None — spec is complete. Ready to implement.
