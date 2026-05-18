# Redesign: Dual Extraction Path + Dual TTS Mode + Parallel Synthesis

**Status:** Design — not yet implemented  
**Date:** 2026-05-18  
**Scope:** Three targeted improvements to OCR path, TTS model selection, and synthesis parallelism. No other changes.

---

## 1. Architecture Diagram

```
POST /upload  (file, format, mode)
     │
     ▼
upload.py: _run_pipeline(job_id, tmp_path, fmt, mode)
     │
     ├─── [A] ocr.py: extract_pages(pdf_path)
     │         │
     │         ├── fitz.open(pdf_path)
     │         │    └── page.get_text("text")  ──► char_count >= 50?
     │         │                                       │ YES → use direct text (fast path)
     │         │                                       │ NO  → rasterize page → Tesseract (fallback)
     │         │
     │         └── normalize_for_tts() + sentence split (unchanged)
     │
     ├─── [B] tts.py: synthesise_parallel(sentences, mode)
     │         │
     │         │   mode="quality"                    mode="fast"
     │         │        │                                 │
     │         │   Coqui Tacotron2-DDC           facebook/mms-tts-eng
     │         │   (22 kHz, chunk ≤150 chars)    (16 kHz, chunk ≤500 chars)
     │         │        │                                 │
     │         │        └─────────────┬───────────────────┘
     │         │                      │
     │         │              ProcessPoolExecutor
     │         │         (max_workers = min(cpu_count, n_sentences))
     │         │              ordered list[np.ndarray]
     │         │
     └─── audio_chain.py: process_and_export(segments, sample_rate, fmt)
               │
               pedalboard DSP chain (sample_rate from tts.get_sample_rate(mode))
               └── MP3 / WAV bytes
```

---

## 2. API Contract

### `POST /upload` (multipart/form-data)

| Field | Type | Default | Notes |
|---|---|---|---|
| `file` | file | required | PDF, validated via `%PDF` magic bytes |
| `format` | `"mp3" \| "wav"` | `"mp3"` | Export format |
| `mode` | `"fast" \| "quality"` | `"fast"` | TTS model selection |

Response (202):
```json
{ "job_id": "...", "status": "queued", "mode": "fast" }
```

The `mode` field is echoed back so callers can confirm which path was accepted.

---

## 3. TTS Mode Comparison

| Property | `quality` | `fast` |
|---|---|---|
| Model | `tts_models/en/ljspeech/tacotron2-DDC` (Coqui) | `facebook/mms-tts-eng` (HuggingFace VITS) |
| Library | `TTS==0.22.*` | `transformers>=4.41.0` |
| Sample rate | 22 050 Hz | 16 000 Hz |
| Relative speed (CPU) | ~1x real-time | ~10x real-time |
| Char limit per chunk | 150 (Tacotron2 fixed decoder steps) | 500 (VITS has no hard decoder step limit) |
| Model size on disk | ~400 MB (model + vocoder) | ~80 MB |
| Output naturalness | Higher; richer prosody | Adequate; intelligible |
| DSP chain (pedalboard) | Yes | Yes |

Both modes pass through the same `audio_chain.process_and_export`. DSP parameters
(EQ, compression, high-pass, gain) and export logic are shared and sample-rate-agnostic.

---

## 4. Data Flow — Two PDF Extraction Paths

```
For each page in fitz document:
  raw_text = page.get_text("text").strip()

  if len(raw_text) >= TEXT_LAYER_MIN_CHARS:   # fast path
      text = raw_text
  else:                                        # fallback
      pixmap = page.get_pixmap(dpi=200)
      image  = PIL.Image.frombytes(...)
      text   = pytesseract.image_to_string(image)

  sentences = split_and_normalize(text)
```

The fallback uses the fitz `Pixmap` directly — no poppler/pdf2image needed on the
fast path. pdf2image remains in requirements as a fallback dependency.

---

## 5. Component Changes

### 5.1 `backend/app/models/schemas.py` — add `TtsMode`

```python
class TtsMode(str, Enum):
    fast = "fast"
    quality = "quality"
```

Add `mode: TtsMode = TtsMode.fast` to the `Job` model for observability and so the
pipeline can retrieve the original choice during background processing.

### 5.2 `backend/requirements.txt`

| Action | Package | Notes |
|--------|---------|-------|
| Add | `pymupdf>=1.24.0` | fitz bindings; PyPI name is `pymupdf` |
| Add | `transformers>=4.41.0` | HuggingFace VITS for MMS-TTS fast mode |
| Add | `scipy>=1.13.0` | transformers TTS output utilities |
| Keep | `TTS==0.22.*` | Coqui TTS required for `quality` mode |
| Keep | `torch==2.5.1` | both models share the same CPU-only torch install |
| Keep | `pdf2image==1.17.*` | OCR fallback path |
| Keep | `pytesseract==0.3.*` | OCR fallback path |

### 5.3 `backend/app/services/tts.py` — full rewrite

Two independent lazy singletons, one per model. Each loads only on first use for
that mode, so an operator running only `fast` jobs never pays the ~400 MB quality
model memory cost.

**Public interface:**

```python
def get_sample_rate(mode: TtsMode = TtsMode.fast) -> int:
    # returns 22050 for "quality", 16000 for "fast"

def synthesise(sentence: str, mode: TtsMode = TtsMode.fast) -> np.ndarray:
    # routes to _synthesise_quality or _synthesise_fast

def synthesise_parallel(
    sentences: list[str], mode: TtsMode = TtsMode.fast
) -> list[np.ndarray]:
    # ProcessPoolExecutor; preserves order
```

Internal details:

- `_quality_tts: TTS | None` — Coqui singleton, loaded via
  `TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)`.
- `_fast_model: VitsModel | None`, `_fast_tokenizer: AutoTokenizer | None` —
  MMS-TTS singletons loaded from `"facebook/mms-tts-eng"`.
- `synthesise` quality path: identical to current code — `_chunk()` at
  `_MAX_CHARS = 150`, `tts.tts(text=c)` per chunk, concatenate.
- `synthesise` fast path: tokenize → `VitsModel.generate(**inputs)` →
  `.waveform.squeeze().numpy()` as `float32`. Chunks only as a safety guard at
  `_MAX_CHARS_FAST = 500`; VITS handles long inputs natively.
- `len(cleaned) < 3` skip guard applies to both paths.
- `synthesise_parallel`: filters empty sentences, then
  `ProcessPoolExecutor(max_workers=min(cpu_count, len(sentences)))` with
  `executor.map(partial(synthesise, mode=mode), filtered, chunksize=1)`.
  Pool is created fresh per call (see D9 below).
- For `quality` mode, cap `max_workers` at `min(cpu_count, 2)` to prevent OOM
  from multiple workers each loading the ~400 MB Coqui model.

### 5.4 `backend/app/services/ocr.py` — full rewrite

**Signature unchanged** — callers in `upload.py` unaffected.

```python
TEXT_LAYER_MIN_CHARS: int = 50

def extract_pages(pdf_path: Path) -> list[list[str]]: ...
```

- Import `fitz` (pymupdf).
- For each page: `page.get_text("text")`. If `len(stripped) >= TEXT_LAYER_MIN_CHARS`,
  use it. Otherwise convert via fitz `Pixmap` at 200 DPI → PIL Image → Tesseract.
- Log extraction method per page at DEBUG level.

**Threshold rationale:** 50 chars filters blank pages and fitz stray whitespace from
form fields. Low enough to avoid false OCR triggers on dense pages. Module-level
constant so tests can override without monkey-patching.

### 5.5 `backend/app/routers/upload.py` — targeted edits

1. Accept `mode: TtsMode = Form(TtsMode.fast)` alongside existing `format` field.
2. Pass `mode` to `storage.create_job` and `_run_pipeline`.
3. Inside the page loop, call `tts.synthesise_parallel(sentences, mode=mode)` instead
   of iterating `tts.synthesise` per sentence.
4. Pass `mode` to `tts.get_sample_rate(mode)` when calling `audio_chain.process_and_export`.
5. Echo `mode` in the 202 response body.

Pause check granularity changes from per-sentence to per-page (one check before
`synthesise_parallel`). Per-page granularity is acceptable; intra-page pause would
require IPC into worker processes.

### 5.6 `backend/app/storage.py` — minor

`create_job` accepts `mode: TtsMode` and stores it on the `Job` object.

### 5.7 `backend/Dockerfile` — pre-bake both models

```dockerfile
# Pre-bake Coqui Tacotron2-DDC (quality mode) — existing line, unchanged
RUN python -c "from TTS.api import TTS; TTS('tts_models/en/ljspeech/tacotron2-DDC', gpu=False)"

# Pre-bake facebook/mms-tts-eng (fast mode) — new
ENV TRANSFORMERS_CACHE=/opt/hf_cache
ENV HF_HOME=/opt/hf_cache
RUN python -c "
from transformers import VitsModel, AutoTokenizer
VitsModel.from_pretrained('facebook/mms-tts-eng')
AutoTokenizer.from_pretrained('facebook/mms-tts-eng')
"
```

`TRANSFORMERS_CACHE` and `HF_HOME` are set explicitly so the cache is at a known
path regardless of which user runs the container, and the layer is deterministic.

`espeak-ng` apt package must remain — MMS-TTS tokenizer requires it for
phonemization via the `phonemizer` transitive dependency.

### 5.8 `backend/app/main.py` — one-line edit

The lifespan warmup should call `tts.get_sample_rate(TtsMode.fast)` to eagerly load
the fast model (default) at startup. The quality model loads lazily on the first
`quality` request — acceptable given its larger footprint.

### 5.9 `backend/app/services/audio_chain.py` — no change

The pedalboard chain is sample-rate-agnostic: all filter frequencies and compressor
times are in Hz / ms, not in samples. Callers already supply `sample_rate`; they
will now source it from `tts.get_sample_rate(mode)`.

---

## 6. File Change Summary

| File | Change type | Summary |
|------|-------------|---------|
| `backend/app/models/schemas.py` | Edit | Add `TtsMode` enum; add `mode` field to `Job` |
| `backend/requirements.txt` | Edit | Add pymupdf, transformers, scipy; keep TTS for quality mode |
| `backend/app/services/ocr.py` | Rewrite | fitz fast path + Tesseract fallback |
| `backend/app/services/tts.py` | Rewrite | Dual-mode lazy singletons; mode-aware `synthesise` + `synthesise_parallel` |
| `backend/app/routers/upload.py` | Edit | Accept `mode` form param; thread through pipeline; per-page parallel synthesis |
| `backend/app/storage.py` | Edit | Store `mode` on job |
| `backend/Dockerfile` | Edit | Pre-bake both models; add `TRANSFORMERS_CACHE` env vars |
| `backend/app/main.py` | Edit | Warmup fast model at startup |
| `backend/app/services/audio_chain.py` | None | Sample-rate-agnostic already |

---

## 7. Decision Log

| # | Decision | Alternatives considered | Rationale |
|---|----------|------------------------|-----------|
| D1 | Expose both modes as selectable via `mode` form field | Single model, two endpoints, query param | Form field is consistent with existing `format` param; no URL change; clean enum validation via Pydantic |
| D2 | Default mode = `"fast"` | Default to `"quality"` | Perceived latency is the primary UX metric for a web service. 10x speed-up turns multi-minute waits into tens of seconds for typical PDFs. Users who need highest fidelity opt in explicitly |
| D3 | Both models in the same Docker image | Two images + API-gateway routing | Single container avoids a routing layer, halves deployment complexity, and keeps cold-start behaviour identical across modes. Combined image growth ~480 MB (80 MB MMS-TTS + 400 MB Coqui) is acceptable for self-hosted use |
| D4 | Independent lazy singletons per model | Shared loader with mode key | Lazy loading means quality model (~400 MB) is never in memory if operator only runs fast jobs, and vice versa. Separate singletons are simpler than a dynamic registry and easier to test in isolation |
| D5 | Use `transformers.VitsModel` directly for MMS-TTS | Route through Coqui TTS wrapper | Coqui 0.22 does not support `facebook/mms-tts-eng` without a custom model config YAML; that approach is fragile across Coqui releases. `transformers` is the canonical Meta-endorsed path |
| D6 | Keep `_chunk()` at 150 chars for quality; new 500-char limit for fast | Unified chunk size | Tacotron2 fixed `max_decoder_steps` constraint does not exist in VITS. Larger fast-mode limit reduces join-point glitches and preserves prosody across longer phrases |
| D7 | Run both modes through the same pedalboard DSP chain | Skip DSP for fast mode to reduce latency | Chain is sample-rate-agnostic and costs ~2–5 ms/segment. Removing it for fast mode would produce inconsistently audible output between modes. It is an explicit quality differentiator |
| D8 | `TRANSFORMERS_CACHE=/opt/hf_cache` in Dockerfile | Default `~/.cache` | Build user may differ from runtime user; explicit path guarantees pre-baked weights are found at runtime and the layer is reproducible |
| D9 | Fresh ProcessPoolExecutor per job | Module-level persistent pool | FastAPI background tasks run in threads; a persistent pool created at module import adds lifecycle and signal-handling complexity inside Docker. 50 ms pool startup cost is negligible vs inference time |
| D10 | Cap `max_workers=2` for quality mode | Let it match `cpu_count` | Each quality worker loads Coqui (~400 MB). On a 4-core/1 GB host, 4 workers = 1.6 GB model memory alone. Capping at 2 keeps peak model memory at ~800 MB |
| D11 | `fitz.get_text("text")` as OCR fast path | `pdfplumber`, `pdfminer.six` | pymupdf is the fastest pure-Python PDF text extractor; already needed for the pixmap OCR fallback so it is a single dep |
| D12 | Threshold = 50 chars for direct-text path | 0, 100, 200 | 0 triggers on form-only pages; 100+ risks OCR-falling on sparse-but-valid text pages; 50 is a pragmatic middle |
| D13 | fitz Pixmap at 200 DPI for OCR fallback | pdf2image (poppler) at 300 DPI | Removes poppler as a hard runtime dep for the fast path; 200 DPI is sufficient for Tesseract on standard fonts |

---

## 8. Assumptions

1. Docker build environment has outbound internet at build time. Runtime is fully offline per SPEC.
2. `facebook/mms-tts-eng` is available on HuggingFace Hub; weights are MIT-licensed. Coqui Tacotron2-DDC is MPL-2.0 (unchanged).
3. `phonemizer` (pulled by transformers for MMS-TTS) requires `espeak-ng` at runtime; the existing apt install in Dockerfile covers this.
4. Deployment host has at least 1 CPU core. `synthesise_parallel` degrades gracefully to `max_workers=1` on a single-core host.
5. Memory budget: quality model ~400 MB + fast model ~80 MB = ~480 MB static footprint. Both models are warm simultaneously only if mixed-mode requests are served concurrently. For single-mode deployments, only one model is ever loaded.
6. `normalize_for_tts` in `services/normalize.py` needs no changes — it operates on raw strings before model input.
7. WAV output in fast mode will be 16 kHz PCM; in quality mode 22 kHz PCM. This is a documented, expected difference, not a bug.

---

## 9. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MMS-TTS quality unacceptable for some sentence structures | Medium | Medium | Evaluate against fixture sentences before shipping; `quality` mode is always available as a fallback |
| Both models warm simultaneously under mixed-mode concurrent load exceeds memory limit | Low | High | Document memory requirements in ops runbook; add `QUALITY_MAX_WORKERS` env var cap if needed post-deployment |
| fitz text extraction returns garbled Unicode on PDFs with custom encoding | Medium | Low | `normalize_for_tts` strips most noise; add printable-char ratio guard (>30%) post-testing if needed |
| ProcessPoolExecutor fork-safety with torch on Linux | Low | High | Torch CPU is fork-safe on Linux; if issues appear, switch pool start method to `spawn` via `mp.set_start_method("spawn")` in pool initializer |
| `espeak-ng` version mismatch between apt and phonemizer expectations | Low | High | Validate at build time; pin `espeak-ng` apt version if a mismatch is observed |
| 200 DPI OCR fallback misses fine print vs current 300 DPI pdf2image default | Medium | Low | Bump to 300 DPI in the fitz Pixmap call if OCR accuracy is reported degraded; one-integer change |
| Image size increase (~480 MB for both models) hits CI/CD artifact limits | Low | Medium | Models are in distinct `RUN` layers; layer caching means rebuilds only re-pull on dep changes |

---

## 10. Open Questions

1. Does `facebook/mms-tts-eng` produce acceptable audio quality for the target use case, or does it need evaluation against fixture sentences before committing to the implementation?
2. Should `TEXT_LAYER_MIN_CHARS = 50` be an env-var override (like `MAX_UPLOAD_MB`) or is a hardcoded constant sufficient?
3. Is there a soft memory limit on the Docker container in the target deployment? Determines whether `QUALITY_MAX_WORKERS` should be env-configurable.
4. Should the 202 response echo `mode`, or is it sufficient to retrieve it via `GET /jobs/{job_id}`?
5. Should `pdf2image` / `poppler-utils` be removed now that fitz Pixmap covers the OCR fallback, or deferred to a follow-up cleanup PR?
