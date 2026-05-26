import logging
import os
import re
import functools
import threading
import concurrent.futures
import numpy as np
import torch
from TTS.api import TTS
from transformers import VitsModel, AutoTokenizer
from app.models.schemas import TtsMode

logger = logging.getLogger(__name__)

torch.backends.mkldnn.enabled = False

_QUALITY_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
_FAST_MODEL_ID = "facebook/mms-tts-eng"
_MAX_CHARS = 150
_MAX_CHARS_FAST = 200
_MIN_CHUNK_CHARS = 20  # Tacotron2 attention fails on very short encoder sequences

SAMPLE_RATES: dict[str, int] = {"fast": 16000, "quality": 22050}

_quality_tts: TTS | None = None
_fast_model: VitsModel | None = None
_fast_tokenizer: AutoTokenizer | None = None

# Tacotron2 mutates decoder state (attention weights, context) in-place on the
# model object — concurrent calls corrupt each other's alignment tensors.
_quality_lock = threading.Lock()


def _get_quality_tts() -> TTS:
    global _quality_tts
    if _quality_tts is None:
        with _quality_lock:
            if _quality_tts is None:
                logger.info("loading quality TTS model: %s", _QUALITY_MODEL)
                _quality_tts = TTS(_QUALITY_MODEL, gpu=False)
                logger.info("quality TTS model loaded")
    return _quality_tts


def _get_fast_model() -> tuple[VitsModel, AutoTokenizer]:
    global _fast_model, _fast_tokenizer
    if _fast_model is None or _fast_tokenizer is None:
        logger.info("loading fast TTS model: %s", _FAST_MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(_FAST_MODEL_ID)
        model = VitsModel.from_pretrained(_FAST_MODEL_ID)
        _fast_tokenizer, _fast_model = tokenizer, model
        logger.info("fast TTS model loaded")
    return _fast_model, _fast_tokenizer


def get_sample_rate(mode: TtsMode = TtsMode.fast) -> int:
    return SAMPLE_RATES[mode]


def _merge_short_chunks(parts: list[str], max_chars: int) -> list[str]:
    """Merge chunks shorter than _MIN_CHUNK_CHARS into adjacent neighbours."""
    merged: list[str] = []
    for p in parts:
        if merged and len(p) < _MIN_CHUNK_CHARS and len(merged[-1]) + 1 + len(p) <= max_chars:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


def _chunk(text: str) -> list[str]:
    """Split text into <=_MAX_CHARS chunks at natural break points."""
    if len(text) <= _MAX_CHARS:
        return [text]
    for sep in ("; ", ", ", " "):
        parts: list[str] = []
        current = ""
        for token in re.split(f"({re.escape(sep)})", text):
            if len(current) + len(token) <= _MAX_CHARS:
                current += token
            else:
                if current:
                    parts.append(current.strip())
                current = token
        if current:
            parts.append(current.strip())
        if all(len(p) <= _MAX_CHARS for p in parts):
            return _merge_short_chunks([p for p in parts if p], _MAX_CHARS)
    return [text[i:i + _MAX_CHARS] for i in range(0, len(text), _MAX_CHARS)]


def _chunk_fast(text: str) -> list[str]:
    """Split text into <=_MAX_CHARS_FAST chunks at natural break points."""
    if len(text) <= _MAX_CHARS_FAST:
        return [text]
    for sep in ("; ", ", ", " "):
        parts: list[str] = []
        current = ""
        for token in re.split(f"({re.escape(sep)})", text):
            if len(current) + len(token) <= _MAX_CHARS_FAST:
                current += token
            else:
                if current:
                    parts.append(current.strip())
                current = token
        if current:
            parts.append(current.strip())
        if all(len(p) <= _MAX_CHARS_FAST for p in parts):
            return _merge_short_chunks([p for p in parts if p], _MAX_CHARS_FAST)
    return [text[i:i + _MAX_CHARS_FAST] for i in range(0, len(text), _MAX_CHARS_FAST)]


def synthesise(sentence: str, mode: TtsMode = TtsMode.fast) -> np.ndarray:
    cleaned = sentence.strip()
    if len(cleaned) < 3:
        return np.array([], dtype=np.float32)

    if mode == TtsMode.quality:
        chunks = [c for c in _chunk(cleaned) if len(c) >= _MIN_CHUNK_CHARS]
        if not chunks:
            return np.array([], dtype=np.float32)
        tts = _get_quality_tts()
        with _quality_lock:
            arrays = [np.array(tts.tts(text=c), dtype=np.float32) for c in chunks]
        return np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)

    # fast mode
    chunks = _chunk_fast(cleaned)
    model, tokenizer = _get_fast_model()
    arrays: list[np.ndarray] = []
    with torch.inference_mode():
        for c in chunks:
            inputs = tokenizer(text=c, return_tensors="pt")
            output = model(**inputs)
            wav = output.waveform.squeeze(0).numpy().astype(np.float32, copy=True)
            del output, inputs
            arrays.append(wav)
    return np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)


def _worker_init() -> None:
    try:
        torch.set_num_threads(1)
    except Exception:
        logger.warning("torch.set_num_threads(1) failed in worker", exc_info=True)


def make_executor(mode: TtsMode) -> concurrent.futures.ThreadPoolExecutor:
    workers = min(8, max(2, (os.cpu_count() or 4) - 1))
    return concurrent.futures.ThreadPoolExecutor(max_workers=workers, initializer=_worker_init)


def synthesise_parallel(
    sentences: list[str],
    mode: TtsMode = TtsMode.fast,
    executor: concurrent.futures.ThreadPoolExecutor | None = None,
) -> list[np.ndarray]:
    filtered = [s for s in sentences if len(s.strip()) >= 3]
    if not filtered:
        return []

    worker_fn = functools.partial(synthesise, mode=mode)
    if executor is not None:
        return list(executor.map(worker_fn, filtered, chunksize=1))

    with make_executor(mode) as ex:
        return list(ex.map(worker_fn, filtered, chunksize=1))
