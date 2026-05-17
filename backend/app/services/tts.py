import re
import numpy as np
from TTS.api import TTS

_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
_MAX_CHARS = 150  # Tacotron2 decoder reliably handles up to ~150 chars

_tts: TTS | None = None


def _get_tts() -> TTS:
    global _tts
    if _tts is None:
        _tts = TTS(_MODEL, gpu=False)
    return _tts


def get_sample_rate() -> int:
    return int(_get_tts().synthesizer.output_sample_rate)


def _chunk(text: str) -> list[str]:
    """Split text into <=_MAX_CHARS chunks at natural break points."""
    if len(text) <= _MAX_CHARS:
        return [text]
    # Try splitting at "; " then ", " then " " (last resort)
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
            return [p for p in parts if p]
    # Fallback: hard-cut at _MAX_CHARS
    return [text[i:i + _MAX_CHARS] for i in range(0, len(text), _MAX_CHARS)]


def synthesise(sentence: str) -> np.ndarray:
    cleaned = sentence.strip()
    # Tacotron2 Conv1d requires at least ~5 phoneme frames; skip trivially short input.
    if len(cleaned) < 3:
        return np.array([], dtype=np.float32)
    chunks = _chunk(cleaned)
    arrays = [np.array(_get_tts().tts(text=c), dtype=np.float32) for c in chunks]
    return np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)
