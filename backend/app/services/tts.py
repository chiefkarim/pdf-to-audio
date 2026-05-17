import numpy as np
from TTS.api import TTS

SAMPLE_RATE: int = 22050

_tts: TTS | None = None


def _get_tts() -> TTS:
    global _tts
    if _tts is None:
        _tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
    return _tts


def synthesise(sentence: str) -> np.ndarray:
    if not sentence.strip():
        return np.array([], dtype=np.float32)
    result = _get_tts().tts(text=sentence)
    return np.array(result, dtype=np.float32)
