import numpy as np
from TTS.api import TTS

_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"

_tts: TTS | None = None


def _get_tts() -> TTS:
    global _tts
    if _tts is None:
        _tts = TTS(_MODEL, gpu=False)
    return _tts


def get_sample_rate() -> int:
    return int(_get_tts().synthesizer.output_sample_rate)


def synthesise(sentence: str) -> np.ndarray:
    cleaned = sentence.strip()
    # Tacotron2 Conv1d requires at least ~5 phoneme frames; skip trivially short input.
    if len(cleaned) < 3:
        return np.array([], dtype=np.float32)
    result = _get_tts().tts(text=cleaned)
    return np.array(result, dtype=np.float32)
