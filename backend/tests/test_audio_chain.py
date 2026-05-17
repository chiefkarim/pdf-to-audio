import logging

import numpy as np
import pytest
import soundfile as sf
from io import BytesIO

from app.models.schemas import ExportFormat
from app.services.audio_chain import process_and_export


def make_segment(duration_s: float = 0.5, sr: int = 22050) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t) * 0.3


SR = 22050


def test_wav_output_magic_bytes() -> None:
    seg = make_segment(sr=SR)
    result = process_and_export([seg], SR, ExportFormat.wav)
    assert result[:4] == b"RIFF"


def test_mp3_output_nonempty() -> None:
    seg = make_segment(sr=SR)
    result = process_and_export([seg], SR, ExportFormat.mp3)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_empty_segments_returns_empty() -> None:
    result = process_and_export([], SR, ExportFormat.wav)
    assert result == b""


def test_rms_logged(caplog: pytest.LogCaptureFixture) -> None:
    seg = make_segment(sr=SR)
    with caplog.at_level(logging.INFO, logger="app.services.audio_chain"):
        process_and_export([seg], SR, ExportFormat.wav)
    assert any("audio rms:" in record.message for record in caplog.records)


def test_output_shape_preserved() -> None:
    duration_s = 1.0
    seg = make_segment(duration_s=duration_s, sr=SR)
    result = process_and_export([seg], SR, ExportFormat.wav)
    buf = BytesIO(result)
    audio, sr = sf.read(buf, dtype="float32")
    expected_samples = int(SR * duration_s)
    assert sr == SR
    assert abs(len(audio) - expected_samples) <= SR * 0.01
