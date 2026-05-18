import logging
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from app.models.schemas import ExportFormat
from app.services.audio_chain import encode, ffmpeg_concat, process_and_export


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


def test_ffmpeg_concat_empty_returns_empty() -> None:
    assert ffmpeg_concat([], ExportFormat.wav) == b""


def test_ffmpeg_concat_single_wav_chunk_returned_directly() -> None:
    seg = make_segment(sr=SR)
    wav_bytes = encode([seg], SR, ExportFormat.wav)
    result = ffmpeg_concat([wav_bytes], ExportFormat.wav)
    assert result is wav_bytes


def test_ffmpeg_concat_single_chunk_mp3_produces_mp3() -> None:
    seg = make_segment(sr=SR)
    wav_bytes = encode([seg], SR, ExportFormat.wav)
    result = ffmpeg_concat([wav_bytes], ExportFormat.mp3)
    assert len(result) > 0
    assert result[:4] != b"RIFF"


def test_ffmpeg_concat_multiple_wav_chunks_produces_valid_wav() -> None:
    seg_a = make_segment(duration_s=0.3, sr=SR)
    seg_b = make_segment(duration_s=0.4, sr=SR)
    wav_a = encode([seg_a], SR, ExportFormat.wav)
    wav_b = encode([seg_b], SR, ExportFormat.wav)
    result = ffmpeg_concat([wav_a, wav_b], ExportFormat.wav)
    assert result[:4] == b"RIFF"
    audio, sr = sf.read(BytesIO(result), dtype="float32")
    assert sr == SR
    expected = len(seg_a) + len(seg_b)
    assert abs(len(audio) - expected) <= SR * 0.02


def test_ffmpeg_concat_multiple_wav_chunks_to_mp3_produces_nonempty() -> None:
    seg_a = make_segment(duration_s=0.3, sr=SR)
    seg_b = make_segment(duration_s=0.3, sr=SR)
    wav_a = encode([seg_a], SR, ExportFormat.wav)
    wav_b = encode([seg_b], SR, ExportFormat.wav)
    result = ffmpeg_concat([wav_a, wav_b], ExportFormat.mp3)
    assert len(result) > 0


def test_ffmpeg_concat_corrupt_input_raises_runtime_error() -> None:
    with pytest.raises(RuntimeError, match="ffmpeg exited"):
        ffmpeg_concat([b"not a wav", b"not a wav"], ExportFormat.wav)
