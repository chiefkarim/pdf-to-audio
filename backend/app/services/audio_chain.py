import logging
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import Compressor, Gain, HighpassFilter, HighShelfFilter, PeakFilter, Pedalboard
from pydub import AudioSegment

from app.models.schemas import ExportFormat

logger = logging.getLogger(__name__)


def _build_board() -> Pedalboard:
    return Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=80.0),
            PeakFilter(cutoff_frequency_hz=3000.0, gain_db=2.5, q=0.9),
            PeakFilter(cutoff_frequency_hz=300.0, gain_db=-1.5, q=1.2),
            HighShelfFilter(cutoff_frequency_hz=7000.0, gain_db=-4.0),
            Gain(gain_db=1.0),
            Compressor(threshold_db=-18.0, ratio=2.5, attack_ms=10.0, release_ms=100.0),
        ]
    )


def _process_segment(board: Pedalboard, segment: np.ndarray, sample_rate: int) -> np.ndarray:
    processed = board(segment.reshape(1, -1), sample_rate)
    return processed.squeeze().astype(np.float32)


def _export_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = BytesIO()
    sf.write(buf, audio, sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def _export_mp3(audio: np.ndarray, sample_rate: int) -> bytes:
    int16 = np.clip(audio, -1.0, 1.0)
    int16 = (int16 * 32767).astype(np.int16)
    pydub_seg = AudioSegment(
        int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    buf = BytesIO()
    pydub_seg.export(buf, format="mp3", bitrate="192k")
    return buf.getvalue()


def apply_dsp(segments: list[np.ndarray], sample_rate: int) -> list[np.ndarray]:
    board = _build_board()
    return [_process_segment(board, seg, sample_rate) for seg in segments if len(seg) > 0]


def encode(dsp_segments: list[np.ndarray], sample_rate: int, fmt: ExportFormat) -> bytes:
    if not dsp_segments:
        return b""
    combined = np.concatenate(dsp_segments, axis=0)
    rms = np.sqrt(np.mean(combined**2))
    logger.info("audio rms: %.4f", rms)
    if fmt == ExportFormat.wav:
        return _export_wav(combined, sample_rate)
    return _export_mp3(combined, sample_rate)


def process_and_export(
    segments: list[np.ndarray],
    sample_rate: int,
    fmt: ExportFormat,
) -> bytes:
    return encode(apply_dsp(segments, sample_rate), sample_rate, fmt)


def ffmpeg_concat_files(wav_paths: list[Path], fmt: ExportFormat) -> bytes:
    """Concatenate on-disk WAV files via ffmpeg without loading them into Python memory."""
    if not wav_paths:
        return b""
    if len(wav_paths) == 1 and fmt == ExportFormat.wav:
        return wav_paths[0].read_bytes()
    codec_args = (
        ["-c", "copy"] if fmt == ExportFormat.wav
        else ["-c:a", "libmp3lame", "-b:a", "192k"]
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        filelist = tmpdir / "filelist.txt"
        filelist.write_text("\n".join(f"file '{p}'" for p in wav_paths))
        out = tmpdir / f"out.{fmt.value}"
        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(filelist),
             *codec_args, str(out)],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited {result.returncode}: "
                f"{result.stderr.decode(errors='replace')}"
            )
        return out.read_bytes()


def ffmpeg_concat(chunks: list[bytes], fmt: ExportFormat) -> bytes:
    """Concatenate WAV chunks via ffmpeg, encoding to fmt in one pass."""
    if not chunks:
        return b""
    if len(chunks) == 1 and fmt == ExportFormat.wav:
        return chunks[0]
    codec_args = (
        ["-c", "copy"] if fmt == ExportFormat.wav
        else ["-c:a", "libmp3lame", "-b:a", "192k"]
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        paths = []
        for i, chunk in enumerate(chunks):
            p = tmpdir / f"{i:06d}.wav"
            p.write_bytes(chunk)
            paths.append(p)
        filelist = tmpdir / "filelist.txt"
        filelist.write_text("\n".join(f"file '{p}'" for p in paths))
        out = tmpdir / f"out.{fmt.value}"
        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(filelist),
             *codec_args, str(out)],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited {result.returncode}: "
                f"{result.stderr.decode(errors='replace')}"
            )
        return out.read_bytes()
