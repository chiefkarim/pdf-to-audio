import logging
import os
import queue
import tempfile
import threading
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.models.schemas import ExportFormat, JobStatus, TtsMode
from app.services import audio_chain, ocr, tts
from app import storage

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_max_bytes() -> int:
    raw = os.environ.get("MAX_UPLOAD_MB", "50")
    return int(raw) * 1024 * 1024


def _run_pipeline(job_id: str, tmp_path: Path, fmt: ExportFormat, mode: TtsMode) -> None:
    with tempfile.TemporaryDirectory() as page_dir_str:
        page_dir = Path(page_dir_str)
        try:
            storage.update_job(job_id, status=JobStatus.processing, progress=0)

            total_pages = ocr.page_count(tmp_path)
            if total_pages == 0:
                storage.update_job(job_id, status=JobStatus.error, error="Empty PDF")
                return

            storage.update_job(job_id, pages_total=total_pages)

            page_queue: queue.Queue = queue.Queue(maxsize=3)

            def _ocr_producer() -> None:
                try:
                    for page_idx, sentences, is_ocr in ocr.stream_pages(tmp_path):
                        page_queue.put((page_idx, sentences, is_ocr))
                except Exception as exc:
                    page_queue.put(exc)
                finally:
                    page_queue.put(None)

            ocr_thread = threading.Thread(target=_ocr_producer, daemon=True)
            ocr_thread.start()

            wav_paths: list[Path] = []
            pages_processed = 0
            ocr_count = 0
            sample_rate = tts.get_sample_rate(mode)
            PARTIAL_EVERY = 5

            with tts.make_executor(mode) as executor:
                while True:
                    item = page_queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item

                    _, sentences, is_ocr = item

                    event = storage.get_pause_event(job_id)
                    if event:
                        event.wait()

                    if sentences:
                        segments = tts.synthesise_parallel(sentences, mode=mode, executor=executor)
                        dsp = audio_chain.apply_dsp(segments, sample_rate)
                        if dsp:
                            wav_path = page_dir / f"{pages_processed:06d}.wav"
                            wav_path.write_bytes(audio_chain.encode(dsp, sample_rate, ExportFormat.wav))
                            wav_paths.append(wav_path)

                    if is_ocr:
                        ocr_count += 1

                    pages_processed += 1
                    progress = int(100 * pages_processed / total_pages)

                    update: dict = dict(pages_done=pages_processed, ocr_pages=ocr_count, progress=progress)
                    if pages_processed % PARTIAL_EVERY == 0 and wav_paths:
                        try:
                            update["partial_bytes"] = audio_chain.ffmpeg_concat_files(wav_paths, fmt)
                        except Exception:
                            logger.warning("partial audio generation failed at page %d", pages_processed)

                    storage.update_job(job_id, **update)

            ocr_thread.join()

            if not wav_paths:
                storage.update_job(job_id, status=JobStatus.error, error="No text found in PDF")
                return

            result = audio_chain.ffmpeg_concat_files(wav_paths, fmt)
            storage.update_job(
                job_id,
                status=JobStatus.done,
                progress=100,
                result_bytes=result,
                partial_bytes=None,
            )
        except Exception as e:
            try:
                storage.update_job(job_id, status=JobStatus.error, error=str(e) or repr(e))
            except Exception:
                logger.exception("failed to mark job %s as error (original: %s); job may appear stuck", job_id, e)
            raise
        finally:
            tmp_path.unlink(missing_ok=True)


@router.post("/upload", status_code=202)
async def upload_pdf(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    format: ExportFormat = Form(ExportFormat.mp3),
    mode: TtsMode = Form(TtsMode.fast),
) -> JSONResponse:
    max_bytes = _get_max_bytes()

    header = await file.read(4)
    if len(header) < 4 or header[:4] != b"%PDF":
        raise HTTPException(status_code=400, detail="File is not a valid PDF")

    rest = await file.read()
    total_size = len(header) + len(rest)
    if total_size > max_bytes:
        raise HTTPException(status_code=413, detail="File exceeds size limit")

    content = header + rest

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    filename = file.filename or ""
    job_id = storage.create_job(format, filename=filename, mode=mode)

    background_tasks.add_task(_run_pipeline, job_id, tmp_path, format, mode)

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "queued", "mode": mode.value},
    )
