import os
import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.models.schemas import ExportFormat, JobStatus, TtsMode
from app.services import audio_chain, ocr, tts
from app import storage

router = APIRouter()


def _get_max_bytes() -> int:
    raw = os.environ.get("MAX_UPLOAD_MB", "50")
    return int(raw) * 1024 * 1024


def _run_pipeline(job_id: str, tmp_path: Path, fmt: ExportFormat, mode: TtsMode) -> None:
    try:
        storage.update_job(job_id, status=JobStatus.processing, progress=0)

        pages = ocr.extract_pages(tmp_path)

        total_pages = len(pages)

        if total_pages == 0:
            storage.update_job(job_id, status=JobStatus.error, error="No text found in PDF")
            tmp_path.unlink(missing_ok=True)
            return

        # Filter out empty pages but keep total count accurate for non-empty ones
        non_empty_pages = [(i, page) for i, page in enumerate(pages) if page]
        if not non_empty_pages:
            storage.update_job(job_id, status=JobStatus.error, error="No text found in PDF")
            tmp_path.unlink(missing_ok=True)
            return

        storage.update_job(job_id, pages_total=total_pages)

        all_segments: list[np.ndarray] = []
        partial: bytes = b""
        pages_processed = 0

        with tts.make_executor(mode) as executor:
            for _, sentences in non_empty_pages:
                # Pause check before each page
                event = storage.get_pause_event(job_id)
                if event:
                    event.wait()

                segments = tts.synthesise_parallel(sentences, mode=mode, executor=executor)
                all_segments.extend(segments)

                pages_processed += 1

                # Re-export full accumulated audio so partial is always valid
                partial = audio_chain.process_and_export(all_segments, tts.get_sample_rate(mode), fmt)
                progress = int(100 * pages_processed / len(non_empty_pages))

                storage.update_job(
                    job_id,
                    pages_done=pages_processed,
                    partial_bytes=partial,
                    progress=progress,
                )

        storage.update_job(
            job_id,
            status=JobStatus.done,
            progress=100,
            result_bytes=partial,
        )
    except Exception as e:
        storage.update_job(job_id, status=JobStatus.error, error=str(e))
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
