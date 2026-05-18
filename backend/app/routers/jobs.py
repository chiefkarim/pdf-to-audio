import logging
from io import BytesIO

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import ExportFormat, JobStatus
from app import storage
from app.services import audio_chain
from app.routers.upload import _active_wav_paths

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs")

_MEDIA_TYPES: dict[ExportFormat, str] = {
    ExportFormat.mp3: "audio/mpeg",
    ExportFormat.wav: "audio/wav",
}

_EXTENSIONS: dict[ExportFormat, str] = {
    ExportFormat.mp3: "mp3",
    ExportFormat.wav: "wav",
}


@router.get("")
def list_jobs() -> list[dict]:
    jobs = storage.list_jobs()
    return [
        {
            "job_id": j.job_id,
            "status": j.status,
            "progress": j.progress,
            "format": j.format,
            "created_at": j.created_at.isoformat(),
            "pages_total": j.pages_total,
            "pages_done": j.pages_done,
            "filename": j.filename,
        }
        for j in jobs
    ]


@router.get("/{job_id}/status")
def get_status(job_id: str) -> dict:
    job = storage.get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job.job_id, "status": job.status, "progress": job.progress}


@router.get("/{job_id}/download")
def download(job_id: str) -> StreamingResponse:
    job = storage.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == JobStatus.error:
        raise HTTPException(status_code=422, detail=f"Job failed: {job.error}")
    if job.status != JobStatus.done:
        raise HTTPException(status_code=409, detail="Job not done yet")
    if job.result_bytes is None:
        raise HTTPException(status_code=500, detail="No audio data")

    ext = _EXTENSIONS[job.format]
    media_type = _MEDIA_TYPES[job.format]

    response = StreamingResponse(
        BytesIO(job.result_bytes),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename=\"audio.{ext}\""},
    )
    storage.update_job(job_id, result_bytes=None, partial_bytes=None)
    return response


@router.get("/{job_id}/partial")
def partial_download(job_id: str) -> StreamingResponse:
    job = storage.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Filter to paths that still exist — the TemporaryDirectory may have been
    # deleted between snapshot and ffmpeg call if the job finished concurrently.
    wav_snapshot = [p for p in _active_wav_paths.get(job_id, []) if p.exists()]
    if wav_snapshot:
        try:
            data = audio_chain.ffmpeg_concat_files(wav_snapshot, job.format)
        except Exception:
            logger.warning("on-demand partial concat failed for job %s", job_id, exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to generate partial audio")
    elif job.partial_bytes:
        data = job.partial_bytes
    else:
        raise HTTPException(status_code=409, detail="No partial audio available yet")

    ext = _EXTENSIONS[job.format]
    media_type = _MEDIA_TYPES[job.format]
    return StreamingResponse(
        BytesIO(data),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename=\"audio_partial.{ext}\""},
    )


@router.post("/{job_id}/pause")
def pause_job(job_id: str) -> dict:
    job = storage.get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.processing:
        raise HTTPException(status_code=409, detail="Job is not processing")
    storage.pause_job(job_id)
    return {"job_id": job_id, "status": "paused"}


@router.post("/{job_id}/resume")
def resume_job(job_id: str) -> dict:
    job = storage.get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.paused:
        raise HTTPException(status_code=409, detail="Job is not paused")
    storage.resume_job(job_id)
    return {"job_id": job_id, "status": "processing"}
