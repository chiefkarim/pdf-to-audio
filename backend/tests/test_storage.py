import time

import pytest

from app.models.schemas import ExportFormat, JobStatus
from app.storage import (
    create_job,
    get_job,
    get_job_status,
    list_jobs,
    pause_job,
    resume_job,
    update_job,
)


def test_create_job_returns_string() -> None:
    job_id = create_job(ExportFormat.mp3)
    assert isinstance(job_id, str)
    assert len(job_id) == 36


def test_get_job_unknown_returns_none() -> None:
    assert get_job("00000000-0000-0000-0000-000000000000") is None


def test_get_job_returns_created_job() -> None:
    job_id = create_job(ExportFormat.wav)
    job = get_job(job_id)
    assert job is not None
    assert job.job_id == job_id
    assert job.status == JobStatus.queued
    assert job.format == ExportFormat.wav


def test_update_job_mutates_status_and_progress() -> None:
    job_id = create_job(ExportFormat.mp3)
    update_job(job_id, status=JobStatus.processing, progress=50)
    job = get_job(job_id)
    assert job.status == JobStatus.processing
    assert job.progress == 50


def test_update_job_unknown_raises_key_error() -> None:
    with pytest.raises(KeyError):
        update_job("00000000-0000-0000-0000-000000000000", status=JobStatus.done)


def test_update_job_sets_result_bytes() -> None:
    job_id = create_job(ExportFormat.mp3)
    update_job(job_id, status=JobStatus.done, progress=100, result_bytes=b"audio")
    job = get_job(job_id)
    assert job.result_bytes == b"audio"


def test_list_jobs_returns_all() -> None:
    id1 = create_job(ExportFormat.mp3)
    id2 = create_job(ExportFormat.wav)
    jobs = list_jobs()
    ids = [j.job_id for j in jobs]
    assert id1 in ids
    assert id2 in ids


def test_list_jobs_ordered_newest_first() -> None:
    id1 = create_job(ExportFormat.mp3)
    time.sleep(0.01)
    id2 = create_job(ExportFormat.mp3)
    jobs = list_jobs()
    ids = [j.job_id for j in jobs]
    assert ids.index(id2) < ids.index(id1)


def test_pause_job_clears_event_and_sets_status() -> None:
    job_id = create_job(ExportFormat.mp3)
    update_job(job_id, status=JobStatus.processing)
    pause_job(job_id)
    job = get_job(job_id)
    assert job.status == JobStatus.paused
    from app import storage
    event = storage.get_pause_event(job_id)
    assert event is not None
    assert not event.is_set()


def test_resume_job_sets_event_and_sets_status() -> None:
    job_id = create_job(ExportFormat.mp3)
    update_job(job_id, status=JobStatus.processing)
    pause_job(job_id)
    resume_job(job_id)
    job = get_job(job_id)
    assert job.status == JobStatus.processing
    from app import storage
    event = storage.get_pause_event(job_id)
    assert event is not None
    assert event.is_set()


def test_pause_unknown_raises_key_error() -> None:
    with pytest.raises(KeyError):
        pause_job("00000000-0000-0000-0000-000000000000")


def test_resume_unknown_raises_key_error() -> None:
    with pytest.raises(KeyError):
        resume_job("00000000-0000-0000-0000-000000000000")


def test_get_job_status_unknown_returns_none() -> None:
    assert get_job_status("00000000-0000-0000-0000-000000000000") is None


def test_get_job_status_returns_correct_fields() -> None:
    job_id = create_job(ExportFormat.mp3)
    update_job(job_id, status=JobStatus.processing, progress=42)
    result = get_job_status(job_id)
    assert result is not None
    assert result.job_id == job_id
    assert result.status == JobStatus.processing
    assert result.progress == 42


def test_get_job_status_does_not_load_result_bytes() -> None:
    job_id = create_job(ExportFormat.mp3)
    update_job(job_id, status=JobStatus.done, progress=100, result_bytes=b"audio")
    result = get_job_status(job_id)
    assert result is not None
    assert result.result_bytes is None


def test_list_jobs_omits_result_bytes() -> None:
    job_id = create_job(ExportFormat.mp3)
    update_job(job_id, status=JobStatus.done, progress=100, result_bytes=b"audio")
    jobs = list_jobs()
    match = next(j for j in jobs if j.job_id == job_id)
    assert match.result_bytes is None


def test_list_jobs_omits_partial_bytes() -> None:
    job_id = create_job(ExportFormat.mp3)
    update_job(job_id, status=JobStatus.processing, partial_bytes=b"chunk")
    jobs = list_jobs()
    match = next(j for j in jobs if j.job_id == job_id)
    assert match.partial_bytes is None


def test_update_job_done_removes_pause_event() -> None:
    from app import storage
    job_id = create_job(ExportFormat.mp3)
    assert job_id in storage._pause_events
    update_job(job_id, status=JobStatus.done)
    assert job_id not in storage._pause_events


def test_update_job_error_removes_pause_event() -> None:
    from app import storage
    job_id = create_job(ExportFormat.mp3)
    assert job_id in storage._pause_events
    update_job(job_id, status=JobStatus.error, error="x")
    assert job_id not in storage._pause_events
