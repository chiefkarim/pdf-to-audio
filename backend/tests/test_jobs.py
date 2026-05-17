import uuid

import pytest

from app import storage
from app.models.schemas import ExportFormat, JobStatus


async def test_status_returns_queued(async_client):
    job_id = storage.create_job(ExportFormat.mp3)
    response = await async_client.get(f"/jobs/{job_id}/status")
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == job_id
    assert body["status"] == "queued"


async def test_status_404_unknown(async_client):
    fake_id = str(uuid.uuid4())
    response = await async_client.get(f"/jobs/{fake_id}/status")
    assert response.status_code == 404


async def test_download_409_while_processing(async_client):
    job_id = storage.create_job(ExportFormat.mp3)
    storage.update_job(job_id, status=JobStatus.processing, progress=33)
    response = await async_client.get(f"/jobs/{job_id}/download")
    assert response.status_code == 409


async def test_download_422_when_error(async_client):
    job_id = storage.create_job(ExportFormat.mp3)
    storage.update_job(job_id, status=JobStatus.error, error="No text found in PDF")
    response = await async_client.get(f"/jobs/{job_id}/download")
    assert response.status_code == 422
    assert "No text found in PDF" in response.json()["detail"]


async def test_download_200_when_done(async_client):
    job_id = storage.create_job(ExportFormat.mp3)
    storage.update_job(
        job_id,
        status=JobStatus.done,
        progress=100,
        result_bytes=b"FAKE_AUDIO",
    )
    response = await async_client.get(f"/jobs/{job_id}/download")
    assert response.status_code == 200
    assert response.content == b"FAKE_AUDIO"
    assert response.headers["content-type"] == "audio/mpeg"
    assert "audio.mp3" in response.headers["content-disposition"]


async def test_download_404_unknown(async_client):
    fake_id = str(uuid.uuid4())
    response = await async_client.get(f"/jobs/{fake_id}/download")
    assert response.status_code == 404


async def test_download_content_type_mp3(async_client):
    job_id = storage.create_job(ExportFormat.mp3)
    storage.update_job(job_id, status=JobStatus.done, progress=100, result_bytes=b"FAKE")
    resp = await async_client.get(f"/jobs/{job_id}/download")
    assert "audio/mpeg" in resp.headers["content-type"]


async def test_list_jobs_returns_all(async_client):
    id1 = storage.create_job(ExportFormat.mp3)
    id2 = storage.create_job(ExportFormat.wav)
    response = await async_client.get("/jobs")
    assert response.status_code == 200
    body = response.json()
    ids = [j["job_id"] for j in body]
    assert id1 in ids
    assert id2 in ids


async def test_list_jobs_no_result_bytes(async_client):
    job_id = storage.create_job(ExportFormat.mp3)
    storage.update_job(job_id, status=JobStatus.done, progress=100, result_bytes=b"DATA")
    response = await async_client.get("/jobs")
    assert response.status_code == 200
    for item in response.json():
        assert "result_bytes" not in item


async def test_pause_returns_200(async_client):
    job_id = storage.create_job(ExportFormat.mp3)
    storage.update_job(job_id, status=JobStatus.processing)
    response = await async_client.post(f"/jobs/{job_id}/pause")
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == job_id
    assert body["status"] == "paused"


async def test_pause_409_not_processing(async_client):
    job_id = storage.create_job(ExportFormat.mp3)
    response = await async_client.post(f"/jobs/{job_id}/pause")
    assert response.status_code == 409


async def test_pause_404_unknown(async_client):
    fake_id = str(uuid.uuid4())
    response = await async_client.post(f"/jobs/{fake_id}/pause")
    assert response.status_code == 404


async def test_resume_returns_200(async_client):
    job_id = storage.create_job(ExportFormat.mp3)
    storage.update_job(job_id, status=JobStatus.processing)
    storage.pause_job(job_id)
    response = await async_client.post(f"/jobs/{job_id}/resume")
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == job_id
    assert body["status"] == "processing"


async def test_resume_409_not_paused(async_client):
    job_id = storage.create_job(ExportFormat.mp3)
    response = await async_client.post(f"/jobs/{job_id}/resume")
    assert response.status_code == 409


async def test_resume_404_unknown(async_client):
    fake_id = str(uuid.uuid4())
    response = await async_client.post(f"/jobs/{fake_id}/resume")
    assert response.status_code == 404
