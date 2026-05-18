import asyncio
import io

import numpy as np
import pytest

from app.main import app


@pytest.fixture(autouse=True)
def patch_services(monkeypatch):
    monkeypatch.setattr("app.routers.upload.ocr.page_count", lambda p: 1)
    monkeypatch.setattr(
        "app.routers.upload.ocr.stream_pages",
        lambda p: iter([(0, ["Hello world."], False)]),
    )
    monkeypatch.setattr(
        "app.routers.upload.tts.synthesise_parallel",
        lambda sentences, mode, executor: [np.zeros(22050, dtype=np.float32)],
    )
    monkeypatch.setattr(
        "app.routers.upload.audio_chain.apply_dsp",
        lambda segments, sr: segments,
    )
    monkeypatch.setattr(
        "app.routers.upload.audio_chain.encode",
        lambda *a, **k: b"FAKE_WAV",
    )
    monkeypatch.setattr(
        "app.routers.upload.audio_chain.ffmpeg_concat_files",
        lambda *a, **k: b"FAKE_AUDIO",
    )


async def test_upload_valid_pdf_returns_202(async_client):
    pdf_bytes = b"%PDF-1.4 minimal content for testing purposes"
    response = await async_client.post(
        "/upload",
        files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"format": "mp3"},
    )
    assert response.status_code == 202
    body = response.json()
    assert "job_id" in body
    assert body["status"] == "queued"


async def test_upload_non_pdf_returns_400(async_client):
    not_pdf = b"PK\x03\x04this is a zip file"
    response = await async_client.post(
        "/upload",
        files={"file": ("evil.pdf", io.BytesIO(not_pdf), "application/pdf")},
        data={"format": "mp3"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "File is not a valid PDF"


async def test_upload_too_large_returns_413(async_client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0")
    pdf_bytes = b"%PDF-1.4 any content here"
    response = await async_client.post(
        "/upload",
        files={"file": ("big.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"format": "mp3"},
    )
    assert response.status_code == 413


async def test_upload_background_sets_done(async_client):
    pdf_bytes = b"%PDF-1.4 1 0 obj\n"
    resp = await async_client.post(
        "/upload",
        data={"format": "mp3"},
        files={"file": ("t.pdf", pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    await asyncio.sleep(0.2)
    status_resp = await async_client.get(f"/jobs/{job_id}/status")
    assert status_resp.json()["status"] == "done"
