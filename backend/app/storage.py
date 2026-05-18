import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.models.schemas import ExportFormat, Job, JobStatus, TtsMode

_db_path: str = os.environ.get("DB_PATH", "jobs.db")
_conn: sqlite3.Connection = sqlite3.connect(_db_path, check_same_thread=False, isolation_level=None)
_lock = threading.Lock()
_pause_events: dict[str, threading.Event] = {}

_STALE_STATUSES = (JobStatus.processing.value, JobStatus.queued.value)


def init_db() -> None:
    with _lock:
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id        TEXT PRIMARY KEY,
                status        TEXT NOT NULL,
                progress      INTEGER NOT NULL DEFAULT 0,
                format        TEXT NOT NULL,
                result_bytes  BLOB,
                error         TEXT,
                created_at    TEXT NOT NULL,
                pages_total   INTEGER NOT NULL DEFAULT 0,
                pages_done    INTEGER NOT NULL DEFAULT 0,
                partial_bytes BLOB,
                filename      TEXT NOT NULL DEFAULT '',
                mode          TEXT NOT NULL DEFAULT 'fast',
                ocr_pages     INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        # Migration for existing DBs that lack the new columns
        for col, definition in [
            ("pages_total", "INTEGER NOT NULL DEFAULT 0"),
            ("pages_done", "INTEGER NOT NULL DEFAULT 0"),
            ("partial_bytes", "BLOB"),
            ("filename", "TEXT NOT NULL DEFAULT ''"),
            ("mode", "TEXT NOT NULL DEFAULT 'fast'"),
            ("ocr_pages", "INTEGER NOT NULL DEFAULT 0"),
        ]:
            try:
                _conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e):
                    raise

        placeholders = ",".join("?" for _ in _STALE_STATUSES)
        _conn.execute(
            f"UPDATE jobs SET status = ?, error = ? WHERE status IN ({placeholders})",
            (
                JobStatus.error.value,
                "Server restarted while job was running",
                *_STALE_STATUSES,
            ),
        )


def _row_to_job(row: tuple) -> Job:
    job_id, status, progress, fmt, result_bytes, error, created_at, pages_total, pages_done, partial_bytes, filename, mode, ocr_pages = row
    return Job(
        job_id=job_id,
        status=JobStatus(status),
        progress=progress,
        format=ExportFormat(fmt),
        result_bytes=result_bytes,
        error=error,
        created_at=datetime.fromisoformat(created_at),
        pages_total=pages_total or 0,
        pages_done=pages_done or 0,
        partial_bytes=partial_bytes,
        filename=filename or "",
        mode=TtsMode(mode),
        ocr_pages=ocr_pages or 0,
    )


def create_job(fmt: ExportFormat, filename: str = "", mode: TtsMode = TtsMode.fast) -> str:
    job_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    event = threading.Event()
    event.set()
    with _lock:
        _conn.execute(
            "INSERT INTO jobs (job_id, status, progress, format, created_at, filename, mode) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (job_id, JobStatus.queued.value, 0, fmt.value, created_at, filename, mode.value),
        )
        _pause_events[job_id] = event
    return job_id


def get_job(job_id: str) -> Optional[Job]:
    with _lock:
        row = _conn.execute(
            """
            SELECT job_id, status, progress, format, result_bytes, error, created_at,
                   pages_total, pages_done, partial_bytes, filename, mode, ocr_pages
            FROM jobs WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    return _row_to_job(row)


def update_job(job_id: str, **kwargs) -> None:
    if not kwargs:
        return
    allowed = {"status", "progress", "result_bytes", "error", "pages_total", "pages_done", "partial_bytes", "filename", "ocr_pages"}
    sets = []
    values = []
    for field, value in kwargs.items():
        if field not in allowed:
            raise ValueError(f"Cannot update field: {field}")
        sets.append(f"{field} = ?")
        if isinstance(value, JobStatus):
            values.append(value.value)
        else:
            values.append(value)
    values.append(job_id)
    with _lock:
        cursor = _conn.execute(
            f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?",
            values,
        )
        if cursor.rowcount == 0:
            raise KeyError(job_id)
        if kwargs.get("status") in (JobStatus.done, JobStatus.done.value, JobStatus.error, JobStatus.error.value):
            _pause_events.pop(job_id, None)


def _row_to_job_slim(row: tuple) -> Job:
    job_id, status, progress, fmt, error, created_at, pages_total, pages_done, filename, mode, ocr_pages = row
    return Job(
        job_id=job_id,
        status=JobStatus(status),
        progress=progress,
        format=ExportFormat(fmt),
        result_bytes=None,
        error=error,
        created_at=datetime.fromisoformat(created_at),
        pages_total=pages_total or 0,
        pages_done=pages_done or 0,
        partial_bytes=None,
        filename=filename or "",
        mode=TtsMode(mode),
        ocr_pages=ocr_pages or 0,
    )


def list_jobs() -> list[Job]:
    with _lock:
        rows = _conn.execute(
            """
            SELECT job_id, status, progress, format, error, created_at,
                   pages_total, pages_done, filename, mode, ocr_pages
            FROM jobs ORDER BY created_at DESC
            """
        ).fetchall()
    return [_row_to_job_slim(row) for row in rows]


def get_job_status(job_id: str) -> Optional[Job]:
    with _lock:
        row = _conn.execute(
            "SELECT job_id, status, progress FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    job_id_, status, progress = row
    return Job(
        job_id=job_id_,
        status=JobStatus(status),
        progress=progress,
        format=ExportFormat.mp3,
        result_bytes=None,
        error=None,
        created_at=datetime.now(timezone.utc),
        pages_total=0,
        pages_done=0,
        partial_bytes=None,
        filename="",
        mode=TtsMode.fast,
        ocr_pages=0,
    )


def get_pause_event(job_id: str) -> Optional[threading.Event]:
    return _pause_events.get(job_id)


def pause_job(job_id: str) -> None:
    event = _pause_events.get(job_id)
    if event is not None:
        event.clear()
    with _lock:
        cursor = _conn.execute(
            "UPDATE jobs SET status = ? WHERE job_id = ?",
            (JobStatus.paused.value, job_id),
        )
        if cursor.rowcount == 0:
            raise KeyError(job_id)


def resume_job(job_id: str) -> None:
    event = _pause_events.get(job_id)
    if event is not None:
        event.set()
    with _lock:
        cursor = _conn.execute(
            "UPDATE jobs SET status = ? WHERE job_id = ?",
            (JobStatus.processing.value, job_id),
        )
        if cursor.rowcount == 0:
            raise KeyError(job_id)
