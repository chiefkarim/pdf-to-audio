from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class ExportFormat(str, Enum):
    mp3 = "mp3"
    wav = "wav"

class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    paused = "paused"
    done = "done"
    error = "error"

class TtsMode(str, Enum):
    fast = "fast"
    quality = "quality"

class Job(BaseModel):
    job_id: str
    status: JobStatus
    progress: int
    format: ExportFormat
    result_bytes: Optional[bytes] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pages_total: int = 0
    pages_done: int = 0
    partial_bytes: Optional[bytes] = None
    filename: str = ""
    mode: TtsMode = TtsMode.fast
