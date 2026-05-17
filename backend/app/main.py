from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import upload, jobs
from app import storage
from app.services import tts


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.init_db()
    # Warm up TTS model at startup so the first request isn't slow.
    # Runs in a thread to avoid blocking the event loop during download.
    import asyncio
    await asyncio.get_event_loop().run_in_executor(None, tts.get_sample_rate)
    yield


app = FastAPI(title="PDF-to-Audio", lifespan=lifespan)
app.include_router(upload.router)
app.include_router(jobs.router)

frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
