from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import upload, jobs
from app import storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.init_db()
    yield


app = FastAPI(title="PDF-to-Audio", lifespan=lifespan)
app.include_router(upload.router)
app.include_router(jobs.router)

frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
