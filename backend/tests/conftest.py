import sqlite3
from pathlib import Path

import pytest
from fpdf import FPDF
from httpx import ASGITransport, AsyncClient

import app.storage as _storage
from app.main import app

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"


@pytest.fixture(scope="session", autouse=True)
def generate_sample_pdf() -> None:
    FIXTURES_DIR.mkdir(exist_ok=True)
    if SAMPLE_PDF.exists():
        return
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)
    pdf.cell(0, 10, "The quick brown fox jumps over the lazy dog.", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, "Pack my box with five dozen liquor jugs!", new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(SAMPLE_PDF))


@pytest.fixture(autouse=True)
def reset_storage(monkeypatch) -> None:
    conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
    monkeypatch.setattr(_storage, "_conn", conn)
    monkeypatch.setattr(_storage, "_pause_events", {})
    _storage.init_db()


@pytest.fixture
async def async_client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
