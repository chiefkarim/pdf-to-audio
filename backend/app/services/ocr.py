import logging
import re
from pathlib import Path
from typing import Iterator

import fitz
import PIL.Image
import pytesseract

from app.services.normalize import normalize_for_tts

logger = logging.getLogger(__name__)

TEXT_LAYER_MIN_CHARS: int = 50


def page_count(pdf_path: Path) -> int:
    with fitz.open(pdf_path) as doc:
        return len(doc)


def stream_pages(pdf_path: Path) -> Iterator[tuple[int, list[str], bool]]:
    """Yield (page_idx, sentences, is_ocr) one page at a time."""
    with open(pdf_path, "rb") as f:
        if f.read(4) != b"%PDF":
            raise ValueError("not a PDF")

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            raw_text = page.get_text("text").strip()
            if len(raw_text) >= TEXT_LAYER_MIN_CHARS:
                text = raw_text
                is_ocr = False
            else:
                pixmap = page.get_pixmap(dpi=200, alpha=False)
                image = PIL.Image.frombytes(
                    "RGB", [pixmap.width, pixmap.height], pixmap.samples
                )
                text = pytesseract.image_to_string(image)
                del image, pixmap
                is_ocr = True
            logger.debug(
                "page %d: extraction=%s chars=%d",
                page_num,
                "ocr" if is_ocr else "direct",
                len(text),
            )
            parts = re.split(r"(?<=[.?!])\s+|\n{2,}", text)
            sentences = [normalize_for_tts(s) for s in parts if s.strip()]
            yield page_num, sentences, is_ocr


def extract_pages(pdf_path: Path) -> list[list[str]]:
    return [sentences for _, sentences, _ in stream_pages(pdf_path)]


def extract_sentences(pdf_path: Path) -> list[str]:
    return [s for page in extract_pages(pdf_path) for s in page]
