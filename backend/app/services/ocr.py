import logging
import re
from pathlib import Path

import fitz
import PIL.Image
import pytesseract

from app.services.normalize import normalize_for_tts

logger = logging.getLogger(__name__)

TEXT_LAYER_MIN_CHARS: int = 50


def extract_pages(pdf_path: Path) -> list[list[str]]:
    with open(pdf_path, "rb") as f:
        magic = f.read(4)
    if magic != b"%PDF":
        raise ValueError("not a PDF")

    result: list[list[str]] = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            raw_text = page.get_text("text").strip()
            if len(raw_text) >= TEXT_LAYER_MIN_CHARS:
                text = raw_text
                extraction_method = "direct"
            else:
                pixmap = page.get_pixmap(dpi=200)
                image = PIL.Image.frombytes(
                    "RGB", [pixmap.width, pixmap.height], pixmap.samples
                )
                text = pytesseract.image_to_string(image)
                extraction_method = "ocr"
            logger.debug(
                "page %d: extraction=%s chars=%d",
                page_num,
                extraction_method,
                len(text),
            )
            parts = re.split(r"(?<=[.?!])\s+", text)
            sentences = [normalize_for_tts(s) for s in parts if s.strip()]
            result.append(sentences)
    return result


def extract_sentences(pdf_path: Path) -> list[str]:
    pages = extract_pages(pdf_path)
    return [sentence for page in pages for sentence in page]
