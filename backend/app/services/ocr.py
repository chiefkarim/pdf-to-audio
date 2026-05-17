import re
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path

from app.services.normalize import normalize_for_tts


def extract_pages(pdf_path: Path) -> list[list[str]]:
    with open(pdf_path, "rb") as f:
        magic = f.read(4)
    if magic != b"%PDF":
        raise ValueError("not a PDF")

    images = convert_from_path(pdf_path)
    result: list[list[str]] = []
    for img in images:
        raw_text = pytesseract.image_to_string(img)
        parts = re.split(r"(?<=[.?!])\s+", raw_text)
        sentences = [normalize_for_tts(s) for s in parts if s.strip()]
        result.append(sentences)
    return result


def extract_sentences(pdf_path: Path) -> list[str]:
    pages = extract_pages(pdf_path)
    return [sentence for page in pages for sentence in page]
