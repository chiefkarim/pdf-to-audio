from pathlib import Path

import pytest

from app.services.ocr import extract_sentences


SAMPLE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"


def test_extract_sentences_returns_list(generate_sample_pdf: None) -> None:
    result = extract_sentences(SAMPLE_PDF)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(s, str) for s in result)


def test_extract_sentences_filters_empty(generate_sample_pdf: None) -> None:
    result = extract_sentences(SAMPLE_PDF)
    assert all(s != "" for s in result)


def test_extract_sentences_invalid_file(tmp_path: Path) -> None:
    not_a_pdf = tmp_path / "fake.pdf"
    not_a_pdf.write_bytes(b"this is not a pdf file")
    with pytest.raises(ValueError, match="not a PDF"):
        extract_sentences(not_a_pdf)
