
from pathlib import Path

import fitz
import pytest

from src.manual_reader import extract_text_from_pdf


def test_extract_text_from_pdf(tmp_path):
    pdf_path = tmp_path / "sample_manual.pdf"

    with fitz.open() as document:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Engine oil should be checked regularly."
        )
        document.save(pdf_path)

    result = extract_text_from_pdf(pdf_path)

    assert len(result) == 1
    assert result[0]["page"] == 1
    assert "Engine oil" in result[0]["text"]


def test_missing_pdf_raises_error(tmp_path):
    missing_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf(missing_path)


def test_non_pdf_file_raises_error(tmp_path):
    text_path = tmp_path / "notes.txt"
    text_path.write_text("Not a PDF", encoding="utf-8")

    with pytest.raises(ValueError):
        extract_text_from_pdf(text_path)
