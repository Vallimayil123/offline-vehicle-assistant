
import fitz
import pytest

from src.ingest import ManualIngestor


@pytest.fixture
def sample_manual(tmp_path):
    pdf_path = tmp_path / "sample_vehicle_manual.pdf"

    with fitz.open() as document:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Engine oil: Check the oil level regularly "
            "and use the recommended engine oil grade."
        )

        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Battery maintenance: Keep battery terminals "
            "clean and inspect them for corrosion."
        )

        document.save(pdf_path)

    return pdf_path


def test_ingest_and_search_manual(sample_manual, tmp_path):
    ingestor = ManualIngestor(db_path=tmp_path / "test_chroma")

    summary = ingestor.ingest_pdf(sample_manual)

    assert summary["filename"] == "sample_vehicle_manual.pdf"
    assert summary["pages_with_text"] == 2
    assert summary["chunks_created"] == 2
    assert summary["chunks_stored"] == 2

    results = ingestor.search_manual(
        "How should I check engine oil?",
        n_results=1,
    )

    assert len(results) == 1
    assert results[0]["page"] == 1
    assert "Engine oil" in results[0]["text"]


def test_ingest_rejects_pdf_without_extractable_text(tmp_path):
    pdf_path = tmp_path / "empty.pdf"

    with fitz.open() as document:
        document.new_page()
        document.save(pdf_path)

    ingestor = ManualIngestor(db_path=tmp_path / "empty_test_chroma")

    with pytest.raises(ValueError, match="No extractable text"):
        ingestor.ingest_pdf(pdf_path)
