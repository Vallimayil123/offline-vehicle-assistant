
import pytest

from src.text_chunker import chunk_pages


def test_short_page_creates_one_chunk():
    pages = [{"page": 1, "text": "Engine oil must be checked regularly."}]

    chunks = chunk_pages(pages)

    assert len(chunks) == 1
    assert chunks[0]["page"] == 1
    assert "Engine oil" in chunks[0]["text"]


def test_long_page_creates_overlapping_chunks():
    pages = [{"page": 4, "text": "A" * 2500}]

    chunks = chunk_pages(pages, chunk_size=1000, overlap=200)

    assert len(chunks) == 3
    assert all(chunk["page"] == 4 for chunk in chunks)
    assert len(chunks[0]["text"]) == 1000
    assert len(chunks[1]["text"]) == 1000
    assert len(chunks[2]["text"]) == 900


def test_empty_pages_are_skipped():
    pages = [
        {"page": 1, "text": "   "},
        {"page": 2, "text": "Battery maintenance"},
    ]

    chunks = chunk_pages(pages)

    assert len(chunks) == 1
    assert chunks[0]["page"] == 2


def test_invalid_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=0)


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=100, overlap=100)
