
import numpy as np
import pytest

from src.vector_store import VehicleVectorStore


@pytest.fixture
def vector_store(tmp_path):
    return VehicleVectorStore(
        db_path=tmp_path / "chroma_test",
        collection_name="test_vehicle_manuals",
    )


def test_add_and_search_chunks(vector_store):
    chunks = [
        {"page": 3, "text": "Check engine oil before driving."},
        {"page": 8, "text": "Inspect battery terminals for corrosion."},
    ]

    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    vector_store.add_chunks(chunks, embeddings, "test_manual.pdf")

    results = vector_store.search(
        np.array([1.0, 0.0, 0.0]),
        n_results=1,
    )

    assert len(results) == 1
    assert results[0]["page"] == 3
    assert results[0]["source"] == "test_manual.pdf"
    assert "engine oil" in results[0]["text"]


def test_empty_store_returns_no_results(vector_store):
    results = vector_store.search(np.array([1.0, 0.0, 0.0]))

    assert results == []
    assert vector_store.count() == 0


def test_mismatched_chunk_and_embedding_counts_are_rejected(vector_store):
    chunks = [{"page": 1, "text": "Engine oil"}]
    embeddings = np.array([])

    with pytest.raises(ValueError):
        vector_store.add_chunks(chunks, embeddings, "test_manual.pdf")


def test_nonpositive_result_count_is_rejected(vector_store):
    with pytest.raises(ValueError):
        vector_store.search(np.array([1.0, 0.0, 0.0]), n_results=0)
