
import numpy as np
import pytest

from src.embeddings import EmbeddingService


@pytest.fixture(scope="module")
def embedding_service():
    return EmbeddingService()


def test_embed_texts_returns_valid_vectors(embedding_service):
    vectors = embedding_service.embed_texts([
        "Check the engine oil level.",
        "Inspect the car battery.",
    ])

    assert vectors.shape == (2, 384)
    assert np.isfinite(vectors).all()


def test_embeddings_are_normalized(embedding_service):
    vectors = embedding_service.embed_texts([
        "Check the engine oil level."
    ])

    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0, abs=1e-5)


def test_embed_query_returns_one_vector(embedding_service):
    vector = embedding_service.embed_query("Why won't my car start?")

    assert vector.shape == (384,)
    assert np.isfinite(vector).all()


def test_empty_text_list_returns_empty_list(embedding_service):
    assert embedding_service.embed_texts([]) == []


def test_blank_query_is_rejected(embedding_service):
    with pytest.raises(ValueError):
        embedding_service.embed_query("   ")
