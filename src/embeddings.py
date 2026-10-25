
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class EmbeddingService:
    """Generate vector embeddings for vehicle-manual text."""

    def __init__(self, model_name=DEFAULT_EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        """Convert a list of text strings into embedding vectors."""
        if not texts:
            return []

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every text must be a non-empty string.")

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return vectors

    def embed_query(self, query):
        """Convert one search query into an embedding vector."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        return self.embed_texts([query])[0]
