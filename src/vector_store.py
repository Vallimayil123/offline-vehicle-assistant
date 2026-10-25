
from pathlib import Path

import chromadb


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chroma"


class VehicleVectorStore:
    """Store and search vehicle-manual chunks locally."""

    def __init__(self, db_path=DEFAULT_DB_PATH, collection_name="vehicle_manuals"):
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks, embeddings, source_name):
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have one embedding.")

        if not chunks:
            return

        ids = [
            f"{source_name}_page{chunk['page']}_chunk{i}"
            for i, chunk in enumerate(chunks)
        ]

        self.collection.upsert(
            ids=ids,
            documents=[chunk["text"] for chunk in chunks],
            embeddings=[embedding.tolist() for embedding in embeddings],
            metadatas=[
                {"page": chunk["page"], "source": source_name}
                for chunk in chunks
            ],
        )

    def search(self, query_embedding, n_results=3):
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero.")

        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        matches = []

        for i, document in enumerate(results["documents"][0]):
            matches.append({
                "text": document,
                "page": results["metadatas"][0][i]["page"],
                "source": results["metadatas"][0][i]["source"],
                "distance": results["distances"][0][i],
            })

        return matches

    def count(self):
        return self.collection.count()
