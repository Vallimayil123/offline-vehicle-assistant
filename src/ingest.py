
from pathlib import Path

from src.embeddings import EmbeddingService
from src.manual_reader import extract_text_from_pdf
from src.text_chunker import chunk_pages
from src.vector_store import VehicleVectorStore


class ManualIngestor:
    """Process vehicle manuals and search their contents."""

    def __init__(self, db_path=None):
        self.embedding_service = EmbeddingService()

        if db_path is None:
            self.vector_store = VehicleVectorStore()
        else:
            self.vector_store = VehicleVectorStore(db_path=db_path)

    def ingest_pdf(self, pdf_path):
        pdf_path = Path(pdf_path)

        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_pages(pages)

        if not chunks:
            raise ValueError(
                f"No extractable text found in PDF: {pdf_path.name}"
            )

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
            source_name=pdf_path.name,
        )

        return {
            "filename": pdf_path.name,
            "pages_with_text": len(pages),
            "chunks_created": len(chunks),
            "chunks_stored": self.vector_store.count(),
        }

    def search_manual(self, question, n_results=3):
        query_embedding = self.embedding_service.embed_query(question)

        return self.vector_store.search(
            query_embedding=query_embedding,
            n_results=n_results,
        )
