from typing import Protocol, Sequence

from llm_rag.domain.models import Document, RetrievedChunk


class EmbeddingModel(Protocol):
    def embed(self, text: str) -> list[float]:
        """Convert text to dense vector representation."""


class VectorStore(Protocol):
    def upsert(self, documents: Sequence[Document]) -> None:
        """Persist documents into the index."""

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Return top-k most relevant chunks."""


class LLMClient(Protocol):
    def generate(self, question: str, contexts: Sequence[RetrievedChunk]) -> str:
        """Generate answer from question and retrieved contexts."""
