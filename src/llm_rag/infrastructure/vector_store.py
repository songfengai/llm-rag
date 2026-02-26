from typing import Sequence

from llm_rag.domain.interfaces import EmbeddingModel, VectorStore
from llm_rag.domain.models import Document, RetrievedChunk


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class InMemoryVectorStore(VectorStore):
    """In-memory vector index implementation."""

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self._embedding_model = embedding_model
        self._rows: list[tuple[Document, list[float]]] = []

    def upsert(self, documents: Sequence[Document]) -> None:
        existing = {doc.doc_id: i for i, (doc, _) in enumerate(self._rows)}
        for doc in documents:
            vec = self._embedding_model.embed(doc.content)
            idx = existing.get(doc.doc_id)
            if idx is None:
                self._rows.append((doc, vec))
            else:
                self._rows[idx] = (doc, vec)

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        q_vec = self._embedding_model.embed(query)
        scored = [
            RetrievedChunk(document=doc, score=_dot(vec, q_vec))
            for doc, vec in self._rows
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: max(top_k, 0)]
