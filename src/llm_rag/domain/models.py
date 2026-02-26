from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Document:
    """Base document unit for indexing and retrieval."""

    doc_id: str
    content: str
    metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    """Retrieved item with score."""

    document: Document
    score: float


@dataclass(frozen=True)
class RAGResult:
    """End-to-end pipeline output."""

    question: str
    answer: str
    contexts: Sequence[RetrievedChunk]
