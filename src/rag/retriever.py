from __future__ import annotations

from .embedding import HashEmbedding
from .vector_store import LocalVectorStore


class Retriever:
    def __init__(self, store: LocalVectorStore, embedding: HashEmbedding, top_k: int = 4):
        self.store = store
        self.embedding = embedding
        self.top_k = top_k

    def retrieve(self, query: str, filters: dict | None = None, top_k: int | None = None):
        query_vector = self.embedding.encode(query)
        return self.store.similarity_search(
            query_vector=query_vector,
            top_k=top_k or self.top_k,
            filters=filters,
        )
