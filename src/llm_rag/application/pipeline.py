from llm_rag.domain.interfaces import LLMClient, VectorStore
from llm_rag.domain.models import Document, RAGResult


class RAGPipeline:
    """Application service that orchestrates retrieval + generation."""

    def __init__(self, vector_store: VectorStore, llm_client: LLMClient) -> None:
        self._vector_store = vector_store
        self._llm_client = llm_client

    def ingest(self, documents: list[Document]) -> None:
        self._vector_store.upsert(documents)

    def ask(self, question: str, top_k: int = 3) -> RAGResult:
        contexts = self._vector_store.search(question, top_k=top_k)
        answer = self._llm_client.generate(question, contexts)
        return RAGResult(question=question, answer=answer, contexts=contexts)
