from typing import Sequence

from llm_rag.application.pipeline import RAGPipeline
from llm_rag.domain.interfaces import LLMClient
from llm_rag.domain.models import Document, RetrievedChunk
from llm_rag.infrastructure.embeddings import BagOfWordsEmbedding
from llm_rag.infrastructure.vector_store import InMemoryVectorStore


class FakeLLM(LLMClient):
    def generate(self, question: str, contexts: Sequence[RetrievedChunk]) -> str:
        if not contexts:
            return "no-context"
        return f"{question}:{contexts[0].document.content}"


def test_pipeline_returns_contextual_answer() -> None:
    store = InMemoryVectorStore(BagOfWordsEmbedding())
    pipeline = RAGPipeline(store, FakeLLM())
    pipeline.ingest(
        [
            Document(doc_id="1", content="订单系统支持退款审核流程", metadata={"source": "ops"}),
            Document(doc_id="2", content="库存服务负责扣减和预占", metadata={"source": "inventory"}),
        ]
    )

    result = pipeline.ask("退款流程是哪个系统负责？", top_k=1)

    assert result.contexts
    assert result.contexts[0].document.doc_id == "1"
    assert "退款" in result.answer
