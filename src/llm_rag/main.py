import os

from llm_rag.application.pipeline import RAGPipeline
from llm_rag.domain.models import Document
from llm_rag.infrastructure.embeddings import BagOfWordsEmbedding
from llm_rag.infrastructure.llm import QwenLLM
from llm_rag.infrastructure.vector_store import InMemoryVectorStore


def build_pipeline() -> RAGPipeline:
    embedding = BagOfWordsEmbedding(dimensions=512)
    store = InMemoryVectorStore(embedding)
    llm = QwenLLM(api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY"))
    pipeline = RAGPipeline(vector_store=store, llm_client=llm)
    pipeline.ingest(
        [
            Document(
                doc_id="arch-1",
                content="RAG 架构一般分为数据接入、检索召回、重排与答案生成层。",
                metadata={"source": "architecture-notes"},
            ),
            Document(
                doc_id="arch-2",
                content="工程上建议通过领域层接口解耦 embedding、vector store 和 LLM。",
                metadata={"source": "engineering-guide"},
            ),
        ]
    )
    return pipeline


def run_demo(question: str) -> str:
    pipeline = build_pipeline()
    return pipeline.ask(question, top_k=2).answer


if __name__ == "__main__":
    print(run_demo("如何在项目中落地 RAG 架构？"))
