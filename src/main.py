from __future__ import annotations

from fastapi import FastAPI

from src.rag.config import get_settings
from src.rag.embedding import HashEmbedding
from src.rag.llm_client import LLMClient
from src.rag.retriever import Retriever
from src.rag.schemas import CopilotRequest, CopilotResponse, QARequest, QAResponse, SourceChunk
from src.rag.service import RAGService
from src.rag.vector_store import LocalVectorStore

settings = get_settings()
store = LocalVectorStore(settings.vector_db_path)
store.load()
embedding = HashEmbedding(settings.embed_dim)
retriever = Retriever(store=store, embedding=embedding, top_k=settings.top_k)
llm = LLMClient(settings)
rag_service = RAGService(retriever=retriever, llm_client=llm)

app = FastAPI(title="Project Lifecycle RAG")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "records": len(store.records)}


@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest) -> QAResponse:
    answer, docs = rag_service.answer_question(
        question=req.question,
        filters=req.filters,
        top_k=req.top_k,
    )
    return QAResponse(
        answer=answer,
        sources=[
            SourceChunk(
                chunk_id=d["chunk_id"],
                source=d["source"],
                metadata=d.get("metadata", {}),
                score=d.get("score", 0.0),
                text=d["text"],
            )
            for d in docs
        ],
    )


@app.post("/copilot", response_model=CopilotResponse)
def copilot(req: CopilotRequest) -> CopilotResponse:
    content, docs = rag_service.copilot(
        task=req.task,
        style=req.style,
        filters=req.filters,
    )
    return CopilotResponse(
        content=content,
        sources=[
            SourceChunk(
                chunk_id=d["chunk_id"],
                source=d["source"],
                metadata=d.get("metadata", {}),
                score=d.get("score", 0.0),
                text=d["text"],
            )
            for d in docs
        ],
    )
