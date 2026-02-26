from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QARequest(BaseModel):
    question: str = Field(..., min_length=2)
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int | None = None


class SourceChunk(BaseModel):
    chunk_id: str
    source: str
    metadata: dict[str, Any]
    score: float
    text: str


class QAResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class CopilotRequest(BaseModel):
    task: str = Field(..., min_length=2)
    style: str = Field(default="default")
    filters: dict[str, Any] = Field(default_factory=dict)


class CopilotResponse(BaseModel):
    content: str
    sources: list[SourceChunk]
