from typing import Sequence

from llm_rag.domain.interfaces import LLMClient
from llm_rag.domain.models import RetrievedChunk


class RuleBasedLLM(LLMClient):
    """A lightweight LLM adapter used to validate architecture integration."""

    def generate(self, question: str, contexts: Sequence[RetrievedChunk]) -> str:
        if not contexts:
            return f"问题：{question}\n当前知识库未命中相关内容。"

        top = contexts[0]
        preview = top.document.content[:200]
        source = top.document.metadata.get("source", top.document.doc_id) if top.document.metadata else top.document.doc_id
        return (
            f"问题：{question}\n"
            f"依据来源：{source}\n"
            f"参考内容：{preview}\n"
            "结论：以上是基于检索结果生成的回答，请结合业务规则复核。"
        )
