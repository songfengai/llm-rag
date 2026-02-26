from __future__ import annotations

from .llm_client import LLMClient
from .retriever import Retriever


class RAGService:
    def __init__(self, retriever: Retriever, llm_client: LLMClient) -> None:
        self.retriever = retriever
        self.llm_client = llm_client

    def answer_question(self, question: str, filters: dict | None = None, top_k: int | None = None):
        docs = self.retriever.retrieve(question, filters=filters, top_k=top_k)
        context = self._build_context(docs)
        prompt = (
            f"问题：{question}\n\n"
            f"参考资料：\n{context}\n\n"
            "请使用中文回答，并给出关键依据。"
        )
        answer = self.llm_client.generate(prompt)
        return answer, docs

    def copilot(self, task: str, style: str = "default", filters: dict | None = None):
        docs = self.retriever.retrieve(task, filters=filters)
        context = self._build_context(docs)
        prompt = (
            f"任务：{task}\n"
            f"输出风格：{style}\n\n"
            f"参考资料：\n{context}\n\n"
            "请输出结构化内容，必要时用清单形式。"
        )
        content = self.llm_client.generate(prompt)
        return content, docs

    @staticmethod
    def _build_context(docs: list[dict]) -> str:
        if not docs:
            return "（未检索到相关资料）"

        lines = []
        for d in docs:
            lines.append(
                f"- [{d['chunk_id']}] source={d['source']} score={d.get('score', 0):.4f}\n"
                f"  {d['text']}"
            )
        return "\n".join(lines)
