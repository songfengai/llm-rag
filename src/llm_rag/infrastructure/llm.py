import json
import os
from typing import Sequence
from urllib import error, request

from llm_rag.domain.interfaces import LLMClient
from llm_rag.domain.models import RetrievedChunk


class QwenLLM(LLMClient):
    """Qwen API client using DashScope OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "qwen-plus",
        endpoint: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if not self._api_key:
            raise ValueError("Missing API key. Set DASHSCOPE_API_KEY or QWEN_API_KEY.")
        self._model = model
        self._endpoint = endpoint
        self._timeout = timeout

    def generate(self, question: str, contexts: Sequence[RetrievedChunk]) -> str:
        context_text = "\n".join(
            f"[{idx + 1}] {chunk.document.content}" for idx, chunk in enumerate(contexts)
        )
        if not context_text:
            context_text = "未检索到相关知识。"

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是企业知识库问答助手。请严格基于提供的上下文回答，不要臆测。",
                },
                {
                    "role": "user",
                    "content": f"问题：{question}\n\n可用上下文：\n{context_text}",
                },
            ],
            "temperature": 0.2,
        }

        req = request.Request(
            self._endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Qwen API HTTP error: {exc.code} {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Qwen API network error: {exc.reason}") from exc

        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError(f"Qwen API invalid response: {body}")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not content:
            raise RuntimeError(f"Qwen API empty content: {body}")

        return content.strip()
