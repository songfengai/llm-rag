from __future__ import annotations

import json
from typing import Any
from urllib import request

from .config import Settings


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, prompt: str) -> str:
        if not (self.settings.openai_base_url and self.settings.openai_api_key):
            return (
                "[离线演示模式]\n"
                "以下是基于检索资料生成的建议答案：\n\n"
                f"{prompt[:1200]}"
            )

        url = f"{self.settings.openai_base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.settings.openai_model,
            "messages": [
                {"role": "system", "content": "你是企业知识助手，请根据给定资料回答并保持准确。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }

        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
