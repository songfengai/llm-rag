from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "data/vector_store.json")
    embed_dim: int = int(os.getenv("EMBED_DIM", "256"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_settings() -> Settings:
    return Settings()
