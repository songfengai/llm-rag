from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalVectorStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.records: list[dict[str, Any]] = []

    def load(self) -> None:
        if self.path.exists():
            self.records = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def upsert(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    def similarity_search(
        self,
        query_vector: list[float],
        top_k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        filters = filters or {}

        candidates = []
        for row in self.records:
            metadata = row.get("metadata", {})
            if any(metadata.get(k) != v for k, v in filters.items()):
                continue

            score = sum(a * b for a, b in zip(query_vector, row["embedding"]))
            candidates.append((score, row))

        candidates.sort(key=lambda x: x[0], reverse=True)
        result = []
        for score, row in candidates[:top_k]:
            r = dict(row)
            r["score"] = score
            result.append(r)
        return result
