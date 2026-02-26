from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoadedChunk:
    chunk_id: str
    source: str
    text: str
    metadata: dict


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _infer_metadata(file_path: Path) -> dict:
    stem = file_path.stem.lower()
    phase = "general"
    if "presale" in stem or "proposal" in stem:
        phase = "presale"
    elif "dev" in stem or "code" in stem:
        phase = "dev"
    elif "delivery" in stem:
        phase = "delivery"

    return {
        "phase": phase,
        "artifact_type": "document",
        "project_code": "demo",
    }


def load_documents(input_dir: str) -> list[LoadedChunk]:
    root = Path(input_dir)
    files = [
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in {".md", ".txt"}
    ]

    chunks: list[LoadedChunk] = []
    for fp in files:
        content = fp.read_text(encoding="utf-8")
        metadata = _infer_metadata(fp)
        for idx, chunk in enumerate(_chunk_text(content)):
            chunk_id = f"{fp.stem}-{idx}"
            chunks.append(
                LoadedChunk(
                    chunk_id=chunk_id,
                    source=fp.name,
                    text=chunk,
                    metadata=metadata,
                )
            )
    return chunks
