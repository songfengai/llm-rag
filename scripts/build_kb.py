from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.config import Settings
from src.rag.document_loader import load_documents
from src.rag.embedding import HashEmbedding
from src.rag.vector_store import LocalVectorStore


def build(input_dir: str, db_path: str, embed_dim: int) -> None:
    chunks = load_documents(input_dir)
    embedder = HashEmbedding(embed_dim)
    store = LocalVectorStore(db_path)

    for chunk in chunks:
        store.upsert(
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "embedding": embedder.encode(chunk.text),
            }
        )

    store.save()
    print(f"Indexed {len(chunks)} chunks into {db_path}")


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Build local knowledge base")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--db_path", default=settings.vector_db_path)
    parser.add_argument("--embed_dim", type=int, default=settings.embed_dim)
    args = parser.parse_args()

    build(args.input_dir, args.db_path, args.embed_dim)


if __name__ == "__main__":
    main()
