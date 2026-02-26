import math
import re
from collections import Counter

from llm_rag.domain.interfaces import EmbeddingModel


class BagOfWordsEmbedding(EmbeddingModel):
    """Simple deterministic embedding for local demo/testing."""

    _token_pattern = re.compile(r"[a-zA-Z0-9_\u4e00-\u9fff]+")

    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        tokens = self._token_pattern.findall(text.lower())
        counts = Counter(tokens)

        for token, count in counts.items():
            index = hash(token) % self._dimensions
            vector[index] += float(count)

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]
