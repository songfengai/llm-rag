from __future__ import annotations

import hashlib
import math


class HashEmbedding:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = [tok for tok in text.lower().split() if tok]
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dim
            sign = -1.0 if int(digest[8:10], 16) % 2 == 0 else 1.0
            vector[index] += sign

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector
