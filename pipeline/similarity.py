from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    av = np.asarray(a, dtype=np.float32).reshape(-1)
    bv = np.asarray(b, dtype=np.float32).reshape(-1)

    if av.size == 0 or bv.size == 0 or av.size != bv.size:
        return 0.0

    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0.0:
        return 0.0

    return float(np.dot(av, bv) / denom)


def top_k_by_cosine(
    query_vector: Sequence[float], candidates: Iterable[Sequence[float]], top_k: int = 3
) -> List[Tuple[int, float]]:
    scored: List[Tuple[int, float]] = []

    for idx, vector in enumerate(candidates):
        score = cosine(query_vector, vector)
        scored.append((idx, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(top_k, 0)]
