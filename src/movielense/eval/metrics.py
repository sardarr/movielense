"""Top-K ranking metrics.

Inputs are per-user. For each user we have:
  - ranked: array of item ids ordered by predicted score (descending), length K
  - relevant: set of held-out positive item ids

All metrics are macro-averaged across users. Users with no held-out positives
are skipped.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def hit_rate_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    return 1.0 if any(item in relevant for item in ranked[:k]) else 0.0


def precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / k


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / len(relevant)


def average_precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)


def reciprocal_rank_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    for i, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 1)
    n_rel = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


METRIC_FNS = {
    "hr": hit_rate_at_k,
    "precision": precision_at_k,
    "recall": recall_at_k,
    "map": average_precision_at_k,
    "mrr": reciprocal_rank_at_k,
    "ndcg": ndcg_at_k,
}


def evaluate_user_rankings(
    rankings: Iterable[tuple[Sequence[int], set[int]]],
    ks: Sequence[int],
    metrics: Sequence[str] = ("hr", "precision", "recall", "map", "mrr", "ndcg"),
) -> dict[str, float]:
    """Average metrics across users, for each (metric, k) combination."""
    sums: dict[str, float] = {f"{m}@{k}": 0.0 for m in metrics for k in ks}
    n = 0
    for ranked, relevant in rankings:
        if not relevant:
            continue
        n += 1
        for m in metrics:
            fn = METRIC_FNS[m]
            for k in ks:
                sums[f"{m}@{k}"] += fn(ranked, relevant, k)
    if n == 0:
        return {key: 0.0 for key in sums}
    return {key: sums[key] / n for key in sums}
