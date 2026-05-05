"""Evaluate a fitted model: builds candidate sets, calls model.recommend, aggregates metrics."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import evaluate_user_rankings

log = logging.getLogger(__name__)


@dataclass
class EvalResult:
    metrics: dict[str, float]
    n_users: int


def build_relevant(positives: pd.DataFrame) -> dict[int, set[int]]:
    """Map user_id -> set of relevant item_ids in this split (positives only)."""
    grouped = positives.groupby("user_id")["item_id"].apply(set).to_dict()
    return {int(u): {int(i) for i in items} for u, items in grouped.items()}


def evaluate_model(
    model,
    train: pd.DataFrame,
    target_positives: pd.DataFrame,
    *,
    num_items: int,
    ks: list[int],
    candidate_strategy: str = "full",
    sampled_negatives: int = 100,
    rng: np.random.Generator | None = None,
) -> EvalResult:
    """Evaluate a model.

    target_positives is a *positives-only* dataframe (val or test).

    candidate_strategy:
      - 'full'   : rank all items not in user's training set; relevant = held-out positives.
      - 'sampled': for each held-out positive, score it vs N random negatives.
                   Metrics are computed treating each (positive + negatives) list as
                   a length-(N+1) ranking with the positive as the only relevant item.
    """
    rng = rng or np.random.default_rng(0)
    relevant = build_relevant(target_positives)
    train_seen: dict[int, set[int]] = {
        int(u): {int(i) for i in items}
        for u, items in train.groupby("user_id")["item_id"].apply(set).to_dict().items()
    }

    max_k = max(ks)
    rankings: list[tuple[list[int], set[int]]] = []

    if candidate_strategy == "full":
        for user, rel_items in relevant.items():
            seen = train_seen.get(user, set())
            candidates = np.array(
                [i for i in range(num_items) if i not in seen], dtype=np.int64
            )
            if candidates.size == 0:
                continue
            scores = model.score(user, candidates)
            top_idx = np.argpartition(-scores, kth=min(max_k, len(scores) - 1))[:max_k]
            top = candidates[top_idx]
            top_sorted = top[np.argsort(-scores[top_idx])]
            rankings.append((top_sorted.tolist(), rel_items))

    elif candidate_strategy == "sampled":
        all_items = np.arange(num_items)
        for user, rel_items in relevant.items():
            seen = train_seen.get(user, set()) | rel_items
            for pos in rel_items:
                negs = []
                attempts = 0
                while len(negs) < sampled_negatives and attempts < sampled_negatives * 4:
                    sample = rng.integers(0, num_items, size=sampled_negatives * 2)
                    for s in sample:
                        s = int(s)
                        if s != pos and s not in seen:
                            negs.append(s)
                            if len(negs) == sampled_negatives:
                                break
                    attempts += 1
                cand = np.array([pos] + negs, dtype=np.int64)
                scores = model.score(user, cand)
                order = np.argsort(-scores)
                ranked = cand[order].tolist()
                rankings.append((ranked, {pos}))
        _ = all_items  # silence
    else:
        raise ValueError(f"unknown candidate_strategy: {candidate_strategy}")

    metrics = evaluate_user_rankings(rankings, ks=ks)
    return EvalResult(metrics=metrics, n_users=len(rankings))
