"""Latency benchmarks for batch production planning.

For each fitted model we measure:
  - fit time (seconds)
  - per-user score latency (median, p95, ms)
  - throughput in users/sec for the chosen candidate set size
  - serialized model size (MB) — proxy for memory and disk footprint

This is designed for *batch* serving: e.g., generating top-K recs for every
user nightly. We score N candidates per user (matches realistic serving where
candidates come from a retrieval stage upstream).
"""
from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    model_name: str
    fit_seconds: float
    score_latency_p50_ms: float
    score_latency_p95_ms: float
    throughput_users_per_sec: float
    serialized_mb: float
    candidates_per_user: int
    n_users_measured: int

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "fit_seconds": round(self.fit_seconds, 3),
            "score_latency_p50_ms": round(self.score_latency_p50_ms, 3),
            "score_latency_p95_ms": round(self.score_latency_p95_ms, 3),
            "throughput_users_per_sec": round(self.throughput_users_per_sec, 1),
            "serialized_mb": round(self.serialized_mb, 2),
            "candidates_per_user": self.candidates_per_user,
            "n_users_measured": self.n_users_measured,
        }


def measure_score_latency(
    model,
    *,
    user_ids: np.ndarray,
    num_items: int,
    candidates_per_user: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Returns (median_ms, p95_ms, throughput_users_per_sec)."""
    timings_ns: list[int] = []
    # Pre-sample candidate sets so we measure scoring, not RNG.
    pools = [
        rng.choice(num_items, size=min(candidates_per_user, num_items), replace=False)
        for _ in user_ids
    ]
    # Warmup: a single non-timed call to amortize lazy init / first-call costs.
    if len(user_ids) > 0:
        model.score(int(user_ids[0]), pools[0])

    t_total_start = time.perf_counter_ns()
    for u, pool in zip(user_ids, pools, strict=True):
        t0 = time.perf_counter_ns()
        _ = model.score(int(u), pool)
        timings_ns.append(time.perf_counter_ns() - t0)
    t_total = (time.perf_counter_ns() - t_total_start) / 1e9

    arr = np.asarray(timings_ns) / 1e6  # ms
    p50 = float(np.median(arr))
    p95 = float(np.quantile(arr, 0.95))
    throughput = len(user_ids) / max(t_total, 1e-9)
    return p50, p95, throughput


def serialized_size_mb(model) -> float:
    try:
        blob = pickle.dumps(model)
        return len(blob) / (1024 * 1024)
    except Exception as e:
        log.warning("could not pickle model for size measurement: %s", e)
        return float("nan")


def benchmark(
    *,
    model_name: str,
    fit_seconds: float,
    model,
    num_users: int,
    num_items: int,
    candidates_per_user: int,
    n_users_to_score: int,
    seed: int,
) -> BenchmarkResult:
    rng = np.random.default_rng(seed)
    sample_users = rng.choice(num_users, size=min(n_users_to_score, num_users), replace=False)
    p50, p95, tps = measure_score_latency(
        model,
        user_ids=sample_users,
        num_items=num_items,
        candidates_per_user=candidates_per_user,
        rng=rng,
    )
    size_mb = serialized_size_mb(model)
    return BenchmarkResult(
        model_name=model_name,
        fit_seconds=fit_seconds,
        score_latency_p50_ms=p50,
        score_latency_p95_ms=p95,
        throughput_users_per_sec=tps,
        serialized_mb=size_mb,
        candidates_per_user=candidates_per_user,
        n_users_measured=len(sample_users),
    )
