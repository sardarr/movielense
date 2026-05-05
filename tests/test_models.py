"""Smoke tests for the model fit/score interface on tiny data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from movielense.models import build


def _toy_train(num_users: int = 8, num_items: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(num_users):
        items = rng.choice(num_items, size=4, replace=False)
        for ts, i in enumerate(items):
            rows.append({"user_id": u, "item_id": int(i), "rating": 5.0, "timestamp": ts})
    return pd.DataFrame(rows)


def _check_model(name: str, params: dict | None = None):
    n_u, n_i = 8, 12
    train = _toy_train(n_u, n_i)
    model = build(name, params=params or {}, num_users=n_u, num_items=n_i, seed=0)
    model.fit(train)
    items = np.arange(n_i)
    scores = model.score(0, items)
    assert scores.shape == (n_i,)
    assert np.isfinite(scores).all()


def test_random():
    _check_model("random")


def test_popularity():
    _check_model("popularity")


def test_itemknn():
    _check_model("itemknn", {"k": 5, "similarity": "cosine"})


def test_als():
    _check_model("als", {"factors": 8, "regularization": 0.05, "iterations": 3, "alpha": 10.0})


def test_bpr_runs():
    _check_model("bpr", {"factors": 8, "lr": 0.01, "reg": 0.01, "epochs": 2, "batch_size": 8, "device": "cpu"})


def test_sasrec_runs():
    _check_model(
        "sasrec",
        {
            "d_model": 16, "num_heads": 1, "num_blocks": 1, "max_len": 6,
            "dropout": 0.0, "lr": 0.01, "epochs": 2, "batch_size": 4, "device": "cpu",
        },
    )
