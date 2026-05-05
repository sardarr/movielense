"""Dataset profiling: sparsity, distributions, basic stats."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .load import Dataset

log = logging.getLogger(__name__)


def profile(ds: Dataset) -> dict[str, Any]:
    n_u, n_i = ds.num_users, ds.num_items
    n_int = len(ds.ratings)
    sparsity = 1.0 - n_int / (n_u * n_i)

    user_counts = ds.ratings.groupby("user_id").size().to_numpy()
    item_counts = ds.ratings.groupby("item_id").size().to_numpy()
    rating_hist = ds.ratings["rating"].value_counts().sort_index().to_dict()

    out = {
        "num_users": int(n_u),
        "num_items": int(n_i),
        "num_interactions": int(n_int),
        "sparsity": float(sparsity),
        "rating_hist": {str(k): int(v) for k, v in rating_hist.items()},
        "user_interactions": {
            "min": int(user_counts.min()),
            "p25": int(np.quantile(user_counts, 0.25)),
            "median": int(np.median(user_counts)),
            "p75": int(np.quantile(user_counts, 0.75)),
            "max": int(user_counts.max()),
            "mean": float(user_counts.mean()),
        },
        "item_interactions": {
            "min": int(item_counts.min()),
            "p25": int(np.quantile(item_counts, 0.25)),
            "median": int(np.median(item_counts)),
            "p75": int(np.quantile(item_counts, 0.75)),
            "max": int(item_counts.max()),
            "mean": float(item_counts.mean()),
        },
        "timestamp_range": {
            "min": int(ds.ratings["timestamp"].min()),
            "max": int(ds.ratings["timestamp"].max()),
        },
    }
    return out


def write_profile(profile_dict: dict[str, Any], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        json.dump(profile_dict, f, indent=2)
