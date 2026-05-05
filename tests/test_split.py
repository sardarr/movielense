import numpy as np
import pandas as pd

from movielense.data.split import temporal_per_user_split


def _make_ratings(n_users: int = 5, n_per_user: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        for t in range(n_per_user):
            rows.append({
                "user_id": u,
                "item_id": int(rng.integers(0, 50)),
                "rating": float(rng.integers(1, 6)),
                "timestamp": t,
            })
    return pd.DataFrame(rows)


def test_split_ratios_approx():
    ratings = _make_ratings(n_users=10, n_per_user=20)
    splits = temporal_per_user_split(
        ratings, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, rating_threshold=4.0
    )
    total = len(splits.train) + len(splits.val) + len(splits.test)
    assert total == len(ratings)
    assert len(splits.train) > len(splits.val)
    assert len(splits.train) > len(splits.test)


def test_split_preserves_temporal_order_per_user():
    ratings = _make_ratings(n_users=3, n_per_user=10)
    splits = temporal_per_user_split(
        ratings, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, rating_threshold=4.0
    )
    for u in range(3):
        train_max = splits.train[splits.train.user_id == u]["timestamp"].max()
        val = splits.val[splits.val.user_id == u]
        test = splits.test[splits.test.user_id == u]
        if len(val) > 0:
            assert val["timestamp"].min() >= train_max
        if len(test) > 0:
            test_min = test["timestamp"].min()
            if len(val) > 0:
                assert test_min >= val["timestamp"].max()
            else:
                assert test_min >= train_max


def test_split_raises_on_bad_ratios():
    ratings = _make_ratings()
    try:
        temporal_per_user_split(ratings, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2, rating_threshold=4.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
