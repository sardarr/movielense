"""Train / validation / test splitting.

Default strategy: temporal_per_user. Within each user, sort by timestamp and put
the oldest train_ratio interactions into train, next val_ratio into val, last
test_ratio into test. Users with too few interactions are dropped from val/test
but kept in train (so cold-start users still influence the model).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    rating_threshold: float

    def positives_only(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["rating"] >= self.rating_threshold]

    def summary(self) -> dict[str, int]:
        return {
            "train": int(len(self.train)),
            "val": int(len(self.val)),
            "test": int(len(self.test)),
            "train_users": int(self.train["user_id"].nunique()),
            "val_users": int(self.val["user_id"].nunique()),
            "test_users": int(self.test["user_id"].nunique()),
        }


def temporal_per_user_split(
    ratings: pd.DataFrame,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rating_threshold: float,
) -> Splits:
    total = train_ratio + val_ratio + test_ratio
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {total}")

    train_parts, val_parts, test_parts = [], [], []
    sorted_df = ratings.sort_values(["user_id", "timestamp"])

    for _, group in sorted_df.groupby("user_id", sort=False):
        n = len(group)
        if n < 3:
            train_parts.append(group)
            continue
        n_train = max(1, int(np.floor(n * train_ratio)))
        n_val = max(0, int(np.floor(n * val_ratio)))
        # Ensure at least 1 in test if there's room
        n_test = n - n_train - n_val
        if n_test == 0 and n_val > 0:
            n_val -= 1
            n_test = 1

        idx = group.index.to_numpy()
        train_parts.append(group.loc[idx[:n_train]])
        if n_val > 0:
            val_parts.append(group.loc[idx[n_train:n_train + n_val]])
        if n_test > 0:
            test_parts.append(group.loc[idx[n_train + n_val:]])

    train = pd.concat(train_parts, ignore_index=True) if train_parts else ratings.iloc[0:0]
    val = pd.concat(val_parts, ignore_index=True) if val_parts else ratings.iloc[0:0]
    test = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0]

    log.info(
        "split: train=%d val=%d test=%d", len(train), len(val), len(test)
    )
    return Splits(train=train, val=val, test=test, rating_threshold=rating_threshold)


def leave_one_out_split(ratings: pd.DataFrame, *, rating_threshold: float) -> Splits:
    """NCF / SASRec-style leave-one-out evaluation.

    Per user, sort positives by timestamp ascending. Hold out the single
    most-recent positive as test and the second-most-recent as val. Everything
    else (including sub-threshold ratings) goes to train. Users with fewer
    than 2 positives keep all interactions in train.
    """
    sorted_df = ratings.sort_values(["user_id", "timestamp"])
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for _, group in sorted_df.groupby("user_id", sort=False):
        idx = group.index.to_numpy()
        is_pos = (group["rating"] >= rating_threshold).to_numpy()
        pos_idx = idx[is_pos]
        held: set[int] = set()
        if len(pos_idx) >= 1:
            held.add(int(pos_idx[-1]))
            test_idx.append(int(pos_idx[-1]))
        if len(pos_idx) >= 2:
            held.add(int(pos_idx[-2]))
            val_idx.append(int(pos_idx[-2]))
        for i in idx:
            if int(i) not in held:
                train_idx.append(int(i))

    train = ratings.loc[train_idx].reset_index(drop=True)
    val = ratings.loc[val_idx].reset_index(drop=True)
    test = ratings.loc[test_idx].reset_index(drop=True)
    log.info(
        "leave-one-out split: train=%d val=%d test=%d", len(train), len(val), len(test)
    )
    return Splits(train=train, val=val, test=test, rating_threshold=rating_threshold)


def make_split(ratings: pd.DataFrame, cfg: dict) -> Splits:
    strategy = cfg.get("strategy", "temporal_per_user")
    if strategy == "temporal_per_user":
        return temporal_per_user_split(
            ratings,
            train_ratio=cfg["train_ratio"],
            val_ratio=cfg["val_ratio"],
            test_ratio=cfg["test_ratio"],
            rating_threshold=cfg["rating_threshold"],
        )
    if strategy == "leave_one_out":
        return leave_one_out_split(ratings, rating_threshold=cfg["rating_threshold"])
    raise NotImplementedError(f"split strategy {strategy} not implemented")
