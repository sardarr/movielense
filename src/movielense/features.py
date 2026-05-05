"""Feature engineering for the LightGBM ranker.

Produces a `(user, item) -> feature row` matrix used both at train time (with
positives + sampled negatives) and at score time (full candidate ranking).

All statistics are computed from the *training* split only — leakage-free.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data.load import GENRE_NAMES, Dataset

log = logging.getLogger(__name__)


@dataclass
class FeatureStore:
    """Pre-computed user and item feature vectors. Combined per (user, item) pair on demand."""
    feature_names: list[str]
    user_features: np.ndarray   # (num_users, k_user)
    item_features: np.ndarray   # (num_items, k_item)
    user_genre_pref: np.ndarray  # (num_users, num_genres) train-derived genre affinity
    genre_matrix: np.ndarray    # (num_items, num_genres)
    n_user_feats: int
    n_item_feats: int

    def build_pairs(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Stack [user_feats | item_feats | interaction_feats] for each (u, i)."""
        u = self.user_features[user_ids]
        v = self.item_features[item_ids]
        # Interaction: dot product of user genre preference with item genre vector.
        u_genre = self.user_genre_pref[user_ids]
        i_genre = self.genre_matrix[item_ids]
        affinity = (u_genre * i_genre).sum(axis=1, keepdims=True)
        return np.concatenate([u, v, affinity], axis=1)


def build_feature_store(
    ds: Dataset,
    train_positives: pd.DataFrame,
) -> FeatureStore:
    """Build all per-user and per-item features from the training split.

    Negative-leakage rules:
      - all popularity / rating averages come from train_positives only
      - val/test are never read here
    """
    n_u, n_i = ds.num_users, ds.num_items

    # Item content features: genres + release year + popularity + mean rating.
    genre_cols = [f"genre_{g}" for g in GENRE_NAMES]
    items = ds.items.set_index("item_id").reindex(np.arange(n_i)).fillna(0)
    genre_mat = items[genre_cols].to_numpy(dtype=np.float32)
    release_year = items["release_year"].to_numpy(dtype=np.float32)
    # Center year for stability.
    release_year = release_year - 1990.0

    item_pop = (
        train_positives.groupby("item_id").size().reindex(np.arange(n_i)).fillna(0).to_numpy(dtype=np.float32)
    )
    item_pop_log = np.log1p(item_pop)
    item_mean_rating = (
        train_positives.groupby("item_id")["rating"].mean().reindex(np.arange(n_i)).fillna(0).to_numpy(dtype=np.float32)
    )

    item_feats = np.concatenate(
        [
            genre_mat,
            release_year[:, None],
            item_pop_log[:, None],
            item_mean_rating[:, None],
        ],
        axis=1,
    )
    item_feature_names = (
        [f"item_{g.lower()}" for g in GENRE_NAMES]
        + ["item_release_year_centered", "item_popularity_log", "item_mean_rating"]
    )

    # User side-info features (age, gender, occupation as one-hot).
    if not ds.users.empty:
        users = ds.users.set_index("user_id").reindex(np.arange(n_u))
        age = users["age"].fillna(users["age"].median()).to_numpy(dtype=np.float32)
        gender_F = (users["gender"].fillna("M").to_numpy() == "F").astype(np.float32)
        occ_dummies = pd.get_dummies(users["occupation"].fillna("other"), prefix="occ").astype(np.float32)
        occ_mat = occ_dummies.to_numpy()
        occ_names = list(occ_dummies.columns)
        side_feats = np.concatenate([age[:, None], gender_F[:, None], occ_mat], axis=1)
        side_names = ["user_age", "user_is_female", *[f"user_{c}" for c in occ_names]]
    else:
        side_feats = np.zeros((n_u, 0), dtype=np.float32)
        side_names = []

    # User behavior features from training split.
    user_count = (
        train_positives.groupby("user_id").size().reindex(np.arange(n_u)).fillna(0).to_numpy(dtype=np.float32)
    )
    user_mean_rating = (
        train_positives.groupby("user_id")["rating"].mean().reindex(np.arange(n_u)).fillna(0).to_numpy(dtype=np.float32)
    )
    user_count_log = np.log1p(user_count)

    # User genre preference: weighted sum of the genre vectors of items they liked.
    if len(train_positives) > 0:
        # Stack genre vectors aligned with each interaction, average per user.
        i_idx = train_positives["item_id"].to_numpy()
        u_idx = train_positives["user_id"].to_numpy()
        per_int_genres = genre_mat[i_idx]
        sum_per_user = np.zeros((n_u, genre_mat.shape[1]), dtype=np.float32)
        np.add.at(sum_per_user, u_idx, per_int_genres)
        with np.errstate(divide="ignore", invalid="ignore"):
            user_genre_pref = sum_per_user / np.where(user_count[:, None] > 0, user_count[:, None], 1.0)
    else:
        user_genre_pref = np.zeros((n_u, genre_mat.shape[1]), dtype=np.float32)

    user_feats = np.concatenate(
        [
            side_feats,
            user_count_log[:, None],
            user_mean_rating[:, None],
            user_genre_pref,
        ],
        axis=1,
    )
    user_feature_names = (
        side_names
        + ["user_interaction_count_log", "user_mean_rating_given"]
        + [f"user_pref_{g.lower()}" for g in GENRE_NAMES]
    )

    # Final names: user_features | item_features | interaction_features
    feature_names = user_feature_names + item_feature_names + ["user_item_genre_affinity"]

    log.info(
        "feature store built: %d user_feats + %d item_feats + 1 interaction = %d total",
        user_feats.shape[1], item_feats.shape[1], len(feature_names),
    )

    return FeatureStore(
        feature_names=feature_names,
        user_features=user_feats.astype(np.float32),
        item_features=item_feats.astype(np.float32),
        user_genre_pref=user_genre_pref.astype(np.float32),
        genre_matrix=genre_mat.astype(np.float32),
        n_user_feats=user_feats.shape[1],
        n_item_feats=item_feats.shape[1],
    )


def sample_negatives(
    train_positives: pd.DataFrame,
    *,
    num_items: int,
    n_neg_per_pos: int,
    seed: int,
) -> pd.DataFrame:
    """Return a frame with columns [user_id, item_id, label] containing the originals
    plus n_neg_per_pos sampled negatives per (user, positive) pair.
    """
    rng = np.random.default_rng(seed)
    seen_by_user: dict[int, set[int]] = (
        train_positives.groupby("user_id")["item_id"].apply(set).to_dict()
    )

    pos = train_positives[["user_id", "item_id"]].copy()
    pos["label"] = 1

    neg_users = []
    neg_items = []
    for u, items in seen_by_user.items():
        n_pos = len(items)
        n_neg = n_pos * n_neg_per_pos
        sampled = rng.integers(0, num_items, size=n_neg * 2)
        kept: list[int] = []
        for s in sampled:
            s = int(s)
            if s not in items:
                kept.append(s)
                if len(kept) >= n_neg:
                    break
        neg_users.extend([u] * len(kept))
        neg_items.extend(kept)

    neg = pd.DataFrame({"user_id": neg_users, "item_id": neg_items, "label": 0})
    out = pd.concat([pos, neg], ignore_index=True)
    out = out.sort_values("user_id").reset_index(drop=True)
    return out
