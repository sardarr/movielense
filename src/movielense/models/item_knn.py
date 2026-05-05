"""Item-item kNN with cosine similarity over user-item interactions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from ..observers import TrainingObserver
from .base import Recommender


class ItemKNNRecommender(Recommender):
    name = "itemknn"

    def __init__(
        self,
        *,
        num_users: int,
        num_items: int,
        k: int = 50,
        similarity: str = "cosine",
        shrinkage: float = 0.0,
    ):
        super().__init__(num_users=num_users, num_items=num_items)
        self.k = k
        self.similarity = similarity
        self.shrinkage = shrinkage
        self._user_item: csr_matrix | None = None
        self._sim: csr_matrix | None = None  # truncated top-k item-item

    def fit(
        self,
        train_positives: pd.DataFrame,
        observer: TrainingObserver | None = None,
    ) -> ItemKNNRecommender:
        obs = self._resolve(observer)
        obs.on_train_start(self.name, self.get_params())
        rows = train_positives["user_id"].to_numpy()
        cols = train_positives["item_id"].to_numpy()
        data = np.ones(len(rows), dtype=np.float32)
        ui = csr_matrix((data, (rows, cols)), shape=(self.num_users, self.num_items))
        self._user_item = ui

        item_user = ui.T.tocsr()
        if self.similarity == "cosine":
            item_user_norm = normalize(item_user, norm="l2", axis=1, copy=True)
            sim = item_user_norm @ item_user_norm.T  # (num_items, num_items)
        else:
            raise ValueError(f"unknown similarity: {self.similarity}")

        sim = sim.tolil()
        sim.setdiag(0.0)  # zero self-similarity
        sim = sim.tocsr()

        # Keep only top-k per row
        self._sim = _truncate_top_k(sim, self.k)
        obs.on_train_end(self.name)
        return self

    def score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        assert self._user_item is not None and self._sim is not None
        # user history vector (1, num_items)
        user_row = self._user_item.getrow(user_id)
        # scores for ALL items: user_row @ sim  -> (1, num_items)
        all_scores = user_row @ self._sim
        all_scores = np.asarray(all_scores.todense()).ravel()
        return all_scores[item_ids]

    def get_params(self) -> dict:
        return {"k": self.k, "similarity": self.similarity, "shrinkage": self.shrinkage}


def _truncate_top_k(sim: csr_matrix, k: int) -> csr_matrix:
    """For each row of sim, keep only the top-k largest values, zero the rest."""
    sim = sim.tocsr()
    new_data = []
    new_indices = []
    new_indptr = [0]
    for r in range(sim.shape[0]):
        start, end = sim.indptr[r], sim.indptr[r + 1]
        row_data = sim.data[start:end]
        row_idx = sim.indices[start:end]
        if row_data.size > k:
            top = np.argpartition(-row_data, kth=k - 1)[:k]
            row_data = row_data[top]
            row_idx = row_idx[top]
        new_data.append(row_data)
        new_indices.append(row_idx)
        new_indptr.append(new_indptr[-1] + row_data.size)

    return csr_matrix(
        (np.concatenate(new_data) if new_data else np.array([], dtype=np.float32),
         np.concatenate(new_indices) if new_indices else np.array([], dtype=np.int32),
         np.array(new_indptr, dtype=np.int32)),
        shape=sim.shape,
    )
