"""Baselines: random and popularity."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..observers import TrainingObserver
from .base import Recommender


class RandomRecommender(Recommender):
    name = "random"

    def __init__(self, *, num_users: int, num_items: int, seed: int = 0):
        super().__init__(num_users=num_users, num_items=num_items)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def fit(
        self,
        train_positives: pd.DataFrame,
        observer: TrainingObserver | None = None,
    ) -> RandomRecommender:
        obs = self._resolve(observer)
        obs.on_train_start(self.name, self.get_params())
        obs.on_train_end(self.name)
        return self

    def score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        # Deterministic per (user, item) so eval is stable across calls.
        rng = np.random.default_rng(self.seed * 1_000_003 + user_id)
        return rng.random(size=item_ids.shape[0])

    def get_params(self) -> dict:
        return {"seed": self.seed}


class PopularityRecommender(Recommender):
    name = "popularity"

    def __init__(self, *, num_users: int, num_items: int):
        super().__init__(num_users=num_users, num_items=num_items)
        self._popularity = np.zeros(num_items, dtype=np.float64)

    def fit(
        self,
        train_positives: pd.DataFrame,
        observer: TrainingObserver | None = None,
    ) -> PopularityRecommender:
        obs = self._resolve(observer)
        obs.on_train_start(self.name, self.get_params())
        counts = train_positives["item_id"].value_counts()
        self._popularity = np.zeros(self.num_items, dtype=np.float64)
        self._popularity[counts.index.to_numpy()] = counts.to_numpy()
        obs.on_train_end(self.name)
        return self

    def score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        return self._popularity[item_ids]
