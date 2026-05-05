"""Base recommender interface.

Every recommender must implement:
  fit(positives_df, observer=None) -> self
  score(user_id, item_ids)         -> 1D float array of scores aligned with item_ids

Observers receive on_train_start / on_epoch_end / on_train_end events for
iterative models (BPR, ALS). Non-iterative models (random, popularity, kNN) just
emit start/end and treat training as one "epoch".
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..observers import NullObserver, TrainingObserver


class Recommender(ABC):
    name: str = "base"

    def __init__(self, *, num_users: int, num_items: int):
        self.num_users = num_users
        self.num_items = num_items

    @abstractmethod
    def fit(
        self,
        train_positives: pd.DataFrame,
        observer: TrainingObserver | None = None,
    ) -> Recommender:
        ...

    @abstractmethod
    def score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        ...

    def get_params(self) -> dict:
        return {}

    @staticmethod
    def _resolve(observer: TrainingObserver | None) -> TrainingObserver:
        return observer if observer is not None else NullObserver()
