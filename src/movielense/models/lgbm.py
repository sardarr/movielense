"""LightGBM LambdaRank ranker.

Unlike the latent-factor models (ALS, BPR), this model consumes engineered
content + behavioral features (genres, year, demographics, popularity, user
genre affinity) and produces both a ranking score and a feature-importance
vector for explainability.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from ..features import FeatureStore, sample_negatives
from ..observers import TrainingObserver
from .base import Recommender

log = logging.getLogger(__name__)


class LGBMRankerRecommender(Recommender):
    name = "lgbm"

    def __init__(
        self,
        *,
        num_users: int,
        num_items: int,
        feature_store: FeatureStore,
        n_neg_per_pos: int = 4,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_estimators: int = 200,
        min_child_samples: int = 20,
        reg_lambda: float = 0.0,
        seed: int = 0,
    ):
        super().__init__(num_users=num_users, num_items=num_items)
        self.feature_store = feature_store
        self.n_neg_per_pos = n_neg_per_pos
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self.seed = seed
        self.booster_ = None
        self.feature_importance_gain_: np.ndarray | None = None
        self.feature_importance_split_: np.ndarray | None = None

    def fit(
        self,
        train_positives: pd.DataFrame,
        observer: TrainingObserver | None = None,
    ) -> LGBMRankerRecommender:
        import lightgbm as lgb

        obs = self._resolve(observer)
        obs.on_train_start(self.name, self.get_params())

        labeled = sample_negatives(
            train_positives,
            num_items=self.num_items,
            n_neg_per_pos=self.n_neg_per_pos,
            seed=self.seed,
        )
        u_ids = labeled["user_id"].to_numpy()
        i_ids = labeled["item_id"].to_numpy()
        X = self.feature_store.build_pairs(u_ids, i_ids)
        y = labeled["label"].to_numpy()
        # group: rows-per-user (already sorted by sample_negatives)
        groups = labeled.groupby("user_id", sort=False).size().to_numpy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ranker = lgb.LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                min_child_samples=self.min_child_samples,
                reg_lambda=self.reg_lambda,
                random_state=self.seed,
                # Single-threaded: avoids an OpenMP conflict with torch on macOS
                # arm64 that segfaults inside Booster.predict.
                n_jobs=1,
                verbose=-1,
            )
            ranker.fit(X, y, group=groups)
        self.booster_ = ranker

        try:
            self.feature_importance_gain_ = ranker.booster_.feature_importance(importance_type="gain")
            self.feature_importance_split_ = ranker.booster_.feature_importance(importance_type="split")
        except Exception as e:
            log.warning("feature importance unavailable: %s", e)

        # We don't run iterative epochs; emit a single end-of-train event.
        obs.on_epoch_end(self.name, 0, {"best_iteration": float(getattr(ranker, "best_iteration_", 0) or self.n_estimators)})
        obs.on_train_end(self.name)
        return self

    def score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        if self.booster_ is None:
            raise RuntimeError("model is not fitted")
        users = np.full_like(item_ids, fill_value=int(user_id), dtype=np.int64)
        X = self.feature_store.build_pairs(users, item_ids)
        with warnings.catch_warnings():
            # sklearn's "X does not have valid feature names" warning is triggered
            # because we pass a numpy array; benign and not actionable here.
            warnings.simplefilter("ignore")
            return self.booster_.predict(X)

    def get_params(self) -> dict:
        return {
            "n_neg_per_pos": self.n_neg_per_pos,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "min_child_samples": self.min_child_samples,
            "reg_lambda": self.reg_lambda,
        }

    def feature_importance(self) -> dict[str, np.ndarray] | None:
        if self.feature_importance_gain_ is None:
            return None
        return {
            "names": np.array(self.feature_store.feature_names),
            "gain": self.feature_importance_gain_,
            "split": self.feature_importance_split_,
        }
