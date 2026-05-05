"""Implicit-feedback ALS (Hu, Koren, Volinsky 2008), pure numpy/scipy.

We solve for user/item factors by alternating ridge regression with the implicit
confidence weighting C_ui = 1 + alpha * r_ui.

We use the Cholesky trick for efficient per-user/per-item solves.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ..observers import TrainingObserver
from .base import Recommender

log = logging.getLogger(__name__)


class ALSRecommender(Recommender):
    name = "als"

    def __init__(
        self,
        *,
        num_users: int,
        num_items: int,
        factors: int = 64,
        regularization: float = 0.05,
        iterations: int = 15,
        alpha: float = 40.0,
        seed: int = 0,
    ):
        super().__init__(num_users=num_users, num_items=num_items)
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.seed = seed
        self.U: np.ndarray | None = None
        self.V: np.ndarray | None = None

    def fit(
        self,
        train_positives: pd.DataFrame,
        observer: TrainingObserver | None = None,
    ) -> ALSRecommender:
        obs = self._resolve(observer)
        obs.on_train_start(self.name, self.get_params())

        rng = np.random.default_rng(self.seed)
        rows = train_positives["user_id"].to_numpy()
        cols = train_positives["item_id"].to_numpy()
        data = np.full(len(rows), self.alpha, dtype=np.float32)
        Cui = csr_matrix((data, (rows, cols)), shape=(self.num_users, self.num_items))
        Ciu = Cui.T.tocsr()

        f = self.factors
        self.U = rng.normal(0, 0.01, size=(self.num_users, f)).astype(np.float32)
        self.V = rng.normal(0, 0.01, size=(self.num_items, f)).astype(np.float32)
        reg_eye = self.regularization * np.eye(f, dtype=np.float32)

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for it in range(self.iterations):
                self.U = _als_step(Cui, self.V, reg_eye)
                self.V = _als_step(Ciu, self.U, reg_eye)
                # Defense in depth: bad hyperparams during tuning can blow up the
                # solve; replace any NaN/inf so downstream code sees finite scores.
                self.U = np.nan_to_num(self.U, nan=0.0, posinf=0.0, neginf=0.0)
                self.V = np.nan_to_num(self.V, nan=0.0, posinf=0.0, neginf=0.0)
                recon = _train_reconstruction_loss(rows, cols, self.U, self.V)
                obs.on_epoch_end(self.name, it, {"recon_mse": recon})
                log.debug("ALS iter %d/%d  recon_mse=%.4f", it + 1, self.iterations, recon)

        obs.on_train_end(self.name)
        return self

    def score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        assert self.U is not None and self.V is not None
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return self.V[item_ids] @ self.U[user_id]

    def get_params(self) -> dict:
        return {
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "alpha": self.alpha,
        }


def _train_reconstruction_loss(rows: np.ndarray, cols: np.ndarray, U: np.ndarray, V: np.ndarray) -> float:
    """Mean squared error between predicted score and 1 on observed positives.

    Cheap proxy for training progress (lower is better). Computed on a sample
    when there are many interactions.
    """
    n = rows.shape[0]
    if n > 50_000:
        idx = np.random.default_rng(0).choice(n, size=50_000, replace=False)
        rows = rows[idx]
        cols = cols[idx]
    pred = (U[rows] * V[cols]).sum(axis=1)
    return float(((pred - 1.0) ** 2).mean())


def _als_step(C: csr_matrix, Y: np.ndarray, reg_eye: np.ndarray) -> np.ndarray:
    """One side of ALS: solve for X given Y using implicit-feedback confidence.

    For each row u of C (a user, say):
       X_u = (Y^T Y + Y^T (C_u - I) Y + reg) ^ -1 . (Y^T C_u p_u)

    where p_u = 1 for nonzero entries in C_u (positives), 0 elsewhere.
    Since C_ui - 1 stored as data, Y^T diag(C_u - 1) Y picks up only the nonzero items.
    """
    # Promote to float64 for the linear-system solve — float32 overflows with
    # large alpha * factors and produces NaN factors that poison the next step.
    Y64 = Y.astype(np.float64, copy=False)
    reg64 = reg_eye.astype(np.float64, copy=False)
    n_rows = C.shape[0]
    f = Y64.shape[1]
    YtY = Y64.T @ Y64
    X = np.zeros((n_rows, f), dtype=np.float32)
    indptr, indices, data = C.indptr, C.indices, C.data
    for u in range(n_rows):
        start, end = indptr[u], indptr[u + 1]
        if start == end:
            continue
        idx = indices[start:end]
        cu_minus_one = data[start:end].astype(np.float64)
        Yu = Y64[idx]
        A = YtY + (Yu * cu_minus_one[:, None]).T @ Yu + reg64
        b = Yu.T @ (1.0 + cu_minus_one)
        X[u] = np.linalg.solve(A, b).astype(np.float32)
    return X
