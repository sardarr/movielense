"""Bayesian Personalized Ranking matrix factorization in PyTorch.

Trains user/item embeddings with the BPR loss (Rendle et al., 2009): for each
observed (u, i), sample a random j the user hasn't seen and maximize sigmoid(s_ui - s_uj).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..observers import TrainingObserver
from .base import Recommender

log = logging.getLogger(__name__)


def _resolve_device(pref: str) -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


class BPRRecommender(Recommender):
    name = "bpr"

    def __init__(
        self,
        *,
        num_users: int,
        num_items: int,
        factors: int = 64,
        lr: float = 0.005,
        reg: float = 0.01,
        epochs: int = 20,
        batch_size: int = 1024,
        device: str = "auto",
        seed: int = 0,
    ):
        super().__init__(num_users=num_users, num_items=num_items)
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = _resolve_device(device)
        self.seed = seed

        self._user_emb: nn.Embedding | None = None
        self._item_emb: nn.Embedding | None = None
        self._U_np: np.ndarray | None = None
        self._V_np: np.ndarray | None = None

    def fit(
        self,
        train_positives: pd.DataFrame,
        observer: TrainingObserver | None = None,
    ) -> BPRRecommender:
        obs = self._resolve(observer)
        obs.on_train_start(self.name, self.get_params())
        torch.manual_seed(self.seed)
        users = torch.as_tensor(train_positives["user_id"].to_numpy(), dtype=torch.long, device=self.device)
        pos_items = torch.as_tensor(train_positives["item_id"].to_numpy(), dtype=torch.long, device=self.device)
        n_pairs = users.shape[0]

        # Per-user seen sets, on CPU for fast negative sampling
        seen_by_user: dict[int, set[int]] = (
            train_positives.groupby("user_id")["item_id"].apply(set).to_dict()
        )

        user_emb = nn.Embedding(self.num_users, self.factors, sparse=False).to(self.device)
        item_emb = nn.Embedding(self.num_items, self.factors, sparse=False).to(self.device)
        nn.init.normal_(user_emb.weight, std=0.01)
        nn.init.normal_(item_emb.weight, std=0.01)
        opt = torch.optim.Adam(list(user_emb.parameters()) + list(item_emb.parameters()), lr=self.lr)

        rng = np.random.default_rng(self.seed)
        users_np = users.cpu().numpy()
        pos_np = pos_items.cpu().numpy()

        for ep in range(self.epochs):
            perm = rng.permutation(n_pairs)
            total_loss = 0.0
            for start in range(0, n_pairs, self.batch_size):
                idx = perm[start:start + self.batch_size]
                u = users_np[idx]
                i = pos_np[idx]
                # Sample negatives: rejection-sample until j not seen by u
                j = rng.integers(0, self.num_items, size=u.shape[0])
                for k_pos, (uu, jj) in enumerate(zip(u, j, strict=True)):
                    seen = seen_by_user.get(int(uu), set())
                    while int(jj) in seen:
                        jj = int(rng.integers(0, self.num_items))
                    j[k_pos] = jj

                u_t = torch.as_tensor(u, dtype=torch.long, device=self.device)
                i_t = torch.as_tensor(i, dtype=torch.long, device=self.device)
                j_t = torch.as_tensor(j, dtype=torch.long, device=self.device)

                ue = user_emb(u_t)
                ie = item_emb(i_t)
                je = item_emb(j_t)
                pos_score = (ue * ie).sum(dim=1)
                neg_score = (ue * je).sum(dim=1)
                diff = pos_score - neg_score
                bpr = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
                reg_loss = self.reg * (ue.pow(2).sum() + ie.pow(2).sum() + je.pow(2).sum()) / u.shape[0]
                loss = bpr + reg_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss.detach().cpu()) * u.shape[0]
            avg = total_loss / n_pairs
            obs.on_epoch_end(self.name, ep, {"bpr_loss": avg})
            log.debug("BPR epoch %d/%d  loss=%.4f", ep + 1, self.epochs, avg)

        self._user_emb = user_emb
        self._item_emb = item_emb
        self._U_np = np.nan_to_num(user_emb.weight.detach().cpu().numpy())
        self._V_np = np.nan_to_num(item_emb.weight.detach().cpu().numpy())
        obs.on_train_end(self.name)
        return self

    def score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        assert self._U_np is not None and self._V_np is not None
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return self._V_np[item_ids] @ self._U_np[user_id]

    def get_params(self) -> dict:
        return {
            "factors": self.factors,
            "lr": self.lr,
            "reg": self.reg,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "device": str(self.device),
        }
