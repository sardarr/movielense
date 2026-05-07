"""SASRec — Self-Attentive Sequential Recommendation (Kang & McAuley, 2018).

Trains a small causal transformer over per-user, time-ordered item histories.
The training objective at each position t is BPR-style:
    -log sigmoid(<h_t, V[s_{t+1}]> - <h_t, V[neg]>)
where h_t is the hidden state at position t (which has only seen s_1..s_t due
to the causal mask). At inference we precompute each user's final hidden state
from their training history, so scoring an item is just a dot product — same
cost as ALS / BPR.

Item id 0 is reserved as a padding token; real item ids are stored as i+1
internally and mapped back at the API boundary.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..observers import TrainingObserver
from .base import Recommender
from .bpr import _resolve_device  # reuse cuda/mps/cpu picker

log = logging.getLogger(__name__)


class _SASBlock(nn.Module):
    """One transformer block: pre-norm self-attention + feedforward, both with residual."""

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask, causal_mask):
        h = self.attn_norm(x)
        a, _ = self.attn(
            h, h, h,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + a
        h = self.ffn_norm(x)
        x = x + self.ffn(h)
        return x


class _SASRecModule(nn.Module):
    def __init__(
        self,
        num_items: int,
        d_model: int = 64,
        num_heads: int = 1,
        num_blocks: int = 2,
        max_len: int = 50,
        dropout: float = 0.2,
        content_embeddings: torch.Tensor | None = None,
        content_mode: str = "off",
    ):
        super().__init__()
        if content_mode not in ("off", "fuse", "only"):
            raise ValueError(f"content_mode must be off|fuse|only, got {content_mode}")
        if content_mode != "off" and content_embeddings is None:
            raise ValueError(f"content_mode={content_mode} requires content_embeddings")

        self.num_items = num_items
        self.d_model = d_model
        self.max_len = max_len
        self.content_mode = content_mode

        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_SASBlock(d_model, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.last_norm = nn.LayerNorm(d_model)
        nn.init.xavier_normal_(self.item_emb.weight)
        nn.init.xavier_normal_(self.pos_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

        if content_mode != "off":
            content_dim = content_embeddings.shape[1]
            padded = torch.zeros(num_items + 1, content_dim, dtype=torch.float32)
            padded[1:] = content_embeddings
            # Frozen content matrix: not trained, kept on the same device as the module.
            self.register_buffer("content_buf", padded)
            self.content_proj = nn.Linear(content_dim, d_model)
            nn.init.xavier_normal_(self.content_proj.weight)
            nn.init.zeros_(self.content_proj.bias)
        else:
            self.content_proj = None

    def _item_repr(self, ids: torch.Tensor) -> torch.Tensor:
        if self.content_mode == "off":
            return self.item_emb(ids)
        proj = self.content_proj(self.content_buf[ids])
        # Pad token (id 0) must stay zero so attention masking works correctly.
        proj = proj.masked_fill((ids == 0).unsqueeze(-1), 0.0)
        if self.content_mode == "only":
            return proj
        return self.item_emb(ids) + proj

    def all_item_repr(self) -> torch.Tensor:
        """Full (num_items + 1, d_model) item-representation matrix used at scoring."""
        ids = torch.arange(self.num_items + 1, device=self.item_emb.weight.device)
        return self._item_repr(ids)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self._item_repr(seq) + self.pos_emb(positions)
        x = self.dropout(x)
        pad_mask = seq == 0
        causal = torch.triu(
            torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1
        )
        for block in self.blocks:
            x = block(x, key_padding_mask=pad_mask, causal_mask=causal)
        return self.last_norm(x)


class SASRecRecommender(Recommender):
    name = "sasrec"

    def __init__(
        self,
        *,
        num_users: int,
        num_items: int,
        d_model: int = 64,
        num_heads: int = 1,
        num_blocks: int = 2,
        max_len: int = 50,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        epochs: int = 50,
        batch_size: int = 128,
        device: str = "auto",
        seed: int = 0,
        content_embeddings: np.ndarray | None = None,
        content_mode: str = "off",
        content_encoder: str | None = None,
    ):
        super().__init__(num_users=num_users, num_items=num_items)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.max_len = max_len
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = _resolve_device(device)
        self.content_mode = content_mode
        self.content_encoder = content_encoder
        self._content_t = (
            torch.as_tensor(content_embeddings, dtype=torch.float32)
            if content_embeddings is not None else None
        )
        self.module: _SASRecModule | None = None
        self._item_emb_np: np.ndarray | None = None
        self._user_h_np: np.ndarray | None = None
        self._seen_by_user: dict[int, set[int]] = {}

    # ---- training ----
    def _build_sequences(self, train_positives: pd.DataFrame) -> dict[int, list[int]]:
        """Return user_id -> [item_id, ...] sorted by timestamp ascending. Item ids are 0..N-1."""
        if "timestamp" in train_positives.columns:
            df = train_positives.sort_values(["user_id", "timestamp"])
        else:
            df = train_positives
        return df.groupby("user_id")["item_id"].apply(list).to_dict()

    def _make_training_tensors(self, sequences: dict[int, list[int]]):
        """Build (input_seq, target_seq) tensors per user with left-padding."""
        L = self.max_len
        users = []
        seq_inputs: list[list[int]] = []
        seq_targets: list[list[int]] = []
        for u, items in sequences.items():
            if len(items) < 2:
                continue
            shifted = [i + 1 for i in items[-(L + 1):]]  # last L+1 items, +1 for pad-aware indexing
            inp = shifted[:-1]
            tgt = shifted[1:]
            pad = L - len(inp)
            inp = [0] * pad + inp
            tgt = [0] * pad + tgt
            users.append(int(u))
            seq_inputs.append(inp)
            seq_targets.append(tgt)
        return (
            torch.as_tensor(np.array(users), dtype=torch.long),
            torch.as_tensor(np.array(seq_inputs), dtype=torch.long),
            torch.as_tensor(np.array(seq_targets), dtype=torch.long),
        )

    def fit(
        self,
        train_positives: pd.DataFrame,
        observer: TrainingObserver | None = None,
    ) -> SASRecRecommender:
        obs = self._resolve(observer)
        obs.on_train_start(self.name, self.get_params())

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        sequences = self._build_sequences(train_positives)
        self._seen_by_user = {int(u): set(items) for u, items in sequences.items()}

        users_t, inputs_t, targets_t = self._make_training_tensors(sequences)
        if users_t.numel() == 0:
            raise ValueError("no training sequences found (all users have <2 interactions)")

        users_t = users_t.to(self.device)
        inputs_t = inputs_t.to(self.device)
        targets_t = targets_t.to(self.device)

        self.module = _SASRecModule(
            num_items=self.num_items,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            max_len=self.max_len,
            dropout=self.dropout,
            content_embeddings=self._content_t,
            content_mode=self.content_mode,
        ).to(self.device)
        opt = torch.optim.Adam(self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        n_seqs = inputs_t.shape[0]
        for ep in range(self.epochs):
            self.module.train()
            perm = torch.as_tensor(rng.permutation(n_seqs), dtype=torch.long, device=self.device)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, n_seqs, self.batch_size):
                idx = perm[start:start + self.batch_size]
                inp = inputs_t[idx]
                tgt = targets_t[idx]
                # Sample one negative per position; reject items already in user's history.
                neg = self._sample_negatives(users_t[idx].cpu().numpy(), inp.shape, rng)
                neg = torch.as_tensor(neg, dtype=torch.long, device=self.device)

                h = self.module(inp)  # (B, L, d)
                pos_emb = self.module._item_repr(tgt)  # (B, L, d)
                neg_emb = self.module._item_repr(neg)
                pos_score = (h * pos_emb).sum(-1)
                neg_score = (h * neg_emb).sum(-1)
                # mask: only positions where target is non-pad
                mask = (tgt != 0).float()
                bpr = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10) * mask
                loss = bpr.sum() / mask.sum().clamp(min=1.0)

                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss.detach().cpu())
                n_batches += 1
            avg = total_loss / max(n_batches, 1)
            obs.on_epoch_end(self.name, ep, {"sasrec_loss": avg})
            log.debug("SASRec epoch %d/%d  loss=%.4f", ep + 1, self.epochs, avg)

        self._cache_user_encodings(sequences)
        obs.on_train_end(self.name)
        return self

    def _sample_negatives(self, users_np: np.ndarray, shape, rng) -> np.ndarray:
        B, L = shape
        neg = rng.integers(1, self.num_items + 1, size=(B, L))
        for i, u in enumerate(users_np):
            seen = self._seen_by_user.get(int(u), set())
            for j in range(L):
                while int(neg[i, j]) - 1 in seen:
                    neg[i, j] = int(rng.integers(1, self.num_items + 1))
        return neg

    def _cache_user_encodings(self, sequences: dict[int, list[int]]) -> None:
        assert self.module is not None
        self.module.eval()
        L = self.max_len
        H = np.zeros((self.num_users, self.d_model), dtype=np.float32)

        # Batched forward over all users.
        all_inputs = []
        user_order: list[int] = []
        for u, items in sequences.items():
            shifted = [i + 1 for i in items[-L:]]
            pad = L - len(shifted)
            seq = [0] * pad + shifted
            all_inputs.append(seq)
            user_order.append(int(u))
        if not all_inputs:
            self._user_h_np = H
            with torch.no_grad():
                self._item_emb_np = self.module.all_item_repr().detach().cpu().numpy()
            return

        x = torch.as_tensor(np.array(all_inputs), dtype=torch.long, device=self.device)
        with torch.no_grad():
            for start in range(0, x.shape[0], 256):
                batch = x[start:start + 256]
                h = self.module(batch)  # (B, L, d)
                # Sequences are left-padded, so the last real item is always at
                # position L-1 (regardless of how many real items the user has).
                # Earlier this used (batch != 0).sum() - 1 which incorrectly
                # indexed into the padding region for short users.
                h_last = h[:, -1, :]
                for k, u in enumerate(user_order[start:start + batch.shape[0]]):
                    H[u] = h_last[k].cpu().numpy()

        self._user_h_np = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            full_item_repr = self.module.all_item_repr().detach().cpu().numpy()
        self._item_emb_np = np.nan_to_num(
            full_item_repr, nan=0.0, posinf=0.0, neginf=0.0,
        )

    # ---- inference ----
    def score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        if self._user_h_np is None or self._item_emb_np is None:
            raise RuntimeError("model is not fitted")
        # item_ids are 0..N-1; module's embedding has +1 offset for the pad token.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return self._item_emb_np[item_ids + 1] @ self._user_h_np[user_id]

    def get_params(self) -> dict:
        return {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_blocks": self.num_blocks,
            "max_len": self.max_len,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "device": str(self.device),
            "content_mode": self.content_mode,
            "content_encoder": self.content_encoder,
        }
