"""Per-item content embeddings for content-aware models.

Builds a short text per item from title + genres + year, then encodes each text
with a frozen sentence-transformer. The resulting (num_items, dim) matrix is
cached on disk keyed by content hash so subsequent runs reuse the same vectors.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .load import GENRE_COLS, GENRE_NAMES

log = logging.getLogger(__name__)

ENCODER_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


def _item_text(row: pd.Series) -> str:
    title = str(row.get("title", "")).strip()
    genres = [
        g for g, col in zip(GENRE_NAMES, GENRE_COLS, strict=False)
        if int(row.get(col, 0)) == 1
    ]
    genre_str = ", ".join(genres) if genres else "unknown"
    parts = [title, f"Genres: {genre_str}"]
    year = row.get("release_year")
    if year is not None and not pd.isna(year):
        parts.append(f"Year: {int(year)}")
    return ". ".join(parts)


def _cache_key(texts: list[str], encoder: str) -> str:
    h = hashlib.sha256()
    h.update(encoder.encode())
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _encode(texts: list[str], encoder: str) -> np.ndarray:
    if encoder not in ENCODER_MODELS:
        raise ValueError(f"unknown content encoder: {encoder}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(ENCODER_MODELS[encoder])
    embs = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=64,
        normalize_embeddings=False,
    )
    return embs.astype(np.float32)


def build_content_embeddings(
    items: pd.DataFrame,
    *,
    num_items: int,
    encoder: str = "minilm",
    cache_dir: Path = Path("data/processed"),
) -> np.ndarray:
    """Return (num_items, dim) content embeddings indexed by densified item_id.

    Items absent from the items frame get a zero row.
    """
    items_sorted = items.sort_values("item_id").drop_duplicates("item_id")
    texts = [_item_text(r) for _, r in items_sorted.iterrows()]

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"content_emb_{encoder}_{_cache_key(texts, encoder)}.npy"

    if cache_path.exists():
        log.info("loading cached content embeddings: %s", cache_path)
        embs_subset = np.load(cache_path)
    else:
        log.info("encoding %d items with %s ...", len(texts), encoder)
        embs_subset = _encode(texts, encoder)
        np.save(cache_path, embs_subset)
        log.info("cached content embeddings: %s (%dx%d)",
                 cache_path, embs_subset.shape[0], embs_subset.shape[1])

    dim = embs_subset.shape[1]
    out = np.zeros((num_items, dim), dtype=np.float32)
    item_ids = items_sorted["item_id"].astype("int64").to_numpy()
    out[item_ids] = embs_subset
    return out
