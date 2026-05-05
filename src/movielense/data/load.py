"""Load MovieLens-100K into pandas, with stable integer ids."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RATINGS_COLS = ["user_raw", "item_raw", "rating", "timestamp"]
ITEM_COLS = ["item_raw", "title", "release_date", "video_release_date", "imdb_url"] + [
    f"genre_{i}" for i in range(19)
]


@dataclass
class Dataset:
    """Holds densified frames + id maps. user_id and item_id are 0..N-1 contiguous."""
    ratings: pd.DataFrame  # columns: user_id, item_id, rating, timestamp
    items: pd.DataFrame    # item_id, title, ...
    user_id_to_raw: np.ndarray
    item_id_to_raw: np.ndarray

    @property
    def num_users(self) -> int:
        return len(self.user_id_to_raw)

    @property
    def num_items(self) -> int:
        return len(self.item_id_to_raw)


def _densify(series: pd.Series) -> tuple[pd.Series, np.ndarray]:
    cats = pd.Categorical(series)
    return pd.Series(cats.codes, index=series.index), np.asarray(cats.categories)


def load_movielens_100k(data_dir: Path) -> Dataset:
    ratings = pd.read_csv(
        data_dir / "u.data", sep="\t", names=RATINGS_COLS, engine="c"
    )
    items = pd.read_csv(
        data_dir / "u.item", sep="|", names=ITEM_COLS, engine="python",
        encoding="latin-1",
    )

    user_codes, user_raw = _densify(ratings["user_raw"])
    item_codes, item_raw = _densify(ratings["item_raw"])
    ratings = ratings.assign(user_id=user_codes, item_id=item_codes)
    ratings = ratings[["user_id", "item_id", "rating", "timestamp"]].sort_values(
        ["user_id", "timestamp"]
    ).reset_index(drop=True)

    item_raw_to_id = pd.Series(
        np.arange(len(item_raw)), index=item_raw, name="item_id"
    )
    items = items.set_index("item_raw").join(item_raw_to_id, how="inner").reset_index()

    log.info(
        "loaded ml-100k: %d ratings, %d users, %d items",
        len(ratings), len(user_raw), len(item_raw),
    )
    return Dataset(
        ratings=ratings,
        items=items,
        user_id_to_raw=user_raw,
        item_id_to_raw=item_raw,
    )
