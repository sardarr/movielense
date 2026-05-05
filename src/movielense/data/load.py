"""Load MovieLens-100K into pandas, with stable integer ids."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RATINGS_COLS = ["user_raw", "item_raw", "rating", "timestamp"]
GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
ITEM_COLS = (
    ["item_raw", "title", "release_date", "video_release_date", "imdb_url"]
    + [f"genre_{i}" for i in range(len(GENRE_NAMES))]
)
USER_COLS = ["user_raw", "age", "gender", "occupation", "zip"]


@dataclass
class Dataset:
    """Holds densified frames + id maps. user_id and item_id are 0..N-1 contiguous.

    `users` may be empty if the side-info file is absent.
    """
    ratings: pd.DataFrame
    items: pd.DataFrame
    users: pd.DataFrame = field(default_factory=pd.DataFrame)
    user_id_to_raw: np.ndarray = field(default_factory=lambda: np.array([]))
    item_id_to_raw: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def num_users(self) -> int:
        return len(self.user_id_to_raw)

    @property
    def num_items(self) -> int:
        return len(self.item_id_to_raw)


def _densify(series: pd.Series) -> tuple[pd.Series, np.ndarray]:
    cats = pd.Categorical(series)
    return pd.Series(cats.codes, index=series.index), np.asarray(cats.categories)


def _parse_release_year(series: pd.Series) -> pd.Series:
    # release_date like "01-Jan-1995" — last 4 chars usually the year.
    years = pd.to_datetime(series, format="%d-%b-%Y", errors="coerce").dt.year
    return years.fillna(years.median()).astype("int32")


def load_movielens_100k(data_dir: Path) -> Dataset:
    ratings = pd.read_csv(
        data_dir / "u.data", sep="\t", names=RATINGS_COLS, engine="c"
    )
    items = pd.read_csv(
        data_dir / "u.item", sep="|", names=ITEM_COLS, engine="python",
        encoding="latin-1",
    )
    users_path = data_dir / "u.user"
    users = pd.read_csv(users_path, sep="|", names=USER_COLS, engine="c") if users_path.exists() else pd.DataFrame()

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
    items["release_year"] = _parse_release_year(items["release_date"])
    # Friendly genre names.
    items = items.rename(columns={f"genre_{i}": f"genre_{name}" for i, name in enumerate(GENRE_NAMES)})

    if not users.empty:
        user_raw_to_id = pd.Series(
            np.arange(len(user_raw)), index=user_raw, name="user_id"
        )
        users = users.set_index("user_raw").join(user_raw_to_id, how="inner").reset_index()
        users = users.dropna(subset=["user_id"]).copy()
        users["user_id"] = users["user_id"].astype("int64")

    log.info(
        "loaded ml-100k: %d ratings, %d users (side-info: %s), %d items",
        len(ratings), len(user_raw), "yes" if not users.empty else "no", len(item_raw),
    )
    return Dataset(
        ratings=ratings,
        items=items,
        users=users,
        user_id_to_raw=user_raw,
        item_id_to_raw=item_raw,
    )
