"""Load a MovieLens dataset into pandas with stable contiguous ids.

Supported variants:
  - ml-100k  : tab-separated u.data, pipe-separated u.item / u.user
  - ml-1m    : ::-separated ratings.dat / movies.dat / users.dat

All variants land in the same `Dataset` shape: user_id and item_id densified to
[0, N), items frame has `genre_<Name>` boolean columns and a `release_year`,
users frame has age/gender/occupation when available.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Canonical genre list shared by all variants. ML-1M's "Children's" gets
# normalized to "Children".
GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
GENRE_COLS = [f"genre_{g}" for g in GENRE_NAMES]


@dataclass
class Dataset:
    """Densified frames + id maps. `users` may be empty if side-info is absent."""
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


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _densify(series: pd.Series) -> tuple[pd.Series, np.ndarray]:
    cats = pd.Categorical(series)
    return pd.Series(cats.codes, index=series.index), np.asarray(cats.categories)


def _attach_user_ids(users: pd.DataFrame, user_raw: np.ndarray) -> pd.DataFrame:
    if users.empty:
        return users
    user_raw_to_id = pd.Series(np.arange(len(user_raw)), index=user_raw, name="user_id")
    users = users.set_index("user_raw").join(user_raw_to_id, how="inner").reset_index()
    users = users.dropna(subset=["user_id"]).copy()
    users["user_id"] = users["user_id"].astype("int64")
    return users


def _attach_item_ids(items: pd.DataFrame, item_raw: np.ndarray) -> pd.DataFrame:
    item_raw_to_id = pd.Series(np.arange(len(item_raw)), index=item_raw, name="item_id")
    items = items.set_index("item_raw").join(item_raw_to_id, how="inner").reset_index()
    return items


def _zero_genre_block(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        np.zeros((n, len(GENRE_COLS)), dtype=np.int8), columns=GENRE_COLS
    )


# ---------------------------------------------------------------------------
# ML-100K
# ---------------------------------------------------------------------------

_ML100K_RATINGS_COLS = ["user_raw", "item_raw", "rating", "timestamp"]
_ML100K_ITEM_COLS = (
    ["item_raw", "title", "release_date", "video_release_date", "imdb_url"]
    + [f"raw_genre_{i}" for i in range(len(GENRE_NAMES))]
)
_ML100K_USER_COLS = ["user_raw", "age", "gender", "occupation", "zip"]


def _parse_release_year_100k(series: pd.Series) -> pd.Series:
    years = pd.to_datetime(series, format="%d-%b-%Y", errors="coerce").dt.year
    return years.fillna(years.median()).astype("int32")


def load_movielens_100k(data_dir: Path) -> Dataset:
    ratings = pd.read_csv(data_dir / "u.data", sep="\t", names=_ML100K_RATINGS_COLS, engine="c")
    items = pd.read_csv(
        data_dir / "u.item", sep="|", names=_ML100K_ITEM_COLS, engine="python",
        encoding="latin-1",
    )
    users_path = data_dir / "u.user"
    users = (
        pd.read_csv(users_path, sep="|", names=_ML100K_USER_COLS, engine="c")
        if users_path.exists() else pd.DataFrame()
    )

    user_codes, user_raw = _densify(ratings["user_raw"])
    item_codes, item_raw = _densify(ratings["item_raw"])
    ratings = ratings.assign(user_id=user_codes, item_id=item_codes)[
        ["user_id", "item_id", "rating", "timestamp"]
    ].sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    items = _attach_item_ids(items, item_raw)
    items["release_year"] = _parse_release_year_100k(items["release_date"])
    items = items.rename(
        columns={f"raw_genre_{i}": f"genre_{name}" for i, name in enumerate(GENRE_NAMES)}
    )
    users = _attach_user_ids(users, user_raw)

    log.info(
        "loaded ml-100k: %d ratings, %d users (side-info: %s), %d items",
        len(ratings), len(user_raw), "yes" if not users.empty else "no", len(item_raw),
    )
    return Dataset(
        ratings=ratings, items=items, users=users,
        user_id_to_raw=user_raw, item_id_to_raw=item_raw,
    )


# ---------------------------------------------------------------------------
# ML-1M
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def _parse_year_from_title(title: str) -> int | None:
    m = _YEAR_RE.search(title)
    return int(m.group(1)) if m else None


def _normalize_genre(name: str) -> str:
    # ML-1M uses "Children's" — strip apostrophe-s to match canonical "Children".
    if name == "Children's":
        return "Children"
    return name


def _expand_genres(genre_strings: pd.Series) -> pd.DataFrame:
    """Turn 'Action|Adventure|Sci-Fi' into a one-hot frame matching GENRE_COLS."""
    rows = np.zeros((len(genre_strings), len(GENRE_NAMES)), dtype=np.int8)
    name_to_idx = {g: i for i, g in enumerate(GENRE_NAMES)}
    for r, s in enumerate(genre_strings.fillna("")):
        for tok in s.split("|"):
            tok = _normalize_genre(tok.strip())
            if tok in name_to_idx:
                rows[r, name_to_idx[tok]] = 1
    return pd.DataFrame(rows, columns=GENRE_COLS, index=genre_strings.index)


def load_movielens_1m(data_dir: Path) -> Dataset:
    # ratings.dat: UserID::MovieID::Rating::Timestamp
    ratings = pd.read_csv(
        data_dir / "ratings.dat", sep="::", engine="python",
        names=["user_raw", "item_raw", "rating", "timestamp"],
        encoding="latin-1",
    )
    # movies.dat: MovieID::Title::Genres
    movies = pd.read_csv(
        data_dir / "movies.dat", sep="::", engine="python",
        names=["item_raw", "title", "genres"],
        encoding="latin-1",
    )
    # users.dat: UserID::Gender::Age::Occupation::Zip-code
    users_path = data_dir / "users.dat"
    if users_path.exists():
        users = pd.read_csv(
            users_path, sep="::", engine="python",
            names=["user_raw", "gender", "age", "occupation", "zip"],
            encoding="latin-1",
        )
        # ML-1M occupation is an integer code; convert to string so one-hot works
        # the same way as ML-100K.
        users["occupation"] = users["occupation"].astype(str)
    else:
        users = pd.DataFrame()

    user_codes, user_raw = _densify(ratings["user_raw"])
    item_codes, item_raw = _densify(ratings["item_raw"])
    ratings = ratings.assign(user_id=user_codes, item_id=item_codes)[
        ["user_id", "item_id", "rating", "timestamp"]
    ].sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    items = _attach_item_ids(movies, item_raw)
    genre_block = _expand_genres(items["genres"])
    items = pd.concat([items.drop(columns=["genres"]), genre_block], axis=1)
    items["release_year"] = items["title"].map(_parse_year_from_title)
    items["release_year"] = items["release_year"].fillna(items["release_year"].median()).astype("int32")

    users = _attach_user_ids(users, user_raw)

    log.info(
        "loaded ml-1m: %d ratings, %d users (side-info: %s), %d items",
        len(ratings), len(user_raw), "yes" if not users.empty else "no", len(item_raw),
    )
    return Dataset(
        ratings=ratings, items=items, users=users,
        user_id_to_raw=user_raw, item_id_to_raw=item_raw,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def load_movielens(name: str, data_dir: Path) -> Dataset:
    name = name.lower()
    if name == "ml-100k":
        return load_movielens_100k(data_dir)
    if name == "ml-1m":
        return load_movielens_1m(data_dir)
    raise ValueError(f"unsupported dataset: {name}")
