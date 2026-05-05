import numpy as np
import pandas as pd

from movielense.data.load import Dataset
from movielense.data.validate import validate


def _good_dataset() -> Dataset:
    rows = []
    for u in range(5):
        for i in range(5):
            rows.append({"user_id": u, "item_id": i, "rating": 5.0, "timestamp": u * 10 + i})
    df = pd.DataFrame(rows)
    return Dataset(
        ratings=df,
        items=pd.DataFrame({"item_id": range(5), "title": [f"t{i}" for i in range(5)]}),
        user_id_to_raw=np.arange(5),
        item_id_to_raw=np.arange(5),
    )


def test_clean_dataset_validates():
    ds = _good_dataset()
    rep = validate(ds, min_user_interactions=1, min_item_interactions=1)
    assert rep.ok
    assert rep.issues == []


def test_duplicate_pair_is_fatal():
    ds = _good_dataset()
    ds.ratings.loc[len(ds.ratings)] = {"user_id": 0, "item_id": 0, "rating": 5.0, "timestamp": 999}
    rep = validate(ds, min_user_interactions=1, min_item_interactions=1)
    assert not rep.ok
