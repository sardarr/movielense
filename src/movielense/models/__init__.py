from .als import ALSRecommender
from .base import Recommender
from .baselines import PopularityRecommender, RandomRecommender
from .bpr import BPRRecommender
from .item_knn import ItemKNNRecommender
from .lgbm import LGBMRankerRecommender


def build(
    name: str,
    params: dict,
    *,
    num_users: int,
    num_items: int,
    seed: int,
    feature_store=None,
) -> Recommender:
    name = name.lower()
    if name == "random":
        return RandomRecommender(num_users=num_users, num_items=num_items, seed=seed)
    if name == "popularity":
        return PopularityRecommender(num_users=num_users, num_items=num_items)
    if name == "itemknn":
        return ItemKNNRecommender(num_users=num_users, num_items=num_items, **params)
    if name == "als":
        return ALSRecommender(num_users=num_users, num_items=num_items, seed=seed, **params)
    if name == "bpr":
        return BPRRecommender(num_users=num_users, num_items=num_items, seed=seed, **params)
    if name == "lgbm":
        if feature_store is None:
            raise ValueError("lgbm model requires feature_store")
        return LGBMRankerRecommender(
            num_users=num_users,
            num_items=num_items,
            feature_store=feature_store,
            seed=seed,
            **params,
        )
    raise ValueError(f"unknown model: {name}")


__all__ = [
    "Recommender",
    "PopularityRecommender",
    "RandomRecommender",
    "BPRRecommender",
    "ItemKNNRecommender",
    "ALSRecommender",
    "LGBMRankerRecommender",
    "build",
]
