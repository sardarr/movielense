from .als import ALSRecommender
from .base import Recommender
from .baselines import PopularityRecommender, RandomRecommender
from .bpr import BPRRecommender
from .item_knn import ItemKNNRecommender
from .lgbm import LGBMRankerRecommender
from .sasrec import SASRecRecommender


def build(
    name: str,
    params: dict,
    *,
    num_users: int,
    num_items: int,
    seed: int,
    feature_store=None,
    content_embeddings=None,
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
    if name == "sasrec" or name.startswith("sasrec_content"):
        ce_cfg = dict(params.pop("content", {}) or {})
        mode = ce_cfg.get("mode", "off")
        encoder = ce_cfg.get("encoder")
        # content_embeddings here is either an ndarray (single-encoder back-compat)
        # or a dict {encoder_name: ndarray} when multiple encoders are in play.
        emb_for_model = None
        if mode != "off":
            if content_embeddings is None:
                raise ValueError(f"{name} requires content_embeddings (mode={mode})")
            if isinstance(content_embeddings, dict):
                if encoder not in content_embeddings:
                    raise ValueError(
                        f"{name} needs encoder='{encoder}' but only "
                        f"{list(content_embeddings.keys())} were precomputed"
                    )
                emb_for_model = content_embeddings[encoder]
            else:
                emb_for_model = content_embeddings
        return SASRecRecommender(
            num_users=num_users,
            num_items=num_items,
            seed=seed,
            content_embeddings=emb_for_model,
            content_mode=mode,
            content_encoder=encoder,
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
    "SASRecRecommender",
    "build",
]
