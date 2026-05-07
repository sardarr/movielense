"""Per-model architecture / explainer text used in the report and dashboard.

Single source of truth so the markdown research report and the HTML dashboard
say the same thing about each model.
"""
from __future__ import annotations

ARCHITECTURE: dict[str, dict[str, str]] = {
    "random": {
        "type": "Baseline (random)",
        "input": "none",
        "training": "no training — generates a deterministic random score per (user, item) using a hash of seed and user id",
        "inference": "O(1) per pair",
        "use_case": "sanity-check floor: any real model must beat this",
        "interpretability": "n/a",
    },
    "popularity": {
        "type": "Baseline (popularity)",
        "input": "global item interaction counts on the training split",
        "training": "single pass over training data to count interactions per item",
        "inference": "O(1) lookup per (user, item)",
        "use_case": "strong baseline; hard to beat for users with mainstream taste; provides no personalization",
        "interpretability": "trivial — score IS popularity",
    },
    "itemknn": {
        "type": "Memory-based collaborative filtering (item-item kNN)",
        "input": "binary user-item interaction matrix",
        "training": "compute cosine similarity between every pair of items; keep the top-k neighbors per item",
        "inference": "score(u, i) = sum of sim(i, j) for j in user u's history (sparse vector dot product)",
        "use_case": "transparent neighbor explanations ('because you watched X'); strong on dense items; struggles with cold-start items",
        "interpretability": "high — top contributing neighbors per recommendation",
    },
    "als": {
        "type": "Matrix factorization (Alternating Least Squares, implicit feedback)",
        "input": "user-item interactions weighted with confidence c(u,i) = 1 + alpha * indicator(u liked i)",
        "training": "alternate ridge regression: fix item factors, solve for user factors (and vice versa) for K iterations",
        "inference": "score(u, i) = U[u] @ V[i] (single dot product of factor vectors)",
        "use_case": "fast batch scoring at scale; learned latent factors capture taste similarities; standard production baseline",
        "interpretability": "low — latent factors have no semantic labels",
    },
    "bpr": {
        "type": "Neural pairwise learning-to-rank (Bayesian Personalized Ranking, PyTorch)",
        "input": "implicit feedback as (user, positive_item, sampled_negative_item) triples",
        "training": (
            "minimize -log sigmoid(<U[u], V[pos]> - <U[u], V[neg]>) over many epochs with Adam; "
            "negatives are uniformly sampled items the user hasn't interacted with"
        ),
        "inference": "score(u, i) = U[u] @ V[i] (dot product of learned embeddings)",
        "use_case": (
            "directly optimizes pairwise ordering, which aligns with ranking metrics like NDCG; "
            "matches the goal of top-K recommendation better than rating-prediction MF"
        ),
        "interpretability": "low — learned embeddings; can be probed via nearest-neighbor inspection",
    },
    "lgbm": {
        "type": "Learning-to-rank with gradient-boosted trees (LightGBM LambdaRank)",
        "input": (
            "engineered features per (user, item): "
            "user side-info (age, gender, occupation), user behavior (interaction count, mean rating, genre preference), "
            "item content (genres, release year, popularity, mean rating), "
            "and an interaction term (user-item genre affinity)"
        ),
        "training": (
            "for each user, sample N negatives per positive; "
            "train LambdaRank with NDCG metric; trees split features to maximize ranking quality"
        ),
        "inference": "predict relevance for each candidate item via summed leaf contributions across trees",
        "use_case": (
            "supports cold-start (no need for prior interactions if features exist) and is fully interpretable via gain / SHAP; "
            "slower at scoring than dot-product models — best when you control candidate-set size upstream"
        ),
        "interpretability": "high — feature importance (gain, split, SHAP) directly available",
    },
    "sasrec": {
        "type": "Sequential transformer (Self-Attentive Sequential Recommendation, PyTorch)",
        "input": (
            "per-user time-ordered item sequence (truncated to max_len most-recent positives); "
            "uses interaction timestamps as the ordering signal"
        ),
        "training": (
            "stack of N causal-masked self-attention blocks with learned positional embeddings; "
            "per-position BPR loss: maximize sigmoid(<h_t, V[s_{t+1}]> - <h_t, V[neg]>); "
            "Adam over batched user sequences"
        ),
        "inference": (
            "encode each user's training history once, cache the final hidden state h_u; "
            "score(u, i) = <h_u, V[i]> — same dot-product cost as ALS / BPR after caching"
        ),
        "use_case": (
            "captures sequential patterns and recency that bag-of-interactions models (ALS, BPR) ignore; "
            "strong when item discovery follows trends or session structure"
        ),
        "interpretability": "low — attention weights can be probed but require additional tooling",
    },
    "sasrec_content_minilm": {
        "type": "Content-collaborative sequential transformer (SASRec + frozen MiniLM)",
        "input": (
            "same time-ordered sequences as SASRec, plus a frozen sentence-transformer "
            "(all-MiniLM-L6-v2) embedding of '{title}. Genres: ... . Year: ...' for each item"
        ),
        "training": (
            "item representation = learned id-embedding + linear projection of frozen text "
            "embedding; transformer + BPR loss are unchanged. Only the projection head and "
            "id-embedding move during training; the 384-d text vectors are read-only"
        ),
        "inference": (
            "score(u, i) = <h_u, id_emb[i] + content_proj(text_emb[i])>; same dot-product cost "
            "as plain SASRec after the fused item matrix is cached"
        ),
        "use_case": (
            "tests whether semantic content (title + genres) can lift cold or long-tail items "
            "where pure id-embeddings have little signal; expected to help most for items with "
            "few training interactions"
        ),
        "interpretability": "low — semantic neighborhoods inspectable via text-embedding cosine, but model decisions still latent",
    },
    "sasrec_content_mpnet": {
        "type": "Content-collaborative SASRec (id + frozen all-mpnet-base-v2, 768d)",
        "input": "same as sasrec_content_minilm but text encoded with sentence-transformers/all-mpnet-base-v2 (768d) — a stronger general-purpose SBERT model than MiniLM-L6",
        "training": "identical to sasrec_content_minilm with a wider content-projection head (768→d_model)",
        "inference": "score(u, i) = <h_u, id_emb[i] + content_proj(text_emb[i])>; same dot-product cost",
        "use_case": "tests whether scaling the encoder (MiniLM 384d → MPNet 768d) lifts the content signal on shallow item text",
        "interpretability": "low",
    },
    "sasrec_content_bge": {
        "type": "Content-collaborative SASRec (id + frozen BAAI/bge-large-en-v1.5, 1024d)",
        "input": "same as sasrec_content_minilm but text encoded with BAAI/bge-large-en-v1.5 (1024d) — a top-tier MTEB encoder in open weights",
        "training": "identical training loop; content-projection head is 1024→d_model",
        "inference": "score(u, i) = <h_u, id_emb[i] + content_proj(text_emb[i])>",
        "use_case": "tests whether a SOTA-class encoder (BGE) recovers more semantic structure than MiniLM/MPNet on title+genre text",
        "interpretability": "low",
    },
    "sasrec_content_mxbai": {
        "type": "Content-collaborative SASRec (id + frozen mxbai-embed-large-v1, 1024d)",
        "input": "same as sasrec_content_minilm but text encoded with mixedbread-ai/mxbai-embed-large-v1 (1024d) — currently #1-2 on MTEB short-text",
        "training": "identical training loop; content-projection head is 1024→d_model",
        "inference": "score(u, i) = <h_u, id_emb[i] + content_proj(text_emb[i])>",
        "use_case": "head-to-head against BGE on the same data — they are both 1024d and same scale; tests encoder choice within the SOTA tier",
        "interpretability": "low",
    },
    "sasrec_content_only": {
        "type": "Content-only sequential transformer (SASRec without id-embeddings)",
        "input": (
            "same time-ordered sequences as SASRec, but the item representation is built "
            "exclusively from frozen sentence-transformer embeddings — there are no learned "
            "per-item id vectors at all"
        ),
        "training": (
            "item representation = linear projection of frozen text embedding (no id-embedding term); "
            "transformer + BPR loss are unchanged. Only the projection head and transformer trunk train"
        ),
        "inference": (
            "score(u, i) = <h_u, content_proj(text_emb[i])>; works for items the model has never "
            "seen during training, since their representation comes purely from text"
        ),
        "use_case": (
            "cold-start probe: how much of SASRec's signal comes from collaborative id-embeddings "
            "vs the semantic content alone? A floor for content-driven recommendation"
        ),
        "interpretability": "low — item representations are projected text; user state still latent",
    },
}
