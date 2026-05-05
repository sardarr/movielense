import math

from movielense.eval.metrics import (
    average_precision_at_k,
    evaluate_user_rankings,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_hit_rate():
    assert hit_rate_at_k([1, 2, 3], {2}, 3) == 1.0
    assert hit_rate_at_k([1, 2, 3], {99}, 3) == 0.0
    assert hit_rate_at_k([1, 2, 3], {3}, 2) == 0.0


def test_precision_recall():
    ranked = [10, 20, 30, 40]
    rel = {20, 30}
    assert precision_at_k(ranked, rel, 4) == 0.5
    assert precision_at_k(ranked, rel, 2) == 0.5
    assert recall_at_k(ranked, rel, 4) == 1.0
    assert recall_at_k(ranked, rel, 1) == 0.0


def test_map():
    # Both relevant items at positions 2 and 3.
    # AP = (1/2 + 2/3) / 2 = (0.5 + 0.6667) / 2
    ranked = [1, 2, 3, 4]
    rel = {2, 3}
    expected = (0.5 + 2 / 3) / 2
    assert math.isclose(average_precision_at_k(ranked, rel, 4), expected, rel_tol=1e-6)


def test_mrr():
    assert reciprocal_rank_at_k([1, 2, 3], {3}, 3) == 1 / 3
    assert reciprocal_rank_at_k([1, 2, 3], {99}, 3) == 0.0


def test_ndcg_perfect_is_one():
    ranked = [1, 2, 3]
    rel = {1, 2}
    # When relevant items appear at top positions, DCG == IDCG.
    assert math.isclose(ndcg_at_k(ranked, rel, 3), 1.0, rel_tol=1e-6)


def test_ndcg_decreases_when_relevant_buried():
    ranked_top = [1, 2, 3]
    ranked_buried = [3, 2, 1]
    assert ndcg_at_k(ranked_top, {1}, 3) > ndcg_at_k(ranked_buried, {1}, 3)


def test_evaluate_user_rankings_skips_users_without_relevant():
    rankings = [
        ([1, 2, 3], {1}),
        ([1, 2, 3], set()),  # skipped
    ]
    out = evaluate_user_rankings(rankings, ks=[3])
    assert math.isclose(out["hr@3"], 1.0)
