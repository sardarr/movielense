"""Schema and integrity validation for the loaded dataset."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .load import Dataset

log = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    ok: bool
    issues: list[str]


def validate(ds: Dataset, *, min_user_interactions: int, min_item_interactions: int) -> ValidationReport:
    issues: list[str] = []

    if ds.ratings.isna().any().any():
        issues.append("ratings frame contains nulls")

    if (ds.ratings["rating"] < 1).any() or (ds.ratings["rating"] > 5).any():
        issues.append("rating values outside [1,5]")

    if ds.ratings["user_id"].max() != ds.num_users - 1:
        issues.append("user_id not densified to [0, num_users)")

    if ds.ratings["item_id"].max() != ds.num_items - 1:
        issues.append("item_id not densified to [0, num_items)")

    dups = ds.ratings.duplicated(["user_id", "item_id"]).sum()
    if dups > 0:
        issues.append(f"{dups} duplicate (user, item) pairs")

    user_counts = ds.ratings.groupby("user_id").size()
    sparse_users = int((user_counts < min_user_interactions).sum())
    if sparse_users:
        issues.append(f"{sparse_users} users with <{min_user_interactions} interactions")

    item_counts = ds.ratings.groupby("item_id").size()
    sparse_items = int((item_counts < min_item_interactions).sum())
    if sparse_items:
        issues.append(f"{sparse_items} items with <{min_item_interactions} interactions")

    fatal_keys = ("null", "outside", "densified", "duplicate")
    fatal = [i for i in issues if any(k in i for k in fatal_keys)]
    ok = len(fatal) == 0

    if issues:
        for i in issues:
            log.warning("validation: %s", i)
    else:
        log.info("validation: OK")

    return ValidationReport(ok=ok, issues=issues)
