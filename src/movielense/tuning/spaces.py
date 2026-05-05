"""Per-model Optuna search spaces.

Each function takes an optuna.Trial and returns a params dict.
"""
from __future__ import annotations

from collections.abc import Callable

import optuna


def itemknn_space(trial: optuna.Trial) -> dict:
    return {
        "k": trial.suggest_int("k", 10, 200, log=True),
        "similarity": "cosine",
    }


def als_space(trial: optuna.Trial) -> dict:
    return {
        "factors": trial.suggest_categorical("factors", [16, 32, 64, 128]),
        "regularization": trial.suggest_float("regularization", 1e-3, 1.0, log=True),
        "iterations": trial.suggest_int("iterations", 5, 25),
        "alpha": trial.suggest_float("alpha", 1.0, 80.0, log=True),
    }


def bpr_space(trial: optuna.Trial) -> dict:
    return {
        "factors": trial.suggest_categorical("factors", [16, 32, 64, 128]),
        "lr": trial.suggest_float("lr", 1e-4, 5e-2, log=True),
        "reg": trial.suggest_float("reg", 1e-5, 1e-1, log=True),
        "epochs": trial.suggest_int("epochs", 5, 25),
        "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),
    }


def lgbm_space(trial: optuna.Trial) -> dict:
    # Search space narrowed to keep tuning latency bounded; tuned from offline runs
    # where wider ranges did not improve NDCG@10 within the budget.
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 63, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.15, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "n_neg_per_pos": trial.suggest_int("n_neg_per_pos", 1, 4),
    }


def sasrec_space(trial: optuna.Trial) -> dict:
    return {
        "d_model": trial.suggest_categorical("d_model", [32, 64]),
        "num_blocks": trial.suggest_int("num_blocks", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.05, 0.4),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),
        "epochs": trial.suggest_int("epochs", 10, 40),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
    }


SPACES: dict[str, Callable[[optuna.Trial], dict]] = {
    "itemknn": itemknn_space,
    "als": als_space,
    "bpr": bpr_space,
    "lgbm": lgbm_space,
    "sasrec": sasrec_space,
}
