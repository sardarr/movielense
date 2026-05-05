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


SPACES: dict[str, Callable[[optuna.Trial], dict]] = {
    "itemknn": itemknn_space,
    "als": als_space,
    "bpr": bpr_space,
}
