"""Optuna-based hyperparameter tuning, one study per model."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import optuna
import pandas as pd

from .. import models as models_module
from ..eval.runner import evaluate_model
from .spaces import SPACES

log = logging.getLogger(__name__)


@dataclass
class TuneResult:
    model_name: str
    best_params: dict
    best_value: float
    study_summary: dict


def tune_model(
    model_name: str,
    *,
    train_positives: pd.DataFrame,
    val_positives: pd.DataFrame,
    train_for_seen: pd.DataFrame,
    num_users: int,
    num_items: int,
    seed: int,
    n_trials: int,
    timeout: int | None,
    primary_metric: str,
    direction: str,
    eval_ks: list[int],
    candidate_strategy: str,
    sampled_negatives: int,
    sampler: str = "tpe",
    pruner: str = "median",
) -> TuneResult:
    if model_name not in SPACES:
        raise ValueError(f"no search space defined for {model_name}")

    space_fn = SPACES[model_name]

    sampler_obj = (
        optuna.samplers.TPESampler(seed=seed) if sampler == "tpe" else optuna.samplers.RandomSampler(seed=seed)
    )
    pruner_obj = optuna.pruners.MedianPruner() if pruner == "median" else optuna.pruners.NopPruner()

    study = optuna.create_study(
        direction=direction,
        sampler=sampler_obj,
        pruner=pruner_obj,
        study_name=f"movielense-{model_name}",
    )

    def objective(trial: optuna.Trial) -> float:
        params = space_fn(trial)
        model = models_module.build(
            model_name,
            params=params,
            num_users=num_users,
            num_items=num_items,
            seed=seed,
        )
        model.fit(train_positives)
        result = evaluate_model(
            model,
            train=train_for_seen,
            target_positives=val_positives,
            num_items=num_items,
            ks=eval_ks,
            candidate_strategy=candidate_strategy,
            sampled_negatives=sampled_negatives,
        )
        return result.metrics[primary_metric]

    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)
    log.info("tuning %s done: best=%.4f params=%s", model_name, study.best_value, study.best_params)

    summary = {
        "n_trials": len(study.trials),
        "completed": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name,
            }
            for t in study.trials
        ],
    }

    return TuneResult(
        model_name=model_name,
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        study_summary=summary,
    )


def write_tuning_summary(result: TuneResult, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    p = dest_dir / f"{result.model_name}.json"
    with p.open("w") as f:
        json.dump(
            {
                "model_name": result.model_name,
                "best_params": result.best_params,
                "best_value": result.best_value,
                "summary": result.study_summary,
            },
            f,
            indent=2,
        )
    return p
