"""MLflow tracking + model registry.

The orchestrator owns the lifetime of each MLflow run so that observers can
stream live metrics into it during training. After training, log_finalize_run
records params + final metrics + the model artifact, then closes the run.

Run lifecycle:
    setup_tracking(...)
    with start_model_run("bpr") as ctx:
        # observers log per-epoch metrics
        model.fit(train_pos, observer=...)
        log_finalize_run(ctx, model, params, val_metrics, test_metrics, artifacts)
"""
from __future__ import annotations

import logging
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import mlflow

log = logging.getLogger(__name__)


@dataclass
class RunContext:
    run_id: str


def setup_tracking(tracking_uri: str, experiment_name: str) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri


@contextmanager
def start_model_run(model_name: str, tags: dict | None = None) -> Iterator[RunContext]:
    tags = tags or {}
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tags({"model": model_name, **tags})
        try:
            yield RunContext(run_id=run.info.run_id)
        finally:
            pass  # context manager closes the run


def _safe_metric_name(name: str) -> str:
    """MLflow rejects '@' in metric names — replace with '_at_'."""
    return name.replace("@", "_at_")


def log_finalize_run(
    ctx: RunContext,
    *,
    model,
    params: dict,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    artifacts: dict[str, Path] | None = None,
) -> None:
    """Log final params, metrics, and the pickled model into the *currently active* run."""
    artifacts = artifacts or {}
    mlflow.log_params({k: v for k, v in params.items()})
    mlflow.log_metrics({f"val/{_safe_metric_name(k)}": v for k, v in val_metrics.items()})
    mlflow.log_metrics({f"test/{_safe_metric_name(k)}": v for k, v in test_metrics.items()})

    artifact_dir = Path(mlflow.get_artifact_uri().replace("file://", ""))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = artifact_dir / "model.pkl"
    try:
        with pkl_path.open("wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(str(pkl_path), artifact_path="model")
    except Exception as e:
        log.warning("failed to pickle model: %s", e)

    for name, path in artifacts.items():
        if Path(path).exists():
            mlflow.log_artifact(str(path), artifact_path=name)


def register_winner(run_id: str, registered_model_name: str, *, stage: str = "Production") -> str | None:
    """Register the winning run's model as a new version of the registered model."""
    try:
        client = mlflow.MlflowClient()
        try:
            client.get_registered_model(registered_model_name)
        except Exception:
            client.create_registered_model(registered_model_name)

        model_uri = f"runs:/{run_id}/model"
        mv = client.create_model_version(
            name=registered_model_name,
            source=model_uri,
            run_id=run_id,
        )
        try:
            client.transition_model_version_stage(
                name=registered_model_name, version=mv.version, stage=stage
            )
        except Exception:
            pass
        log.info("registered %s v%s -> %s", registered_model_name, mv.version, stage)
        return mv.version
    except Exception as e:
        log.warning("registry registration failed: %s", e)
        return None
