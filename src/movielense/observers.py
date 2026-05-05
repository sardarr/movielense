"""Training observers — both MLflow and W&B consume the same event stream.

Models call observer.on_epoch_end(epoch, metrics) during training. The
orchestrator wires up which observers are active per run.
"""
from __future__ import annotations

import logging
from typing import Protocol

log = logging.getLogger(__name__)


class TrainingObserver(Protocol):
    def on_train_start(self, model_name: str, params: dict) -> None: ...
    def on_epoch_end(self, model_name: str, epoch: int, metrics: dict[str, float]) -> None: ...
    def on_train_end(self, model_name: str) -> None: ...


class NullObserver:
    """No-op observer — used when no dashboards are wired up."""
    def on_train_start(self, model_name: str, params: dict) -> None: ...
    def on_epoch_end(self, model_name: str, epoch: int, metrics: dict[str, float]) -> None: ...
    def on_train_end(self, model_name: str) -> None: ...


class CompositeObserver:
    """Fans out events to a list of observers; one failing observer doesn't break others."""

    def __init__(self, observers: list[TrainingObserver]):
        self.observers = observers

    def on_train_start(self, model_name: str, params: dict) -> None:
        for o in self.observers:
            try:
                o.on_train_start(model_name, params)
            except Exception as e:
                log.warning("observer %s on_train_start failed: %s", type(o).__name__, e)

    def on_epoch_end(self, model_name: str, epoch: int, metrics: dict[str, float]) -> None:
        for o in self.observers:
            try:
                o.on_epoch_end(model_name, epoch, metrics)
            except Exception as e:
                log.warning("observer %s on_epoch_end failed: %s", type(o).__name__, e)

    def on_train_end(self, model_name: str) -> None:
        for o in self.observers:
            try:
                o.on_train_end(model_name)
            except Exception as e:
                log.warning("observer %s on_train_end failed: %s", type(o).__name__, e)


class MLflowObserver:
    """Logs per-epoch metrics to the currently active MLflow run.

    The orchestrator owns the run; the observer just calls log_metric.
    """

    def on_train_start(self, model_name: str, params: dict) -> None:
        import mlflow
        mlflow.log_params({f"hp/{k}": v for k, v in params.items()})

    def on_epoch_end(self, model_name: str, epoch: int, metrics: dict[str, float]) -> None:
        import mlflow
        for k, v in metrics.items():
            mlflow.log_metric(f"train/{k}", float(v), step=int(epoch))

    def on_train_end(self, model_name: str) -> None:
        return None


class WandbObserver:
    """Logs per-epoch metrics to W&B. Initialization is owned by the orchestrator."""

    def __init__(self, run):
        self.run = run

    def on_train_start(self, model_name: str, params: dict) -> None:
        try:
            self.run.config.update({f"{model_name}/{k}": v for k, v in params.items()},
                                   allow_val_change=True)
        except Exception:
            pass

    def on_epoch_end(self, model_name: str, epoch: int, metrics: dict[str, float]) -> None:
        payload = {f"{model_name}/train/{k}": float(v) for k, v in metrics.items()}
        payload[f"{model_name}/epoch"] = int(epoch)
        self.run.log(payload)

    def on_train_end(self, model_name: str) -> None:
        return None
