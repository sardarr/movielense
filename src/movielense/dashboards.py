"""Dashboard wiring: pick which observers to attach based on config."""
from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

from .observers import CompositeObserver, MLflowObserver, NullObserver, TrainingObserver

log = logging.getLogger(__name__)


@contextmanager
def wandb_session(cfg: dict, run_id: str) -> Iterator[object | None]:
    """Initialize a W&B run if enabled in config and `wandb` is installed."""
    wcfg = cfg.get("dashboards", {}).get("wandb", {})
    if not wcfg.get("enabled"):
        yield None
        return
    try:
        import wandb  # type: ignore
    except ImportError:
        log.warning("W&B enabled in config but `wandb` is not installed; skipping")
        yield None
        return

    mode = wcfg.get("mode", "online")
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        log.warning("WANDB_API_KEY not set — falling back to offline mode")
        mode = "offline"

    run = wandb.init(
        project=wcfg.get("project", "movielense"),
        entity=wcfg.get("entity"),
        name=run_id,
        mode=mode,
        config=cfg,
        reinit=True,
    )
    try:
        yield run
    finally:
        try:
            wandb.finish()
        except Exception:
            pass


def make_observer(cfg: dict, wandb_run) -> TrainingObserver:
    """Compose the active observers based on config + which dashboards initialized."""
    observers: list[TrainingObserver] = []
    if cfg.get("dashboards", {}).get("mlflow", {}).get("enabled", True):
        observers.append(MLflowObserver())
    if wandb_run is not None:
        from .observers import WandbObserver
        observers.append(WandbObserver(wandb_run))
    if not observers:
        return NullObserver()
    return CompositeObserver(observers)
