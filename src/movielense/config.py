"""Config loading. The full config dict is captured into every run snapshot."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def models_enabled(self) -> list[str]:
        return list(self.raw["models"].keys())

    @property
    def primary_metric(self) -> str:
        return self.raw["selection"]["objective"]


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, path=path)


def write_config_snapshot(cfg: Config, dest: Path) -> Path:
    """Persist the exact config used for a run into the run's artifact dir."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        yaml.safe_dump(cfg.raw, f, sort_keys=False)
    return dest
