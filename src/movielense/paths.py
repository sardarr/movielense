"""Run-scoped paths. All artifacts for one run live under a single directory."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RunPaths:
    run_id: str
    root: Path

    @property
    def config_snapshot(self) -> Path:
        return self.root / "config.snapshot.yaml"

    @property
    def profile_json(self) -> Path:
        return self.root / "profile.json"

    @property
    def split_dir(self) -> Path:
        return self.root / "splits"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def tuning_dir(self) -> Path:
        return self.root / "tuning"

    @property
    def metrics_json(self) -> Path:
        return self.root / "metrics.json"

    @property
    def selection_json(self) -> Path:
        return self.root / "selection.json"

    @property
    def report_html(self) -> Path:
        return self.root / "dashboard.html"

    @property
    def report_md(self) -> Path:
        return self.root / "research_report.md"

    @property
    def log_file(self) -> Path:
        return self.root / "run.log"


def make_run_paths(name: str | None, base: Path = Path("artifacts/runs")) -> RunPaths:
    if name is None:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        name = f"{ts}-{uuid.uuid4().hex[:6]}"
    root = base / name
    root.mkdir(parents=True, exist_ok=True)
    return RunPaths(run_id=name, root=root)
