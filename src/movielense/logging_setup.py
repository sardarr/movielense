"""Tiny logging wrapper. Stdlib logging is enough — no extra deps."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    root = logging.getLogger()
    if root.handlers:
        return logging.getLogger("movielense")

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            datefmt="%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    root.setLevel(level)
    return logging.getLogger("movielense")
