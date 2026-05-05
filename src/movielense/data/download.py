"""Download MovieLens-100K (or load from cache)."""
from __future__ import annotations

import hashlib
import logging
import zipfile
from pathlib import Path

import requests

log = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(url: str, raw_dir: Path) -> Path:
    """Download zip if missing, extract, return the directory containing the data files.

    Returns the inner extracted folder (e.g. data/raw/ml-100k/).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_name = url.rsplit("/", 1)[-1]
    zip_path = raw_dir / zip_name

    if not zip_path.exists():
        log.info("Downloading %s -> %s", url, zip_path)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with zip_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    else:
        log.info("Cache hit: %s", zip_path)

    digest = _sha256(zip_path)
    log.info("dataset zip sha256=%s", digest)

    inner = raw_dir / zip_path.stem  # e.g. ml-100k
    if not inner.exists() or not any(inner.iterdir()):
        log.info("Extracting %s", zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)

    if not inner.exists():
        # Some zips extract into a different directory; find any folder with u.data
        for sub in raw_dir.iterdir():
            if sub.is_dir() and (sub / "u.data").exists():
                inner = sub
                break

    if not (inner / "u.data").exists():
        raise FileNotFoundError(f"Expected u.data in {inner}")

    return inner
