"""Shared helpers for the MIMIC-IV → EHRFormer preprocessing pipeline.

Conventions:
- All intermediate tables are parquet (fast columnar IO; chunkable).
- All paths come from configs/preprocess.yaml.
- Each pipeline stage writes to {work_dir}/{stage}/ so stages are independently rerunnable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "preprocess.yaml"


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else CONFIG_PATH
    with open(p) as f:
        cfg = yaml.safe_load(f)
    # Resolve work_dir relative to repo root if not absolute
    work = Path(cfg["paths"]["work_dir"]).expanduser()
    if not work.is_absolute():
        work = REPO_ROOT / work
    cfg["paths"]["work_dir"] = str(work)
    return cfg


def setup_logging(stage: str, work_dir: str | Path):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [logging.FileHandler(log_path), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    return logging.getLogger(stage)


def stage_dir(cfg: dict, stage: str) -> Path:
    p = Path(cfg["paths"]["work_dir"]) / stage
    p.mkdir(parents=True, exist_ok=True)
    return p


def detect_table_path(root: str | Path, name: str) -> Path:
    """Find a MIMIC table file by stem name; supports parquet, csv.gz, csv.

    MIMIC-IV layout (v3.1) uses subdirectories like `hosp/` and `icu/`.
    """
    root = Path(root)
    candidates = []
    for ext in [".parquet", ".csv.gz", ".csv"]:
        candidates += list(root.rglob(f"{name}{ext}"))
    if not candidates:
        raise FileNotFoundError(f"Cannot find table '{name}' under {root}")
    # Prefer parquet > csv.gz > csv
    candidates.sort(key=lambda p: {".parquet": 0, ".gz": 1, ".csv": 2}.get(p.suffix, 9))
    return candidates[0]


def read_table(path: str | Path, columns: list[str] | None = None, **kwargs):
    """Read a MIMIC table into pandas, regardless of csv/csv.gz/parquet."""
    import pandas as pd

    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns, **kwargs)
    return pd.read_csv(path, usecols=columns, low_memory=False, **kwargs)
