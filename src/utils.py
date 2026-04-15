"""Shared helpers: config loading, run-dir management, metric logging."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p) as f:
        return yaml.safe_load(f)


def new_run_dir(tag: str) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    run = REPO_ROOT / "results" / "runs" / f"{ts}-{tag}"
    run.mkdir(parents=True, exist_ok=True)
    return run


def save_metrics(run_dir: Path, metrics: dict[str, Any]) -> Path:
    out = run_dir / "metrics.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return out


@dataclass
class RunContext:
    """Fixed-schema record for one experimental run; aggregated in 04_pareto."""
    tag: str                # e.g. "baseline-1b", "ptq4-1b", "distill+ptq4-1b"
    model: str              # HF model id or "distilled-1b"
    quantization: str       # "none" | "int8" | "nf4"
    distilled: bool
    perplexity: float | None = None
    tokens_per_sec: float | None = None
    peak_vram_gb: float | None = None
    ttft_ms: float | None = None
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        return d
