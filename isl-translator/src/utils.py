"""Shared utilities: config loading, logging, device selection, timing.

Small, dependency-light helpers used across the pipeline so behaviour stays
consistent between training, evaluation and inference.
"""
from __future__ import annotations

import json
import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def setup_logger(name: str = "isl", level: str = "INFO") -> logging.Logger:
    """Return a module logger with a single, stable handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger


log = setup_logger()


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def load_config(path: str | Path) -> dict[str, Any]:
    """Load and return a YAML config as a nested dict."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config at {path} did not parse to a mapping")
    return cfg


def load_json(path: str | Path, default: Any = None) -> Any:
    """Load JSON from disk, returning ``default`` when the file is missing."""
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, obj: Any) -> None:
    """Atomically-ish write JSON, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    tmp.replace(p)


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #

def sanitize_array(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Replace non-finite values with ``fill``; never silently propagate NaN."""
    bad = ~np.isfinite(arr)
    if np.any(bad):
        arr = np.array(arr, dtype=np.float64)
        arr[bad] = fill
    return arr


def set_seed(seed: int) -> None:
    """Deterministic seeds for python, numpy and (if importable) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: PLC0415 (lazy import so utils stays light)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def select_device() -> str:
    """Return 'cuda' when available and requested, else 'cpu'."""
    try:
        import torch  # noqa: PLC0415

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #

class LatencyMeter:
    """Percentile latency meter (defaults to p95) for inference budgets."""

    def __init__(self, percentile: float = 95.0, maxlen: int = 10_000) -> None:
        self.percentile = percentile
        self._samples: list[float] = []

    def record(self, seconds: float) -> None:
        self._samples.append(seconds)

    def p(self) -> float:
        if not self._samples:
            return float("nan")
        return float(np.percentile(self._samples, self.percentile))

    @property
    def count(self) -> int:
        return len(self._samples)


@contextmanager
def timed(meter: LatencyMeter | None = None) -> Iterator[None]:
    """Time a block; optionally feed the elapsed seconds into a meter."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if meter is not None:
        meter.record(elapsed)
