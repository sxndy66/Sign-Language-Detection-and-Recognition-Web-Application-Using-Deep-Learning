"""Utility tests: config loading, sanitization, latency meter, seeds."""
from __future__ import annotations

import numpy as np
import pytest

from src.utils import LatencyMeter, sanitize_array, set_seed


def test_sanitize_array_replaces_nonfinite() -> None:
    arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
    out = sanitize_array(arr)
    assert np.isfinite(out).all()
    assert out[1] == 0.0 and out[2] == 0.0 and out[3] == 0.0
    assert out[0] == 1.0 and out[4] == 2.0


def test_latency_meter_percentile() -> None:
    meter = LatencyMeter(percentile=50.0)
    for s in [0.001, 0.002, 0.003, 0.004]:
        meter.record(s)
    assert meter.p() == pytest.approx(0.0025, abs=1e-6)
    assert meter.count == 4


def test_set_seed_deterministic() -> None:
    set_seed(42)
    a = np.random.rand(10)
    set_seed(42)
    b = np.random.rand(10)
    assert np.array_equal(a, b)


def test_empty_latency_meter_is_nan() -> None:
    assert np.isnan(LatencyMeter().p())
