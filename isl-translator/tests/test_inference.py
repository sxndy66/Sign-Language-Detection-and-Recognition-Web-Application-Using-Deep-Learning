"""Inference state-machine tests: abstention, stability, duplicate suppression."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.inference import InferenceEngine, SentenceBuilder


class _StubModel(nn.Module):
    """A deterministic classifier that always returns a fixed distribution."""

    def __init__(self, logits: np.ndarray):
        super().__init__()
        self._logits = logits

    def forward(self, x):
        return torch.tensor(self._logits, dtype=torch.float32).unsqueeze(0)


@pytest.fixture
def classes() -> list[str]:
    return ["HELLO", "THANK-YOU", "YES"]


@pytest.fixture
def features() -> np.ndarray:
    return np.zeros(186, dtype=np.float32)


def make_engine(classes, logits, **kw) -> InferenceEngine:
    model = _StubModel(logits)
    defaults = dict(sequence_length=60, min_frames=5, confidence_threshold=0.6,
                    ema_alpha=0.35, stability_window=3, cooldown_seconds=0.1)
    defaults.update(kw)
    return InferenceEngine(model=model, classes=classes, device="cpu", **defaults)


def test_abstains_below_threshold(classes, features) -> None:
    # confidence for class 0 is ~0.58 < 0.6 → must abstain
    engine = make_engine(classes, np.log([0.58, 0.30, 0.12]))
    for _ in range(10):
        engine.push(features)
    pred = engine.predict()
    assert pred.status == "UNCERTAIN"
    assert pred.word is None


def test_not_enough_frames_returns_no_hands(classes, features) -> None:
    engine = make_engine(classes, np.log([0.9, 0.05, 0.05]), min_frames=5)
    engine.push(features)  # only one frame buffered
    assert engine.predict().status == "NO_HANDS"


def test_commits_after_stability_window(classes, features) -> None:
    engine = make_engine(classes, np.log([0.9, 0.05, 0.05]), stability_window=3)
    for _ in range(20):
        engine.push(features)
    statuses = [engine.predict().status for _ in range(5)]
    assert "COMMIT" in statuses
    assert engine.sentence.sentence == "HELLO"


def test_duplicate_suppression(classes, features) -> None:
    engine = make_engine(classes, np.log([0.9, 0.05, 0.05]), stability_window=1,
                         cooldown_seconds=10.0)
    for _ in range(20):
        engine.push(features)
    engine.predict()  # commit once
    engine.predict()  # second commit attempt within cooldown
    assert engine.sentence.sentence == "HELLO"  # not "HELLO HELLO"


def test_sentence_undo_and_clear() -> None:
    sb = SentenceBuilder(cooldown_seconds=0.0)
    sb.append("HELLO", 0.9)
    sb.append("GOOD", 0.8)
    assert sb.sentence == "HELLO GOOD"
    sb.undo()
    assert sb.sentence == "HELLO"
    sb.clear()
    assert sb.sentence == ""
