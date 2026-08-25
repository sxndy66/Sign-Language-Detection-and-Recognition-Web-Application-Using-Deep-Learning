"""The inference state machine — from landmarks to committed text.

Stages: normalize → temporal buffer → model forward → tempered softmax →
confidence filter (abstain below threshold) → per-class EMA smoothing →
stability window → commit → duplicate-suppressing sentence builder.

A translator that says "I don't know" is worth more than one that guesses.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from .preprocessing import FEATURE_DIMENSION
from .utils import log, timed


@dataclass
class Prediction:
    """A single classification decision, with or without a commitment."""

    word: str | None
    confidence: float
    status: str = "UNCERTAIN"       # COMMIT | STABLE | UNCERTAIN | NO_HANDS
    timestamp: float = field(default_factory=time.time)

    @property
    def committed(self) -> bool:
        return self.status == "COMMIT"


@dataclass
class Word:
    """A committed token in the output sentence."""

    text: str
    confidence: float
    t: float = field(default_factory=time.time)


class SentenceBuilder:
    """Committed words → "HELLO GOOD MORNING". Corrections are first-class."""

    def __init__(self, cooldown_seconds: float = 1.4) -> None:
        self.cooldown = cooldown_seconds
        self.words: list[Word] = []
        self._last_commit_at = -float("inf")
        self._last_text: str | None = None

    @property
    def sentence(self) -> str:
        return " ".join(w.text for w in self.words)

    def append(self, word: str, confidence: float) -> bool:
        """Append unless it duplicates the previous word inside the cooldown."""
        now = time.time()
        if word == self._last_text and now - self._last_commit_at < self.cooldown:
            log.info("duplicate suppressed: %s", word)
            return False
        self.words.append(Word(text=word, confidence=confidence, t=now))
        self._last_text = word
        self._last_commit_at = now
        return True

    def undo(self) -> None:
        if self.words:
            removed = self.words.pop()
            self._last_text = self.words[-1].text if self.words else None
            self._last_commit_at = -float("inf")
            log.info("undo removed: %s", removed.text)

    def clear(self) -> None:
        self.words.clear()
        self._last_text = None
        self._last_commit_at = -float("inf")

    def reset(self) -> None:
        self.clear()


class InferenceEngine:
    """Holds the temporal buffer and the gated commit state machine."""

    def __init__(
        self,
        model: torch.nn.Module,
        classes: list[str],
        device: str = "cpu",
        sequence_length: int = 60,
        min_frames: int = 30,
        confidence_threshold: float = 0.60,
        temperature: float = 1.0,
        ema_alpha: float = 0.35,
        stability_window: int = 12,
        cooldown_seconds: float = 1.4,
    ) -> None:
        self.model = model.to(device).eval()
        self.classes = classes
        self.device = device
        self.sequence_length = sequence_length
        self.min_frames = min_frames
        self.threshold = confidence_threshold
        self.temperature = temperature
        self.alpha = ema_alpha
        self.stability_window = stability_window

        self.buffer: deque[np.ndarray] = deque(maxlen=sequence_length)
        self.ema: np.ndarray = np.zeros(len(classes), dtype=np.float64)
        self._prev_class: int | None = None
        self._stable: int = 0
        self._cooldown_until: float = 0.0
        self.sentence = SentenceBuilder(cooldown_seconds)

    # -- ingestion ---------------------------------------------------------- #

    def push(self, features: np.ndarray) -> None:
        """Append a (186,) feature row; a missing hand is a flagged zero-row."""
        if features.shape != (FEATURE_DIMENSION,):
            raise ValueError(f"expected ({FEATURE_DIMENSION},) features, got {features.shape}")
        self.buffer.append(np.asarray(features, dtype=np.float32))

    # -- decision ----------------------------------------------------------- #

    def _logits(self, seq: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(seq).unsqueeze(0).to(self.device)  # (1, T, D)
        with torch.no_grad():
            logits = self.model(x)
        return logits.squeeze(0).cpu().numpy()

    def predict(self) -> Prediction:
        """Run one decision step. Returns COMMIT only after stability+cooldown."""
        now = time.time()
        if len(self.buffer) < self.min_frames:
            return Prediction(word=None, confidence=0.0, status="NO_HANDS",
                              timestamp=now)  # not enough motion yet

        seq = np.stack(self.buffer, axis=0)               # (T, 186)
        if seq.shape[0] < self.sequence_length:
            pad = np.zeros((self.sequence_length - seq.shape[0], FEATURE_DIMENSION),
                           dtype=np.float32)
            seq = np.vstack([seq, pad])

        with timed():
            logits = self._logits(seq)
        p = torch.softmax(torch.from_numpy(logits / self.temperature), dim=-1).numpy()
        conf, idx = float(p.max()), int(p.argmax())

        if conf < self.threshold:
            return Prediction(word=None, confidence=conf, status="UNCERTAIN",
                              timestamp=now)  # abstain, never guess

        # Per-class EMA damps single-frame spikes.
        self.ema = (1 - self.alpha) * self.ema + self.alpha * p

        top = int(self.ema.argmax())
        if top == self._prev_class:
            self._stable += 1
        else:
            self._stable = 1
        self._prev_class = top

        if self._stable >= self.stability_window and now >= self._cooldown_until:
            word = self.classes[top]
            self._cooldown_until = now + 1.4
            self._stable = 0
            self.sentence.append(word, float(self.ema[top]))
            return Prediction(word=word, confidence=float(self.ema[top]),
                              status="COMMIT", timestamp=now)

        return Prediction(word=self.classes[top], confidence=float(self.ema[top]),
                          status="STABLE", timestamp=now)

    def reset(self) -> None:
        """Clear the buffer, smoothing state and sentence (but not the model)."""
        self.buffer.clear()
        self.ema[:] = 0.0
        self._prev_class = None
        self._stable = 0
        self._cooldown_until = 0.0
        self.sentence.reset()
