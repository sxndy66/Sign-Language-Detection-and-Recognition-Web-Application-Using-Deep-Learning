"""Feedback loop — user corrections flow back to retraining, with a human in it.

Feedback is ALWAYS persisted locally (logs/feedback.jsonl) before it touches
any webhook, so a dead webhook never loses a correction. Automation can request
training and report results — it can never promote a model.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from .utils import log

# Automation may emit these events; none of them is a promotion.
EVENTS = [
    "prediction_feedback",
    "model_training_requested",
    "dataset_updated",
    "training_completed",
    "model_evaluation_completed",
    "deployment_requested",
]


@dataclass
class FeedbackEvent:
    event: str
    payload: dict
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.event not in EVENTS:
            raise ValueError(f"unknown event {self.event!r}; allowed: {EVENTS}")
        self.timestamp = self.timestamp or time.time()


class FeedbackRecorder:
    """Append-only JSONL store. Local-first by construction."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: FeedbackEvent) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(event), default=str) + "\n")

    def replay(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out


class WebhookNotifier:
    """POSTs events to Make.com / n8n. Failures are logged, never fatal."""

    def __init__(self, url: str | None, timeout: float = 5.0) -> None:
        self.url = (url or "").strip()
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def notify(self, event: FeedbackEvent) -> bool:
        if not self.enabled:
            return False
        try:
            resp = requests.post(self.url, json=asdict(event), timeout=self.timeout)
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            # Local persistence already happened — a dead webhook loses nothing.
            log.warning("webhook failed (%s); event kept in local log", exc)
            return False


def record_feedback(recorder: FeedbackRecorder, notifier: WebhookNotifier,
                    event: str, payload: dict) -> None:
    """Persist locally first, then attempt the webhook."""
    ev = FeedbackEvent(event=event, payload=payload)
    recorder.record(ev)
    notifier.notify(ev)
