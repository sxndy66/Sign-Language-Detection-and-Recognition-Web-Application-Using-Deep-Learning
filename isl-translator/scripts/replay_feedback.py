"""Replay locally-persisted feedback through the webhook.

A dead webhook never loses a correction because feedback is written to
logs/feedback.jsonl first; this script replays the backlog.

Usage:
    python scripts/replay_feedback.py --log logs/feedback.jsonl \
        --webhook https://hook.make.com/...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feedback import FeedbackEvent, WebhookNotifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay feedback backlog to a webhook.")
    p.add_argument("--log", default="logs/feedback.jsonl")
    p.add_argument("--webhook", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.log)
    if not path.exists():
        print(f"no backlog at {path}")
        return

    notifier = WebhookNotifier(args.webhook)
    sent = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        import json
        data = json.loads(line)
        ev = FeedbackEvent(event=data["event"], payload=data["payload"],
                           timestamp=data.get("timestamp", 0.0))
        if notifier.notify(ev):
            sent += 1
    print(f"replayed {sent} event(s) to {args.webhook}")


if __name__ == "__main__":
    main()
