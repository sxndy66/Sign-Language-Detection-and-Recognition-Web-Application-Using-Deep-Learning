"""Lightweight visualization: landmarks, confidence bars and overlays."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm
]


def draw_hand(frame: np.ndarray, landmarks: Any, color: tuple[int, int, int] = (0, 255, 0)) -> None:
    """Draw MediaPipe hand landmarks + skeleton onto a BGR frame (in place)."""
    if landmarks is None:
        return
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks.landmark]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, color, -1, cv2.LINE_AA)


def draw_prediction(frame: np.ndarray, text: str, confidence: float,
                    color: tuple[int, int, int] = (255, 255, 255)) -> None:
    """Overlay the current prediction + confidence bar in the top-left corner."""
    label = f"{text}  {confidence:.2f}"
    cv2.rectangle(frame, (10, 10), (10 + 260, 64), (0, 0, 0), -1)
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2, cv2.LINE_AA)
    bar_w = int(240 * max(0.0, min(1.0, confidence)))
    cv2.rectangle(frame, (20, 50), (20 + 240, 58), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, 50), (20 + bar_w, 58), (0, 200, 0), -1)


def draw_sentence(frame: np.ndarray, sentence: str) -> None:
    """Draw the committed sentence along the bottom edge."""
    cv2.putText(frame, sentence or "—", (10, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
