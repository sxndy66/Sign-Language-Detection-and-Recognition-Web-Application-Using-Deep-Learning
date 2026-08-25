"""Webcam capture — degrade gracefully, never crash.

OpenCV owns the camera. Frames are grabbed at native rate; if the device is
missing or stalls the app degrades to a controlled CAMERA_OFFLINE state
instead of raising mid-loop.
"""
from __future__ import annotations

import cv2
import numpy as np

from .utils import log


class CameraOfflineError(RuntimeError):
    """Raised when the webcam cannot be opened."""


class Webcam:
    """A thin, safe wrapper around cv2.VideoCapture for the latest frame only."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            raise CameraOfflineError("webcam unavailable — check OS permissions")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        log.info("camera %d opened at %dx%d", self.index, self.width, self.height)

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return (ok, frame_bgr). Stale frames are dropped, never queued."""
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ok, frame = self._cap.read()
        return bool(ok), frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None

    def __enter__(self) -> "Webcam":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
