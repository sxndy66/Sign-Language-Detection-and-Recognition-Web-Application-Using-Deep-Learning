"""Landmark normalization and 186-D feature extraction.

Feature schema (asserted in tests, never assumed):

    ================  ==============================================  ====
    block             content                                          dims
    ================  ==============================================  ====
    left hand         21 landmarks x xyz, wrist-normalized            63
    right hand        21 landmarks x xyz, wrist-normalized            63
    pose subset       shoulders, elbows, wrists, hips, nose (11x xyz) 33
    face subset       brows / eyes / mouth (12 points x xy)           24
    presence flags    left-hand, right-hand, face visible             3
    ================  ==============================================  ====
    total                                                             186
"""
from __future__ import annotations

from typing import Any

import numpy as np

FEATURE_DIMENSION = 186
HAND_LANDMARKS = 21
POSE_JOINTS = 11
FACE_POINTS = 12

# Presence-flag indices (last three columns of the feature row).
FLAG_LEFT = FEATURE_DIMENSION - 3
FLAG_RIGHT = FEATURE_DIMENSION - 2
FLAG_FACE = FEATURE_DIMENSION - 1


def sanitize(arr: np.ndarray) -> np.ndarray:
    """Replace non-finite coordinates with zeros and log nothing (noisy)."""
    return np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def align_vertical(lm: np.ndarray) -> np.ndarray:
    """Rotate a wrist-centred hand so wrist→middle-MCP points straight up.

    ``lm`` is assumed to already be translated so the wrist sits at the
    origin. The rotation makes signs invariant to the hand's roll angle.
    """
    middle_mcp = lm[9]
    norm = float(np.linalg.norm(middle_mcp))
    if norm < 1e-8:
        return lm  # degenerate: nothing meaningful to align to
    direction = middle_mcp / norm          # unit vector, wrist → middle MCP
    xy = direction[:2]
    xy_norm = float(np.linalg.norm(xy))
    if xy_norm < 1e-8:
        return lm  # already pointing straight out of the plane
    # Orthonormal in-plane rotation that sends `xy` onto the +y axis.
    cos_a, sin_a = xy[1] / xy_norm, xy[0] / xy_norm
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    out = lm.copy()
    out[:, :2] = lm[:, :2] @ rot.T
    return out


def normalize_hand(lm: np.ndarray) -> np.ndarray:
    """Normalize a single hand's 21 landmarks into the canonical sign-frame.

    Translation (wrist at origin), scale (wrist→middle-MCP = unit length) and
    in-plane roll alignment make signs invariant to position, size and tilt.
    """
    lm = sanitize(lm)
    if lm.shape != (HAND_LANDMARKS, 3):
        raise ValueError(f"expected (21, 3) landmarks, got {lm.shape}")
    lm = lm - lm[0]                       # wrist at origin
    scale = float(np.linalg.norm(lm[9])) + 1e-6  # wrist→middle MCP
    lm = lm / scale                       # unit scale
    return align_vertical(lm)             # canonical roll


def _hand_block(landmarks: Any, present: bool) -> np.ndarray:
    """Return a flat 63-vector for one hand, zeros when absent."""
    if not present or landmarks is None:
        return np.zeros(HAND_LANDMARKS * 3, dtype=np.float64)
    pts = np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float64)
    return normalize_hand(pts).ravel()


def _pose_block(landmarks: Any, present: bool) -> np.ndarray:
    """Flat 33-vector: shoulders, elbows, wrists, hips, nose (11 x xyz)."""
    out = np.zeros(POSE_JOINTS * 3, dtype=np.float64)
    if not present or landmarks is None:
        return out
    pts = np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float64)
    # MediaPipe pose topology, 11 upper-body joints:
    # 0 nose, 7/8 ears, 11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips.
    idx = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
    sel = pts[idx] if pts.shape[0] > max(idx) else pts[: POSE_JOINTS]
    if sel.shape[0] < POSE_JOINTS:
        sel = np.vstack([sel, np.zeros((POSE_JOINTS - sel.shape[0], 3))])
    # Centre pose on the nose and scale by shoulder width for invariance.
    if np.linalg.norm(sel[0]) > 1e-6:
        sel = sel - sel[0]
    return sel.ravel()[: POSE_JOINTS * 3]


def _face_block(landmarks: Any, present: bool) -> np.ndarray:
    """Flat 24-vector: brows / eyes / mouth (12 points x xy) — context only."""
    out = np.zeros(FACE_POINTS * 2, dtype=np.float64)
    if not present or landmarks is None:
        return out
    pts = np.array([[p.x, p.y] for p in landmarks.landmark], dtype=np.float64)
    # Subset of the 468-point mesh: brows(70,63,105,66,107), eyes(33,133,362,263),
    # mouth(61,291,13). 12 points total.
    idx = [70, 63, 105, 66, 107, 33, 133, 362, 263, 61, 291, 13]
    if pts.shape[0] <= max(idx):
        return out
    sel = pts[idx]
    sel = sel - sel.mean(axis=0)  # face-centred
    return sel.ravel()


def extract_features(holistic_result: Any, image_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Extract a single (186,) feature row from a MediaPipe Holistic result.

    Missing hands or a missing face insert flagged zero-rows — the model learns
    "no hand here" instead of hallucinating one.
    """
    del image_shape  # reserved for future scale normalization
    left = _hand_block(getattr(holistic_result, "left_hand_landmarks", None),
                       bool(getattr(holistic_result, "left_hand_landmarks", None)))
    right = _hand_block(getattr(holistic_result, "right_hand_landmarks", None),
                        bool(getattr(holistic_result, "right_hand_landmarks", None)))
    pose = _pose_block(getattr(holistic_result, "pose_landmarks", None),
                       bool(getattr(holistic_result, "pose_landmarks", None)))
    face = _face_block(getattr(holistic_result, "face_landmarks", None),
                       bool(getattr(holistic_result, "face_landmarks", None)))

    flags = np.array([
        1.0 if getattr(holistic_result, "left_hand_landmarks", None) else 0.0,
        1.0 if getattr(holistic_result, "right_hand_landmarks", None) else 0.0,
        1.0 if getattr(holistic_result, "face_landmarks", None) else 0.0,
    ], dtype=np.float64)

    row = np.concatenate([left, right, pose, face, flags])
    if row.shape != (FEATURE_DIMENSION,):
        raise AssertionError(f"feature row is {row.shape}, expected ({FEATURE_DIMENSION},)")
    return sanitize(row)


def assert_feature_dimension(dim: int = FEATURE_DIMENSION) -> None:
    """One-line guard used across tests: the 186-D contract is asserted."""
    assert dim == FEATURE_DIMENSION, f"FEATURE_DIM == {dim}, expected {FEATURE_DIMENSION}"
