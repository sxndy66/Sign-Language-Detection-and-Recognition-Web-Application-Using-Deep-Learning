"""Data-pipeline tests: feature schema, normalization invariance, caching."""
from __future__ import annotations

import numpy as np
import pytest

from src.data_pipeline import LandmarkCache, SampleRecord
from src.preprocessing import (
    FEATURE_DIMENSION,
    assert_feature_dimension,
    extract_features,
    normalize_hand,
)
from src.augmentation import AugmentationConfig, augment_sequence


def test_feature_dimension_contract() -> None:
    assert_feature_dimension()


def test_normalize_hand_wrist_at_origin_and_unit_scale() -> None:
    lm = np.random.default_rng(0).normal(size=(21, 3))
    lm[:, 0] += 50  # arbitrary translation
    lm *= 3         # arbitrary scale
    out = normalize_hand(lm)
    assert np.linalg.norm(out[0]) < 1e-8          # wrist at origin
    assert np.linalg.norm(out[9]) == pytest.approx(1.0, abs=1e-4)  # unit scale


class _FakeLandmarks:
    def __init__(self, pts):
        self.landmark = pts


class _FakeResult:
    """Minimal stand-in for a MediaPipe Holistic result."""

    def __init__(self, left=None, right=None, pose=None, face=None):
        self.left_hand_landmarks = left
        self.right_hand_landmarks = right
        self.pose_landmarks = pose
        self.face_landmarks = face


def _fake_hand():
    pts = []
    for i in range(21):
        pts.append(type("P", (), {"x": float(i), "y": float(i % 3), "z": float(i % 2)})())
    return _FakeLandmarks(pts)


def _fake_pose():
    pts = [type("P", (), {"x": 0.5, "y": 0.5, "z": 0.0})() for _ in range(33)]
    return _FakeLandmarks(pts)


def _fake_face():
    pts = [type("P", (), {"x": 0.5, "y": 0.5})() for _ in range(468)]
    return _FakeLandmarks(pts)


def test_extract_features_shape_and_flags() -> None:
    res = _FakeResult(left=_fake_hand(), right=None, pose=_fake_pose(), face=_fake_face())
    row = extract_features(res)
    assert row.shape == (FEATURE_DIMENSION,)
    assert row[-3] == 1.0  # left hand present
    assert row[-2] == 0.0  # right hand absent
    assert row[-1] == 1.0  # face present
    assert np.isfinite(row).all()


def test_augmentation_preserves_shape_and_finiteness() -> None:
    rng = np.random.default_rng(1)
    seq = rng.normal(size=(60, FEATURE_DIMENSION))
    seq[:, -3:] = 1.0  # presence flags
    out = augment_sequence(seq, AugmentationConfig(seed=7))
    assert out.shape == seq.shape
    assert np.isfinite(out).all()


def test_augmentation_forbids_flip() -> None:
    with pytest.raises(ValueError):
        AugmentationConfig(left_right_flip=True)


def test_landmark_cache_roundtrip(tmp_path) -> None:
    cache = LandmarkCache(tmp_path / "cache", tmp_path / "manifest.json", version="v1")
    rec = SampleRecord(sample_id="s1", signer_id="a", session_id="s", label="HELLO",
                       dataset_version="v1")
    seq = np.ones((30, FEATURE_DIMENSION), dtype=np.float32)
    cache.put("s1", seq, rec)
    assert "s1" in cache.cached_ids
    loaded = cache.get("s1")
    assert loaded is not None and loaded.shape == seq.shape
