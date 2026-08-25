"""Realistic sequence augmentation with semantic-safety guards.

Every operation is toggleable in :class:`AugmentationConfig`, applied per
sequence with a seeded RNG, and constrained so it can never corrupt the
meaning of a sign. Left–right flip is disabled by design: it can invert sign
meaning in ISL.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .preprocessing import HAND_LANDMARKS


@dataclass
class AugmentationConfig:
    """Toggles for every augmentation op. All are SAFE unless marked otherwise."""

    coordinate_jitter: bool = True        # ±2-3 px Gaussian — tracker noise
    jitter_sigma: float = 2.5
    gaussian_noise: bool = True           # small σ on xyz channels only
    noise_sigma: float = 0.02
    scale: bool = True                    # 0.9-1.1x — signer distance varies
    scale_range: tuple[float, float] = (0.9, 1.1)
    translation: bool = True              # shift inside frame bounds
    translation_range: float = 0.05
    in_plane_rotation: bool = True        # ±6° around wrist — camera tilt
    rotation_degrees: float = 6.0
    temporal_jitter: bool = True          # warp sampling indices ±2 frames
    temporal_jitter_frames: int = 2
    frame_drop: bool = True               # drop ≤10%, np.interp fills gaps
    frame_drop_prob: float = 0.10
    speed_variation: bool = True          # 0.85-1.2x resample
    speed_range: tuple[float, float] = (0.85, 1.2)
    landmark_dropout: bool = True         # zero a joint + keep flag — occlusion
    landmark_dropout_prob: float = 0.05
    horizontal_perturbation: bool = False  # CAUTION: small shifts only, never midline
    left_right_flip: bool = False         # NEVER — can invert sign meaning
    seed: int | None = None

    # per-sequence RNG state (not serialized)
    rng: np.random.Generator = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        if self.left_right_flip:
            raise ValueError(
                "left_right_flip is disabled by design: it can invert sign meaning."
            )
        self.rng = np.random.default_rng(self.seed)


def _spatial_noise(seq: np.ndarray, rng: np.random.Generator, cfg: AugmentationConfig) -> np.ndarray:
    out = seq.copy()
    t, d = out.shape
    # Coordinate jitter on all channels except the last three presence flags.
    if cfg.coordinate_jitter:
        out[:, :-3] += rng.normal(0.0, cfg.jitter_sigma, size=(t, d - 3))
    if cfg.gaussian_noise:
        out[:, :-3] += rng.normal(0.0, cfg.noise_sigma, size=(t, d - 3))
    return out


def _scale_translate_rotate(seq: np.ndarray, rng: np.random.Generator, cfg: AugmentationConfig) -> np.ndarray:
    out = seq.copy()
    t = out.shape[0]

    scale = 1.0
    if cfg.scale:
        scale = rng.uniform(*cfg.scale_range)

    tx = ty = 0.0
    if cfg.translation:
        tx, ty = rng.uniform(-cfg.translation_range, cfg.translation_range, size=2)

    angle = 0.0
    if cfg.in_plane_rotation:
        angle = np.deg2rad(rng.uniform(-cfg.rotation_degrees, cfg.rotation_degrees))

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Hands are stored per-block: 63 (left) + 63 (right) + 33 (pose) + 24 (face) + 3 (flags).
    hand_blocks = [(0, 63), (63, 126)]
    for start, end in hand_blocks:
        block = out[:, start:end].reshape(t, -1, 3)
        # Landmark dropout: zero a joint, keep the presence flag untouched.
        if cfg.landmark_dropout:
            mask = rng.random((t, block.shape[1])) < cfg.landmark_dropout_prob
            block[mask] = 0.0
        xy = block[:, :, :2] @ rot.T * scale
        xy += np.array([tx, ty])
        block[:, :, :2] = xy
        out[:, start:end] = block.reshape(t, -1)
    return out


def _temporal_augment(seq: np.ndarray, rng: np.random.Generator, cfg: AugmentationConfig) -> np.ndarray:
    out = seq.copy()
    t = out.shape[0]
    if t < 4:
        return out  # too short to resample meaningfully

    idx = np.arange(t, dtype=np.float64)

    if cfg.temporal_jitter:
        idx = idx + rng.uniform(-cfg.temporal_jitter_frames, cfg.temporal_jitter_frames, size=t)

    if cfg.speed_variation:
        speed = rng.uniform(*cfg.speed_range)
        idx = idx * speed

    # Frame drop + interpolation: drop a random subset, then interpolate back.
    if cfg.frame_drop:
        keep = rng.random(t) >= cfg.frame_drop_prob
        if keep.sum() >= 2:
            kept_idx = idx[keep]
            kept_vals = out[keep]
            out = np.stack([
                np.interp(idx, kept_idx, kept_vals[:, d]) for d in range(out.shape[1])
            ], axis=1)

    # Re-sample onto a uniform grid over the (possibly warped) index axis.
    if idx.min() != idx.max():
        new_idx = np.linspace(idx.min(), idx.max(), t)
        out = np.stack([
            np.interp(new_idx, idx, out[:, d]) for d in range(out.shape[1])
        ], axis=1)
    return out


def augment_sequence(seq: np.ndarray, cfg: AugmentationConfig | None = None) -> np.ndarray:
    """Apply a safe, seeded pipeline of augmentations to a (T, 186) sequence.

    Shape and finiteness are preserved by construction; presence flags in the
    last three columns are never scaled, translated or zeroed by augmentation.
    """
    if cfg is None:
        cfg = AugmentationConfig()
    rng = cfg.rng

    out = np.asarray(seq, dtype=np.float64)
    out = _spatial_noise(out, rng, cfg)
    out = _scale_translate_rotate(out, rng, cfg)
    if cfg.horizontal_perturbation:
        # CAUTION: tiny x-shifts only; amplitude is capped so hands can never
        # cross the body midline and flip the sign's meaning.
        amp = 0.01
        out[:, :-3:3] += rng.uniform(-amp, amp)
    out = _temporal_augment(out, rng, cfg)
    assert out.shape == seq.shape, "augmentation must preserve shape"
    assert np.isfinite(out).all(), "augmentation must preserve finiteness"
    return out
