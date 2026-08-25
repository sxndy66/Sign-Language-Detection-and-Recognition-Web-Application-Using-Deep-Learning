"""PyTorch dataset + signer-aware splitting + class weighting.

Splits are by signer/session, never by frame: a signer who appears in the
training set never leaks a frame into the validation or test set.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_pipeline import SampleRecord
from .preprocessing import FEATURE_DIMENSION


@dataclass
class SignSample:
    """A dataset item: an id, a label and the (T, 186) sequence path."""

    sample_id: str
    signer_id: str
    session_id: str
    label: str
    sequence_path: str


class SignDataset(Dataset):
    """Maps sample records to padded (T, 186) tensors for training/eval."""

    def __init__(
        self,
        samples: Sequence[SignSample],
        classes: Sequence[str],
        sequence_length: int,
        cache_dir: str | Path,
    ) -> None:
        self.samples = list(samples)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.sequence_length = sequence_length
        self.cache_dir = Path(cache_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, sample: SignSample) -> tuple[np.ndarray, int]:
        path = Path(sample.sequence_path) if sample.sequence_path else self.cache_dir / f"{sample.sample_id}.npz"
        with np.load(path, allow_pickle=False) as data:
            seq = np.asarray(data["sequence"], dtype=np.float32)
        seq = seq[: self.sequence_length]
        return seq, seq.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]
        seq, length = self._load(sample)
        padded = np.zeros((self.sequence_length, FEATURE_DIMENSION), dtype=np.float32)
        padded[:length] = seq
        label = self.class_to_idx[sample.label]
        return torch.from_numpy(padded), torch.tensor(label, dtype=torch.long), length


def signer_aware_split(
    samples: Sequence[SignSample],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[SignSample], list[SignSample], list[SignSample]]:
    """Split by signer so the same person never spans train/val/test."""
    rng = np.random.default_rng(seed)
    signers = sorted({s.signer_id for s in samples})
    rng.shuffle(signers)

    n_test = max(1, round(len(signers) * test_ratio))
    n_val = max(1, round(len(signers) * val_ratio))
    test_signers = set(signers[:n_test])
    val_signers = set(signers[n_test:n_test + n_val])

    train, val, test = [], [], []
    for s in samples:
        if s.signer_id in test_signers:
            test.append(s)
        elif s.signer_id in val_signers:
            val.append(s)
        else:
            train.append(s)
    return train, val, test


def compute_class_weights(
    labels: Sequence[str],
    classes: Sequence[str],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Inverse-frequency class weights for the cross-entropy loss."""
    counts = np.zeros(len(classes), dtype=np.float64)
    idx = {c: i for i, c in enumerate(classes)}
    for lab in labels:
        if lab in idx:
            counts[idx[lab]] += 1.0
    counts += eps
    weights = counts.sum() / (counts * len(classes))
    return torch.tensor(weights, dtype=torch.float32)
