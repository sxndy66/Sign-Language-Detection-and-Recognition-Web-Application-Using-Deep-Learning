"""Incremental, failure-logged caching pipeline for landmark sequences.

Raw sign videos become MediaPipe landmark arrays (.npz) exactly once; every
subsequent run reuses the cache. Each cached artefact is tagged with its
dataset version so a mixed-language model can never be trained by accident.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .utils import log, sanitize_array


@dataclass
class SampleRecord:
    """One labelled sample: where it came from and how it was cached."""

    sample_id: str
    signer_id: str
    session_id: str
    label: str
    dataset_version: str
    sequence_path: str = ""
    status: str = "pending"          # pending | cached | failed
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "signer_id": self.signer_id,
            "session_id": self.session_id,
            "label": self.label,
            "dataset_version": self.dataset_version,
            "sequence_path": self.sequence_path,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class LandmarkCache:
    """Filesystem cache for (T, 186) landmark sequences keyed by sample id."""

    cache_dir: Path
    manifest_path: Path
    version: str = "unknown"
    _manifest: dict[str, SampleRecord] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.manifest_path = Path(self.manifest_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_manifest()

    # -- manifest ----------------------------------------------------------- #

    def _load_manifest(self) -> None:
        if self.manifest_path.exists():
            try:
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                self._manifest = {
                    k: SampleRecord(**v) for k, v in data.get("samples", {}).items()
                }
            except (json.JSONDecodeError, TypeError) as exc:  # pragma: no cover
                log.warning("corrupt cache manifest %s: %s", self.manifest_path, exc)
                self._manifest = {}

    def _save_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "samples": {k: v.to_dict() for k, v in self._manifest.items()},
        }
        self.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- IO ----------------------------------------------------------------- #

    def _path(self, sample_id: str) -> Path:
        return self.cache_dir / f"{sample_id}.npz"

    def put(self, sample_id: str, sequence: np.ndarray, record: SampleRecord) -> None:
        """Cache a (T, 186) sequence and mark the record as cached."""
        sequence = sanitize_array(sequence)
        np.savez_compressed(self._path(sample_id), sequence=sequence, version=self.version)
        record.sequence_path = str(self._path(sample_id))
        record.status = "cached"
        record.dataset_version = self.version
        record.error = None
        self._manifest[sample_id] = record
        self._save_manifest()

    def get(self, sample_id: str) -> np.ndarray | None:
        """Return the cached sequence, or None when absent."""
        path = self._path(sample_id)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            return sanitize_array(data["sequence"])

    def mark_failed(self, sample_id: str, record: SampleRecord, error: str) -> None:
        """Record a failure without crashing the batch — logged, never silent."""
        record.status = "failed"
        record.error = error
        self._manifest[sample_id] = record
        self._save_manifest()
        log.error("sample %s failed: %s", sample_id, error)

    @property
    def cached_ids(self) -> set[str]:
        return {
            k for k, v in self._manifest.items()
            if v.status == "cached" and self._path(k).exists()
        }
