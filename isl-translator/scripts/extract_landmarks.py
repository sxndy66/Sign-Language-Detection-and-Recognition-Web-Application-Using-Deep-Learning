"""Extract MediaPipe landmarks from raw clips into cached .npz sequences.

Incremental: samples already present in the cache are skipped, and failures are
logged to the manifest rather than crashing the batch.

Usage:
    python scripts/extract_landmarks.py --cfg configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402

from src.data_pipeline import LandmarkCache, SampleRecord  # noqa: E402
from src.preprocessing import extract_features  # noqa: E402
from src.utils import load_config  # noqa: E402


def discover_raw(raw_dir: Path) -> list[tuple[Path, dict]]:
    """Yield (video_path, meta) for every recorded clip with metadata."""
    out = []
    for meta_path in raw_dir.glob("*_meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        signer, session = meta["signer_id"], meta["session_id"]
        for label in meta["labels"]:
            video = raw_dir / f"{signer}_{session}_{label}.avi"
            if video.exists():
                out.append((video, meta, label))
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg)

    raw_dir = Path(cfg["data"]["raw_dir"])
    cache = LandmarkCache(
        cache_dir=Path(cfg["data"]["cache_dir"]),
        manifest_path=Path(cfg["data"]["cache_dir"]) / "manifest.json",
        version=cfg["data"]["dataset_version"],
    )

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    for video, meta, label in discover_raw(raw_dir):
        sample_id = f"{meta['signer_id']}_{meta['session_id']}_{label}"
        if sample_id in cache.cached_ids:
            continue  # incremental — already cached
        record = SampleRecord(
            sample_id=sample_id,
            signer_id=meta["signer_id"],
            session_id=meta["session_id"],
            label=label,
            dataset_version=cfg["data"]["dataset_version"],
        )
        try:
            seq = process_video(video, holistic)
            if seq.shape[0] < int(cfg["data"]["min_frames"]):
                raise ValueError(f"only {seq.shape[0]} valid frames")
            cache.put(sample_id, seq, record)
        except Exception as exc:  # noqa: BLE001 — logged per sample
            cache.mark_failed(sample_id, record, str(exc))

    holistic.close()
    print(f"cached {len(cache.cached_ids)} sequences (version={cfg['data']['dataset_version']})")


def process_video(path: Path, holistic) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = holistic.process(rgb)
        rgb.flags.writeable = True
        rows.append(extract_features(res))
    cap.release()
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 186), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract landmarks to .npz cache.")
    p.add_argument("--cfg", default="configs/config.yaml")
    return p.parse_args()


if __name__ == "__main__":
    main()
