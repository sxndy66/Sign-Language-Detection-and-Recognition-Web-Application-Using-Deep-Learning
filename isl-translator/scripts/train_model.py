"""Train the ISL model from cached landmark sequences.

Usage:
    python scripts/train_model.py --cfg configs/config.yaml --epochs 60
    # → checkpoints/best_model.pth + logs/metrics.json + confusion_matrix.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data_pipeline import LandmarkCache, SampleRecord  # noqa: E402
from src.dataset import SignDataset, SignSample, compute_class_weights, signer_aware_split  # noqa: E402
from src.model import SequenceModelConfig, build_model, count_parameters  # noqa: E402
from src.train import train_model  # noqa: E402
from src.utils import load_config, log  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ISL recognition model.")
    p.add_argument("--cfg", default="configs/config.yaml")
    p.add_argument("--epochs", type=int, default=None, help="override config epochs")
    return p.parse_args()


def load_samples(cache: LandmarkCache) -> list[SignSample]:
    samples = []
    for sample_id, rec in cache._manifest.items():  # noqa: SLF001 — internal access
        if rec.status == "cached" and (Path(cache.cache_dir) / f"{sample_id}.npz").exists():
            samples.append(SignSample(
                sample_id=rec.sample_id, signer_id=rec.signer_id,
                session_id=rec.session_id, label=rec.label,
                sequence_path=rec.sequence_path,
            ))
    return samples


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs

    classes_path = Path(cfg["data"]["classes_file"])
    classes = [l.strip() for l in classes_path.read_text(encoding="utf-8")
               if l.strip() and not l.startswith("#")]
    cfg["model"]["num_classes"] = len(classes)

    cache = LandmarkCache(Path(cfg["data"]["cache_dir"]),
                          Path(cfg["data"]["cache_dir"]) / "manifest.json",
                          version=cfg["data"]["dataset_version"])
    samples = load_samples(cache)
    if not samples:
        log.error("no cached samples — run scripts/extract_landmarks.py first")
        return

    train_s, val_s, test_s = signer_aware_split(
        samples,
        val_ratio=float(cfg["data"]["val_ratio"]),
        test_ratio=float(cfg["data"]["test_ratio"]),
        seed=int(cfg["training"]["seed"]),
    )
    log.info("split: %d train / %d val / %d test (by signer)",
             len(train_s), len(val_s), len(test_s))

    seq_len = int(cfg["data"]["sequence_length"])
    cache_dir = Path(cfg["data"]["cache_dir"])

    def make_loader(ss):
        ds = SignDataset(ss, classes, seq_len, cache_dir)
        return DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]),
                          shuffle=True, num_workers=int(cfg["training"]["num_workers"]),
                          collate_fn=_collate)

    model_cfg = SequenceModelConfig(
        feature_dimension=int(cfg["model"]["feature_dimension"]),
        num_classes=len(classes),
        embedding_dim=int(cfg["model"]["embedding_dim"]),
        num_heads=int(cfg["model"]["num_heads"]),
        transformer_layers=int(cfg["model"]["transformer_layers"]),
        ff_dim=int(cfg["model"]["ff_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    model = build_model(cfg["model"]["name"], model_cfg)
    log.info("%s params: %d", cfg["model"]["name"], count_parameters(model))

    weights = compute_class_weights([s.label for s in train_s], classes) \
        if cfg["data"]["balance_classes"] else None

    train_model(
        model=model,
        train_loader=make_loader(train_s),
        val_loader=make_loader(val_s),
        classes=classes,
        cfg=cfg["training"],
        paths={"checkpoints": cfg["paths"]["checkpoints"], "logs": cfg["paths"]["logs"]},
        class_weights=weights,
    )


def _collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    lens = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return xs, ys, lens


if __name__ == "__main__":
    main()
