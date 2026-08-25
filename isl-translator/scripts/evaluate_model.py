"""Evaluate a trained checkpoint on the held-out test split.

Usage:
    python scripts/evaluate_model.py --split test --checkpoint checkpoints/best_model.pth
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data_pipeline import LandmarkCache  # noqa: E402
from src.dataset import SignDataset, SignSample, signer_aware_split  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.model import SequenceModelConfig, build_model  # noqa: E402
from src.utils import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on the test split.")
    p.add_argument("--cfg", default="configs/config.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    return p.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg)

    classes_path = Path(cfg["data"]["classes_file"])
    classes = [l.strip() for l in classes_path.read_text(encoding="utf-8")
               if l.strip() and not l.startswith("#")]

    cache = LandmarkCache(Path(cfg["data"]["cache_dir"]),
                          Path(cfg["data"]["cache_dir"]) / "manifest.json",
                          version=cfg["data"]["dataset_version"])
    samples = []
    for sid, rec in cache._manifest.items():  # noqa: SLF001
        if rec.status == "cached":
            samples.append(SignSample(sid, rec.signer_id, rec.session_id,
                                      rec.label, rec.sequence_path))

    _, val_s, test_s = signer_aware_split(samples, seed=int(cfg["training"]["seed"]))
    chosen = val_s if args.split == "val" else test_s

    seq_len = int(cfg["data"]["sequence_length"])
    cache_dir = Path(cfg["data"]["cache_dir"])
    ds = SignDataset(chosen, classes, seq_len, cache_dir)
    loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)

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
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    evaluate(model, loader, classes,
             out_dir=cfg["paths"]["logs"],
             dataset_version=cfg["data"]["dataset_version"],
             model_version=args.checkpoint,
             git_commit=git_commit())


if __name__ == "__main__":
    main()
