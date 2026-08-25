"""Training harness — reproducible down to the seed.

AdamW + linear warm-up + cosine annealing, AMP when CUDA is available,
gradient clipping, inverse-frequency class weights, signer-aware splits and
early stopping on validation macro-F1. The best checkpoint is selected by
validation macro-F1 — never by training accuracy.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .evaluate import compute_metrics
from .utils import log, set_seed, write_json


class EarlyStopper:
    """Stop when validation macro-F1 fails to improve for ``patience`` epochs."""

    def __init__(self, patience: int = 12, minimize: bool = False) -> None:
        self.patience = patience
        self.minimize = minimize
        self.best = -math.inf if not minimize else math.inf
        self.counter = 0

    def step(self, metric: float) -> bool:
        improved = metric > self.best if not self.minimize else metric < self.best
        if improved:
            self.best = metric
            self.counter = 0
            return True
        self.counter += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.counter >= self.patience


def _build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    warmup_steps = int(cfg["warmup_epochs"]) * max(1, steps_per_epoch)
    total_steps = int(cfg["epochs"]) * max(1, steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)  # linear warm-up
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine annealing

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler: torch.cuda.amp.GradScaler | None,
    grad_clip: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total_loss / max(1, n)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    classes: Sequence[str],
    cfg: dict[str, Any],
    paths: dict[str, str],
    class_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Run the full training loop; return training history + best metrics."""
    set_seed(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg, len(train_loader))
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    scaler = torch.cuda.amp.GradScaler() if (cfg.get("amp") and device == "cuda") else None
    stopper = EarlyStopper(patience=int(cfg["early_stop_patience"]))

    checkpoints = Path(paths["checkpoints"])
    checkpoints.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_f1 = -math.inf

    for epoch in range(1, int(cfg["epochs"]) + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                 scaler, float(cfg["gradient_clip"]), scheduler)
        val_metrics = compute_metrics(model, val_loader, classes, device)
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "seconds": round(time.time() - t0, 2),
        }
        history.append(epoch_log)
        log.info("epoch %03d | loss %.4f | val macro-F1 %.4f",
                 epoch, train_loss, val_metrics["macro_f1"])

        # Last checkpoint is always written; best is written on improvement.
        torch.save(model.state_dict(), checkpoints / "last_model.pth")
        if stopper.step(val_metrics["macro_f1"]):
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), checkpoints / "best_model.pth")
            log.info("  ↑ new best model (val macro-F1 %.4f)", best_f1)

        if stopper.should_stop:
            log.info("early stopping after %d stagnant epochs", stopper.counter)
            break

    write_json(paths["logs"] + "/training_history.json", history)
    result = {
        "epochs_run": len(history),
        "best_val_macro_f1": best_f1,
        "history": history,
    }
    write_json(paths["logs"] + "/metrics.json", {"training": result})
    return result
