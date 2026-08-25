"""Sequence models for ISL recognition.

Primary:   ISLTransformer — temporal transformer + attention pooling.
Baseline:  ISLBiLSTM     — bidirectional LSTM, same pooling/head for fair comparison.

Both consume (B, T, D) landmark sequences and return raw logits (B, C);
softmax + confidence filtering happen in inference/evaluation, never here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class SequenceModelConfig:
    feature_dimension: int
    num_classes: int
    embedding_dim: int = 256
    num_heads: int = 8
    transformer_layers: int = 4
    ff_dim: int = 512
    dropout: float = 0.15
    max_sequence_length: int = 512


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positions — no learned drift on short windows."""

    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, seq_len: int) -> Tensor:
        return self.pe[:, :seq_len]


class AttentionPooling(nn.Module):
    """Learned weighted mean over the time axis — also yields interpretable weights."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        scores = self.score(x).squeeze(-1)              # (B, T)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        return (x * weights).sum(dim=1)                 # (B, dim)


class ISLTransformer(nn.Module):
    def __init__(self, cfg: SequenceModelConfig) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cfg.feature_dimension, cfg.embedding_dim),
            nn.LayerNorm(cfg.embedding_dim),
        )
        self.pos = SinusoidalPositionalEncoding(cfg.embedding_dim, cfg.max_sequence_length)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embedding_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.transformer_layers)
        self.pool = AttentionPooling(cfg.embedding_dim)
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embedding_dim, cfg.num_classes),
        )

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """x: (B, T, D) normalized landmark sequence -> logits (B, C)."""
        if x.dim() != 3:
            raise ValueError(f"expected (B, T, D), got {tuple(x.shape)}")
        mask: Tensor | None = None
        if lengths is not None:
            idx = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            mask = idx >= lengths.unsqueeze(1)          # True = padded step
        h = self.proj(x) + self.pos(x.size(1))
        h = self.encoder(h, src_key_padding_mask=mask)
        h = self.pool(h, key_padding_mask=mask)
        return self.head(h)


class ISLBiLSTM(nn.Module):
    """Baseline with an identical pooling/head so comparisons are apples-to-apples."""

    def __init__(self, cfg: SequenceModelConfig) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            cfg.feature_dimension,
            cfg.embedding_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=cfg.dropout,
        )
        self.pool = AttentionPooling(cfg.embedding_dim)
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embedding_dim, cfg.num_classes),
        )

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        h, _ = self.lstm(x)
        mask: Tensor | None = None
        if lengths is not None:
            idx = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            mask = idx >= lengths.unsqueeze(1)
        return self.head(self.pool(h, key_padding_mask=mask))


def build_model(name: str, cfg: SequenceModelConfig) -> nn.Module:
    if name == "isl_transformer":
        return ISLTransformer(cfg)
    if name == "isl_bilstm":
        return ISLBiLSTM(cfg)
    raise ValueError(f"unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
