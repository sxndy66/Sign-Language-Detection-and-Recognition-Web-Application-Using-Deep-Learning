"""Model contract tests: shapes, masks, parameter counts."""
from __future__ import annotations

import pytest
import torch

from src.model import (
    AttentionPooling,
    ISLBiLSTM,
    ISLTransformer,
    SequenceModelConfig,
    build_model,
    count_parameters,
)


@pytest.fixture
def cfg() -> SequenceModelConfig:
    return SequenceModelConfig(feature_dimension=186, num_classes=10)


def test_transformer_output_shape(cfg: SequenceModelConfig) -> None:
    model = ISLTransformer(cfg)
    x = torch.randn(2, 60, 186)
    out = model(x)
    assert out.shape == (2, 10)


def test_transformer_rejects_wrong_rank(cfg: SequenceModelConfig) -> None:
    model = ISLTransformer(cfg)
    with pytest.raises(ValueError):
        model(torch.randn(2, 186))


def test_bilstm_matches_transformer_shape(cfg: SequenceModelConfig) -> None:
    x = torch.randn(3, 60, 186)
    assert ISLBiLSTM(cfg)(x).shape == (3, 10)
    assert ISLTransformer(cfg)(x).shape == (3, 10)


def test_attention_pooling_masks_padding() -> None:
    pool = AttentionPooling(8)
    x = torch.randn(2, 5, 8)
    mask = torch.tensor([[False, False, True, True, True],
                         [False, True, True, True, True]])
    out = pool(x, key_padding_mask=mask)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_build_model_unknown_raises(cfg: SequenceModelConfig) -> None:
    with pytest.raises(ValueError):
        build_model("nope", cfg)


def test_count_parameters_positive(cfg: SequenceModelConfig) -> None:
    assert count_parameters(ISLTransformer(cfg)) > 0
