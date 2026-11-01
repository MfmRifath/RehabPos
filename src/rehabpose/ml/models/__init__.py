"""Temporal models. Input (B, T, F) features, output (B, C) logits."""

from __future__ import annotations

from typing import Any

from torch import nn

from rehabpose.ml.models.lstm import LSTMClassifier
from rehabpose.ml.models.stgcn import STGCN
from rehabpose.ml.models.tcn import TCN
from rehabpose.ml.models.transformer import CausalTransformer

TORCH_MODELS = ("tcn", "lstm", "bilstm", "transformer", "stgcn")
CAUSAL_MODELS = ("tcn", "lstm", "transformer", "stgcn")


def build_model(name: str, n_features: int, n_outputs: int, **kw: Any) -> nn.Module:
    """Construct a torch model by name."""
    if name == "tcn":
        return TCN(n_features, n_outputs, **kw)
    if name in ("lstm", "bilstm"):
        return LSTMClassifier(n_features, n_outputs, bidirectional=name == "bilstm", **kw)
    if name == "transformer":
        return CausalTransformer(n_features, n_outputs, **kw)
    if name == "stgcn":
        return STGCN(n_features, n_outputs, **kw)
    raise ValueError(f"unknown model {name!r}; choose from {TORCH_MODELS} or 'lgbm'")


def n_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
