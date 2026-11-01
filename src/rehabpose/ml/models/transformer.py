"""Small causal transformer encoder (2-4 layers, d=64) reading the last token."""

from __future__ import annotations

import torch
from torch import nn


class CausalTransformer(nn.Module):
    """(B, T, F) -> (B, C) logits; a causal mask hides future frames."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        d_model: int = 64,
        layers: int = 2,
        heads: int = 4,
        ff: int = 128,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.inp = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, ff, dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_outputs)

    def forward_steps(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, F) -> (B, T, C) per-step logits under the causal mask."""
        t = x.size(1)
        h = self.inp(x) + self.pos[:, :t]
        mask = torch.triu(torch.full((t, t), float("-inf"), device=x.device), diagonal=1)
        # is_causal=True: skips torch's value-based mask check, which ONNX export cannot trace
        h = self.enc(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_steps(x)[:, -1]
