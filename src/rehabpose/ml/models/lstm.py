"""LSTM classifier: causal (last hidden state) or bidirectional (offline rep classification)."""

from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """(B, T, F) -> (B, C) logits."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), n_outputs)

    def forward_steps(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, F) -> (B, T, C) per-step logits (causal direction only)."""
        if self.bidirectional:
            raise RuntimeError("a bidirectional LSTM has no causal per-step output")
        out, _ = self.lstm(x)
        return self.head(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        if self.bidirectional:
            h = out.size(-1) // 2
            last = torch.cat([out[:, -1, :h], out[:, 0, h:]], dim=-1)  # forward end + backward end
        else:
            last = out[:, -1]
        return self.head(last)
