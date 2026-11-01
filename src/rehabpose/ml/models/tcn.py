"""Causal dilated temporal convolution network (left padding only: no future frames)."""

from __future__ import annotations

import torch
from torch import nn


class CausalConv1d(nn.Module):
    """Conv1d that only sees the past (left padding)."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    """Two causal convs with a residual connection."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(c_in, c_out, kernel, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(c_out, c_out, kernel, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.net(x) + self.skip(x))


class TCN(nn.Module):
    """(B, T, F) -> (B, C) logits from the last time step (causal)."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        channels: int = 64,
        levels: int = 4,
        kernel: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks = [
            TCNBlock(n_features if i == 0 else channels, channels, kernel, 2**i, dropout)
            for i in range(levels)
        ]
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, n_outputs)

    def forward_steps(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, F) -> (B, T, C): logits at every step, each from past frames only."""
        h = self.tcn(x.transpose(1, 2))  # (B, C, T)
        return self.head(h.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_steps(x)[:, -1]
