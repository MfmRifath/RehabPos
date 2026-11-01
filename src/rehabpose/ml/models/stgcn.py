"""Small spatio-temporal graph conv net over the 13-joint upper-body graph (causal in time)."""

from __future__ import annotations

import torch
from torch import nn

# edges inside UPPER13 order: 0 nose, 1/2 ears, 3/4 shoulders, 5/6 elbows, 7/8 wrists,
# 9/10 index fingers, 11/12 hips
EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 4),
    (3, 5),
    (5, 7),
    (7, 9),
    (4, 6),
    (6, 8),
    (8, 10),
    (3, 11),
    (4, 12),
    (11, 12),
]


def normalized_adjacency(n: int = 13) -> torch.Tensor:
    """D^-1/2 (A + I) D^-1/2."""
    a = torch.eye(n)
    for i, j in EDGES:
        a[i, j] = a[j, i] = 1.0
    d = a.sum(1).pow(-0.5)
    return d[:, None] * a * d[None, :]


class STBlock(nn.Module):
    """Graph conv over joints followed by a causal temporal conv."""

    def __init__(
        self, c_in: int, c_out: int, kernel: int = 5, dilation: int = 1, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.register_buffer("adj", normalized_adjacency())
        self.gcn = nn.Conv2d(c_in, c_out, 1)
        self.pad = (kernel - 1) * dilation
        self.tcn = nn.Conv2d(c_out, c_out, (kernel, 1), dilation=(dilation, 1))
        self.act = nn.Sequential(nn.BatchNorm2d(c_out), nn.GELU(), nn.Dropout(dropout))
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T, V)
        h = torch.einsum("bctv,vw->bctw", self.gcn(x), self.adj)
        h = self.tcn(nn.functional.pad(h, (0, 0, self.pad, 0)))
        return self.act(h + self.skip(x))


class STGCN(nn.Module):
    """(B, T, F) -> (B, C). Joint coords (+ per-joint visibility) go through the graph;
    the angle/velocity features are concatenated to the pooled graph embedding."""

    def __init__(
        self, n_features: int, n_outputs: int, channels: int = 32, dim: int | None = None
    ) -> None:
        super().__init__()
        self.dim = dim or (2 if n_features == 45 else 3)
        self.n_coord = 13 * self.dim
        self.n_extra = n_features - self.n_coord - 13  # angles + velocities
        # dilations 1, 2, 4, 8 with kernel 5: the last step sees 61 frames (a whole rep window).
        # Two undilated blocks saw only 9 of 64 frames and scored near chance on the rep task.
        self.blocks = nn.Sequential(
            STBlock(self.dim + 1, channels, dilation=1),
            STBlock(channels, channels, dilation=2),
            STBlock(channels, channels, dilation=4),
            STBlock(channels, channels, dilation=8),
        )
        self.head = nn.Linear(channels + self.n_extra, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_steps(x)[:, -1]

    def forward_steps(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, F) -> (B, T, C) per-step logits (causal temporal convs)."""
        b, t, _ = x.shape
        coords = x[..., : self.n_coord].reshape(b, t, 13, self.dim)
        n_ang = self.n_extra // 2
        vis = x[..., self.n_coord + n_ang : self.n_coord + n_ang + 13].unsqueeze(-1)
        extra = torch.cat([x[..., self.n_coord : self.n_coord + n_ang], x[..., -n_ang:]], dim=-1)
        g = torch.cat([coords, vis], dim=-1).permute(0, 3, 1, 2)  # (B, D+1, T, V)
        h = self.blocks(g).mean(-1).transpose(1, 2)  # (B, T, C): mean over joints
        return self.head(torch.cat([h, extra], dim=-1))
