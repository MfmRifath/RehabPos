"""Augmentations and resampling on canonical upper-body points (T, 13, D) + visibility (T, 13)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rehabpose.core.features import _SWAP

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AugmentConfig:
    """Training-time augmentation strengths (all in canonical, shoulder-width units)."""

    jitter: float = 0.01  # gaussian sd per coordinate
    scale: float = 0.10  # uniform +-
    rotate_deg: float = 10.0  # in-plane rotation +-
    time_warp: float = 0.20  # max relative shift of the middle knot
    joint_dropout: float = 0.05  # per-joint probability for the whole window
    mirror: float = 0.0  # probability of mirror + L/R swap


def resample_time(x: FloatArray, n: int) -> FloatArray:
    """Linear resampling along axis 0 to n steps."""
    a = np.asarray(x, dtype=np.float64)
    t = a.shape[0]
    if t == n:
        return a.copy()
    src = np.linspace(0, t - 1, n)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (src - lo).reshape(-1, *([1] * (a.ndim - 1)))
    return np.asarray(a[lo] * (1 - w) + a[hi] * w, dtype=np.float64)


def mirror(p: FloatArray, v: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Flip x and swap left/right joints (an involution)."""
    q = np.asarray(p, dtype=np.float64).copy()
    q[..., 0] *= -1.0
    return q[:, _SWAP], np.asarray(v)[:, _SWAP].copy()


def time_warp(
    p: FloatArray, v: FloatArray, amount: float, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """Piecewise-linear warp keeping length and endpoints; the middle knot moves by +-amount."""
    t = p.shape[0]
    if t < 4 or amount <= 0:
        return p, v
    mid = (t - 1) / 2
    knot = mid + rng.uniform(-amount, amount) * mid
    src = np.interp(np.arange(t), [0, mid, t - 1], [0, knot, t - 1])
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (src - lo)[:, None, None]
    p2 = p[lo] * (1 - w) + p[hi] * w
    v2 = np.minimum(v[lo], v[hi])
    return p2, v2


def augment(
    p: FloatArray, v: FloatArray, rng: np.random.Generator, cfg: AugmentConfig
) -> tuple[FloatArray, FloatArray]:
    """Random augmentation of one window; invisible joints stay zero."""
    q = np.asarray(p, dtype=np.float64).copy()
    w = np.asarray(v, dtype=np.float64).copy()
    if cfg.mirror and rng.random() < cfg.mirror:
        q, w = mirror(q, w)
    if cfg.time_warp:
        q, w = time_warp(q, w, cfg.time_warp, rng)
    if cfg.rotate_deg:
        th = np.radians(rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
        c, s = np.cos(th), np.sin(th)
        x, y = q[..., 0].copy(), q[..., 1].copy()
        q[..., 0], q[..., 1] = c * x - s * y, s * x + c * y
    if cfg.scale:
        q *= rng.uniform(1 - cfg.scale, 1 + cfg.scale)
    if cfg.jitter:
        q += rng.normal(0.0, cfg.jitter, q.shape)
    if cfg.joint_dropout:
        drop = rng.random(q.shape[1]) < cfg.joint_dropout
        w[:, drop] = 0.0
    visible = w > 0
    return np.where(visible[..., None], q, 0.0), w


def augment_batch(
    p: FloatArray, v: FloatArray, rng: np.random.Generator, cfg: AugmentConfig
) -> tuple[FloatArray, FloatArray]:
    """Vectorized `augment` for equal-length windows: (N, T, 13, D), (N, T, 13)."""
    q = np.asarray(p, dtype=np.float64).copy()
    w = np.asarray(v, dtype=np.float64).copy()
    n, t = q.shape[:2]
    if cfg.mirror:
        flip = rng.random((n,)) < cfg.mirror
        if flip.any():
            q[flip, ..., 0] *= -1.0
            q[flip] = q[flip][:, :, _SWAP]
            w[flip] = w[flip][:, :, _SWAP]
    if cfg.time_warp and t >= 4:
        mid = (t - 1) / 2
        knots = mid + rng.uniform(-cfg.time_warp, cfg.time_warp, (n,)) * mid
        grid = np.arange(t, dtype=np.float64)
        src = np.where(
            grid <= mid,
            grid / mid * knots[:, None],
            knots[:, None] + (grid - mid) / mid * (t - 1 - knots[:, None]),
        )
        lo = np.floor(src).astype(int)
        hi = np.minimum(lo + 1, t - 1)
        frac = (src - lo)[..., None, None]
        rows = np.arange(n)[:, None]
        q = q[rows, lo] * (1 - frac) + q[rows, hi] * frac
        w = np.minimum(w[rows, lo], w[rows, hi])
    if cfg.rotate_deg:
        th = np.radians(rng.uniform(-cfg.rotate_deg, cfg.rotate_deg, (n,)))[:, None, None]
        c, s = np.cos(th), np.sin(th)
        x, y = q[..., 0].copy(), q[..., 1].copy()
        q[..., 0], q[..., 1] = c * x - s * y, s * x + c * y
    if cfg.scale:
        q *= rng.uniform(1 - cfg.scale, 1 + cfg.scale, (n,))[:, None, None, None]
    if cfg.jitter:
        q += rng.normal(0.0, cfg.jitter, q.shape)
    if cfg.joint_dropout:
        drop = rng.random((n, q.shape[2])) < cfg.joint_dropout
        w = np.where(drop[:, None, :], 0.0, w)
    visible = w > 0
    return np.where(visible[..., None], q, 0.0), w
