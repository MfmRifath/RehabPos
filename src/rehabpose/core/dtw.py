"""Dynamic time warping with a Sakoe-Chiba band, and rep-vs-reference similarity."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def znorm(x: FloatArray, eps: float = 1e-8) -> FloatArray:
    """Zero-mean, unit-variance copy of a 1D curve (all zeros if the curve is flat)."""
    a = np.asarray(x, dtype=np.float64)
    sd = a.std()
    return np.zeros_like(a) if sd < eps else (a - a.mean()) / sd


def resample(x: FloatArray, n: int) -> FloatArray:
    """Linearly resample a 1D curve to n samples."""
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        raise ValueError("cannot resample an empty curve")
    if a.size == 1:
        return np.full(n, a[0])
    return np.interp(np.linspace(0, a.size - 1, n), np.arange(a.size), a)


def dtw_distance(a: FloatArray, b: FloatArray, band: float = 0.1) -> tuple[float, int]:
    """DTW with |a_i - b_j| cost inside a Sakoe-Chiba band.

    band: half-width as a fraction of the longer length (widened to at least |len(a)-len(b)|).
    Returns (total cost along the optimal path, path length).
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    n, m = x.size, y.size
    if n == 0 or m == 0:
        raise ValueError("curves must be non-empty")
    w = max(math.ceil(band * max(n, m)), abs(n - m))
    inf = math.inf
    cost = np.full((n + 1, m + 1), inf)
    steps = np.zeros((n + 1, m + 1), dtype=np.int64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - w), min(m, i + w)
        xi = x[i - 1]
        for j in range(lo, hi + 1):
            # predecessors: diagonal, up, left (ties prefer the diagonal = shorter path)
            c_d, c_u, c_l = cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]
            if c_d <= c_u and c_d <= c_l:
                best, prev = c_d, steps[i - 1, j - 1]
            elif c_u <= c_l:
                best, prev = c_u, steps[i - 1, j]
            else:
                best, prev = c_l, steps[i, j - 1]
            cost[i, j] = abs(xi - y[j - 1]) + best
            steps[i, j] = prev + 1
    return float(cost[n, m]), int(steps[n, m])


def dtw_similarity(
    curve: FloatArray, reference: FloatArray, n: int = 64, band: float = 0.1
) -> float:
    """Similarity in (0, 1]: 1 / (1 + path-normalized DTW distance) of z-normalized curves.

    Both curves are resampled to n samples first, so the score reflects shape, not tempo or
    amplitude.
    """
    a = znorm(resample(curve, n))
    b = znorm(resample(reference, n))
    dist, length = dtw_distance(a, b, band)
    return 1.0 / (1.0 + dist / length)
