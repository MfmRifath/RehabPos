"""Tempo: causal angular velocity and per-rep tempo statistics (degrees, seconds)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

CONTROLLED_TEMPO_DPS = 45.0  # reference for a controlled pace (UCOPhyRehab++ rubric), not a rule


def angular_velocity(values: FloatArray, times_s: FloatArray, lag: int = 1) -> FloatArray:
    """Causal finite difference (x[i] - x[i-lag]) / (t[i] - t[i-lag]), units/s; first lag = 0."""
    x = np.asarray(values, dtype=np.float64)
    t = np.asarray(times_s, dtype=np.float64)
    out = np.zeros_like(x)
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if x.size > lag:
        dt = t[lag:] - t[:-lag]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[lag:] = np.where(dt > 0, (x[lag:] - x[:-lag]) / dt, 0.0)
    return out


class VelocityEstimator:
    """Streaming causal velocity over a short window (default 0.1 s) of (t, value) samples."""

    def __init__(self, window_s: float = 0.1) -> None:
        self.window_s = window_s
        self._buf: deque[tuple[float, float]] = deque()

    def reset(self) -> None:
        """Drop all samples."""
        self._buf.clear()

    def update(self, t_s: float, value: float) -> float:
        """Add a sample; return velocity (units/s) vs. the oldest sample within the window."""
        self._buf.append((t_s, value))
        while len(self._buf) > 2 and t_s - self._buf[1][0] >= self.window_s:
            self._buf.popleft()
        t0, x0 = self._buf[0]
        return (value - x0) / (t_s - t0) if t_s > t0 else 0.0


@dataclass(frozen=True, slots=True)
class RepTempo:
    """Tempo of one rep: duration_s, peak and mean absolute velocity (units/s)."""

    duration_s: float
    peak_vel: float
    mean_abs_vel: float


def rep_tempo(times_s: FloatArray, velocities: FloatArray) -> RepTempo:
    """Tempo statistics from a rep's sample times and (causal) velocities."""
    t = np.asarray(times_s, dtype=np.float64)
    v = np.abs(np.asarray(velocities, dtype=np.float64))
    if t.size == 0:
        return RepTempo(0.0, 0.0, 0.0)
    return RepTempo(float(t[-1] - t[0]), float(v.max()), float(v.mean()))
