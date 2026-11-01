"""One Euro filter (Casiez et al., CHI 2012), vectorized, plus an offline Savitzky-Golay helper."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

from rehabpose.core.angles import body_scale
from rehabpose.core.landmarks import Frame

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OneEuroParams:
    """One Euro parameters.

    min_cutoff: Hz at rest; beta: cutoff gain per (normalized unit / s) of speed;
    d_cutoff: Hz for the derivative low-pass; reset_gap_s: reset state after a gap > this (s).
    """

    min_cutoff: float = 1.0
    beta: float = 0.5
    d_cutoff: float = 1.0
    reset_gap_s: float = 0.5


def _alpha(cutoff: FloatArray | float, dt: float) -> FloatArray:
    tau = 1.0 / (2.0 * math.pi * np.asarray(cutoff, dtype=np.float64))
    return np.asarray(1.0 / (1.0 + tau / dt), dtype=np.float64)


class OneEuroFilter:
    """Causal One Euro filter over an array of any fixed shape (e.g. joints x axes).

    Speeds are divided by `scale` (e.g. shoulder width) so beta acts in normalized units.
    Non-finite inputs pass through as NaN and leave that element's state untouched.
    """

    def __init__(self, params: OneEuroParams | None = None) -> None:
        self.params = params or OneEuroParams()
        self._x: FloatArray | None = None
        self._dx: FloatArray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        """Forget all state; the next sample passes through unchanged."""
        self._x = self._dx = None
        self._t = None

    def __call__(self, x: FloatArray, t_s: float, scale: float = 1.0) -> FloatArray:
        """Filter sample `x` taken at time `t_s` (seconds); returns the smoothed sample."""
        x = np.asarray(x, dtype=np.float64)
        finite = np.isfinite(x)
        gap = self._t is not None and (t_s - self._t) > self.params.reset_gap_s
        if self._x is None or self._dx is None or self._t is None or gap:
            self._x = np.where(finite, x, np.nan)
            self._dx = np.zeros_like(x)
            self._t = t_s
            return self._x.copy()
        if x.shape != self._x.shape:
            raise ValueError(f"shape changed from {self._x.shape} to {x.shape}")

        dt = t_s - self._t
        if dt <= 0:
            return self._x.copy()
        self._t = t_s

        fresh = finite & ~np.isfinite(self._x)  # first valid sample for an element
        prev = np.where(fresh, x, self._x)
        dx = (x - prev) / dt
        dx_hat = _alpha(self.params.d_cutoff, dt) * dx + (1 - _alpha(self.params.d_cutoff, dt)) * (
            self._dx
        )
        speed = np.abs(dx_hat) / (scale if scale > 0 else 1.0)
        a = _alpha(self.params.min_cutoff + self.params.beta * speed, dt)
        x_hat = a * x + (1 - a) * prev

        self._x = np.where(finite, x_hat, self._x)
        self._dx = np.where(finite, dx_hat, self._dx)
        return np.where(finite, x_hat, np.nan)


class FrameSmoother:
    """One Euro smoothing of a Frame: pixel xy + z and world xyz; visibility is not filtered."""

    def __init__(self, params: OneEuroParams | None = None) -> None:
        self._image = OneEuroFilter(params)
        self._world = OneEuroFilter(params)

    def reset(self) -> None:
        """Reset both image and world filters."""
        self._image.reset()
        self._world.reset()

    def __call__(self, frame: Frame) -> Frame:
        """Return a smoothed copy of `frame`; speeds normalized by body_scale (shoulder widths)."""
        t_s = frame.ts_ms / 1000.0
        size = np.array([frame.img_w, frame.img_h, frame.img_w], dtype=np.float64)
        img = frame.lm[:, :3] * size  # z shares the x scale in MediaPipe
        world = frame.world_xyz
        img_s = self._image(img, t_s, scale=float(body_scale(img[:, :2])))
        world_s = self._world(world, t_s, scale=float(body_scale(world)))

        lm = frame.lm.copy()
        lm[:, :3] = img_s / size
        w = frame.world.copy()
        w[:, :3] = world_s
        return Frame(lm, w, frame.ts_ms, frame.img_w, frame.img_h, frame.frame_idx)


def savgol_smooth(
    x: FloatArray, fps: float, window_s: float = 0.25, polyorder: int = 2, axis: int = 0
) -> FloatArray:
    """Offline (non-causal) Savitzky-Golay smoothing along `axis`; window in seconds.

    Input must be finite; the window is rounded to the nearest odd length >= polyorder + 2
    and clipped to the series length.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[axis]
    win = max(int(round(window_s * fps)) | 1, polyorder + 2 | 1)
    if win > n:
        win = n if n % 2 == 1 else n - 1
    if win <= polyorder:
        return x.copy()
    return np.asarray(savgol_filter(x, win, polyorder, axis=axis), dtype=np.float64)
