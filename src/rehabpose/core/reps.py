"""Hysteresis rep state machine: rest -> ascending -> peak -> descending -> rest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rehabpose.core.exercise import FsmConfig

Phase = Literal["rest", "ascending", "peak", "descending"]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RepCandidate:
    """A finished movement that passed validation (times in s, values in primary units)."""

    t_start: float
    t_peak: float
    t_end: float
    peak_value: float
    reached_peak: bool
    times: FloatArray
    values: FloatArray
    velocities: FloatArray

    @property
    def duration_s(self) -> float:
        """t_end - t_start in seconds."""
        return self.t_end - self.t_start


class RepCounter:
    """Counts reps on a smoothed primary measure using YAML FSM thresholds.

    A movement becomes a rep when it returns below `return_below` after peaking at least
    `start_below + min_excursion` and lasting >= `min_rep_duration_s`. Reps that never pass
    `peak_threshold` are still returned with reached_peak=False (-> INCOMPLETE_ROM).
    Unknown values hold the state; more than `unknown_reset_s` of unknowns resets to rest.
    After a reset (and at start) the counter is disarmed until the value drops below
    `start_below`, so a rep seen only from its middle is never counted.
    """

    def __init__(self, fsm: FsmConfig, peak_threshold: float, unknown_reset_s: float = 0.5) -> None:
        self.fsm = fsm
        self.peak_threshold = peak_threshold
        self.unknown_reset_s = unknown_reset_s
        self.phase: Phase = "rest"
        self._unknown_since: float | None = None
        self._armed = False
        self._clear()

    def _clear(self) -> None:
        self._t: list[float] = []
        self._x: list[float] = []
        self._v: list[float] = []
        self._max = -np.inf
        self._t_max = 0.0
        self._min_desc = np.inf

    def reset(self) -> None:
        """Abandon any rep in progress and return to rest."""
        self.phase = "rest"
        self._unknown_since = None
        self._armed = False
        self._clear()

    def _begin(self, t: float, x: float, v: float) -> None:
        self._clear()
        self._push(t, x, v)
        self.phase = "peak" if x >= self.peak_threshold else "ascending"

    def _push(self, t: float, x: float, v: float) -> None:
        self._t.append(t)
        self._x.append(x)
        self._v.append(v)
        if x > self._max:
            self._max, self._t_max = x, t

    def _finish(self) -> RepCandidate | None:
        f = self.fsm
        cand = RepCandidate(
            t_start=self._t[0],
            t_peak=self._t_max,
            t_end=self._t[-1],
            peak_value=float(self._max),
            reached_peak=bool(self._max >= self.peak_threshold),
            times=np.array(self._t),
            values=np.array(self._x),
            velocities=np.array(self._v),
        )
        self.phase = "rest"
        self._armed = True
        self._clear()
        valid = (
            cand.peak_value >= f.start_below + f.min_excursion
            and cand.duration_s >= f.min_rep_duration_s
        )
        return cand if valid else None

    def update(self, t_s: float, value: float | None, velocity: float = 0.0) -> RepCandidate | None:
        """Feed one sample (time s, primary value or None); returns a rep when one completes."""
        f = self.fsm
        if value is None:
            if self._unknown_since is None:
                self._unknown_since = t_s
            elif t_s - self._unknown_since > self.unknown_reset_s and self.phase != "rest":
                self.reset()
            return None
        self._unknown_since = None

        if self.phase == "rest":
            if value <= f.start_below:
                self._armed = True
            elif self._armed:
                self._begin(t_s, value, velocity)
            return None

        if self.phase == "ascending" and value <= self._x[0]:
            self._begin(t_s, value, velocity)  # re-anchor the start at the lowest point
            if value <= f.start_below:
                self.phase = "rest"
                self._clear()
            return None

        self._push(t_s, value, velocity)
        if self.phase == "ascending":
            if value >= self.peak_threshold:
                self.phase = "peak"
            elif value < self._max - f.peak_drop:
                self.phase = "descending"
                self._min_desc = value
        elif self.phase == "peak":
            if value < self._max - f.peak_drop:
                self.phase = "descending"
                self._min_desc = value
        elif self.phase == "descending":
            self._min_desc = min(self._min_desc, value)
            if value > self._min_desc + f.peak_drop and value >= f.return_below:
                self.phase = "peak" if self._max >= self.peak_threshold else "ascending"

        if self.phase in ("peak", "descending") and value < f.return_below:
            return self._finish()
        return None
