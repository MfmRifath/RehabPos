"""Rule engine: calibration baseline, per-frame and per-rep rules, unknown handling."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Literal

from rehabpose.core.exercise import ExerciseConfig, FrameRule, Severity, View

Source = Literal["rule", "model"]
RepLabel = Literal["correct", "incomplete_rom", "too_fast", "compensated"]
Measures = dict[str, float | None]


@dataclass(frozen=True, slots=True)
class Flag:
    """A raised form/setup flag; confidence in [0, 1]."""

    code: str
    severity: Severity
    confidence: float
    source: Source = "rule"


class Calibrator:
    """Baseline = per-measure median over the first `duration_s` of a set.

    Done once `duration_s` has elapsed and at least one measure has `min_samples` valid
    values, or once `max_duration_s` has elapsed (possibly with an empty baseline).
    """

    def __init__(self, duration_s: float = 1.0, max_duration_s: float = 3.0, min_samples: int = 5):
        self.duration_s = duration_s
        self.max_duration_s = max_duration_s
        self.min_samples = min_samples
        self.reset()

    def reset(self) -> None:
        """Start a new calibration window at the next update."""
        self._t0: float | None = None
        self._samples: dict[str, list[float]] = {}
        self.baseline: dict[str, float] = {}
        self.done = False

    def update(self, t_s: float, measures: Measures) -> None:
        """Collect one frame of measures (ignored once calibration is done)."""
        if self.done:
            return
        if self._t0 is None:
            self._t0 = t_s
        for name, value in measures.items():
            if value is not None:
                self._samples.setdefault(name, []).append(value)
        elapsed = t_s - self._t0
        if elapsed >= self.duration_s:
            base = {k: median(v) for k, v in self._samples.items() if len(v) >= self.min_samples}
            if base or elapsed >= self.max_duration_s:
                self.baseline, self.done = base, True

    @property
    def timed_out(self) -> bool:
        """True if calibration finished without any usable baseline."""
        return self.done and not self.baseline


def _condition(rule: FrameRule, value: float, baseline: float | None) -> bool | None:
    if rule.op == "lt":
        return value < rule.value
    if rule.op == "gt":
        return value > rule.value
    if baseline is None:
        return None
    if rule.op == "abs_delta_gt":
        return abs(value - baseline) > rule.value
    if baseline <= 0:
        return None
    return (1.0 - value / baseline) > rule.value  # drop_frac_gt


class RuleEngine:
    """Evaluates an exercise's YAML rules for one session view."""

    def __init__(
        self, exercise: ExerciseConfig, view: View, include_experimental: bool = False
    ) -> None:
        self.exercise = exercise
        self.view = view
        self._since: dict[str, float] = {}
        self._frame_rules = tuple(
            r for r in exercise.frame_rules if include_experimental or not r.experimental
        )

    def reset(self) -> None:
        """Clear rule persistence timers."""
        self._since.clear()

    def evaluate(
        self,
        t_s: float,
        measures: Measures,
        confidences: dict[str, float],
        baseline: dict[str, float],
    ) -> tuple[list[Flag], dict[str, bool | None]]:
        """Return (active flags, status per rule code: True raised / False clear / None unknown).

        A rule is raised only after its condition holds for `min_duration_s`; unknown inputs
        (invisible landmarks, missing baseline, view in unknown_in_views) never raise a flag.
        Rules marked experimental are skipped unless include_experimental was set.
        """
        flags: list[Flag] = []
        status: dict[str, bool | None] = {}
        for rule in self._frame_rules:
            value = measures.get(rule.measure)
            cond: bool | None = None
            if self.view not in rule.unknown_in_views and value is not None:
                cond = _condition(rule, value, baseline.get(rule.measure))
            if not cond:
                self._since.pop(rule.code, None)
                status[rule.code] = cond
                continue
            start = self._since.setdefault(rule.code, t_s)
            active = t_s - start >= rule.min_duration_s
            status[rule.code] = active
            if active:
                conf = confidences.get(rule.measure, 1.0)
                flags.append(Flag(rule.code, rule.severity, conf))
        return flags, status

    def evaluate_rep(self, peak_value: float, duration_s: float, peak_vel: float) -> list[Flag]:
        """Rep-level rules: INCOMPLETE_ROM (peak < rom_fraction*target), TOO_FAST."""
        ex = self.exercise
        flags: list[Flag] = []
        for rule in ex.rep_rules:
            if rule.code == "INCOMPLETE_ROM":
                hit = peak_value < ex.rom_fraction * ex.target
            else:
                hit = duration_s < ex.min_rep_s or peak_vel > ex.max_peak_vel_dps
            if hit:
                flags.append(Flag(rule.code, rule.severity, 1.0))
        return flags


def label_rep(codes: set[str] | frozenset[str]) -> RepLabel:
    """Rep label by precedence compensated > too_fast > incomplete_rom > correct."""
    if any(c.startswith(("TRUNK_", "SHOULDER_", "ELBOW_", "PLANE_")) for c in codes):
        return "compensated"
    if "TOO_FAST" in codes:
        return "too_fast"
    if "INCOMPLETE_ROM" in codes:
        return "incomplete_rom"
    return "correct"
