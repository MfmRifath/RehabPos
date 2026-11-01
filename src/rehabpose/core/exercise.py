"""Exercise configuration models (configs/exercises/*.yaml), validated with pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rehabpose.core.angles import MEASURES

View = Literal["sagittal", "frontal"]
Severity = Literal["info", "warn", "error"]

PRIMARY_MEASURES = ("upper_arm_trunk_angle", "elbow_flexion", "forearm_rotation", "reach_ratio")
FRAME_MEASURES = (*MEASURES, "reach_ratio")


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FsmConfig(_Frozen):
    """Hysteresis rep FSM thresholds in units of the primary measure."""

    start_below: float
    peak_above_offset: float = Field(ge=0)
    return_below: float
    min_excursion: float = Field(default=20.0, gt=0)  # peak must reach start_below + this
    peak_drop: float = Field(default=5.0, gt=0)  # drop from running max that ends the peak
    min_rep_duration_s: float = Field(default=0.8, gt=0)

    @model_validator(mode="after")
    def _order(self) -> FsmConfig:
        if self.return_below < self.start_below:
            raise ValueError("return_below must be >= start_below")
        return self


class FrameRule(_Frozen):
    """Per-frame threshold rule on one measure.

    ops: lt / gt (measure vs value), abs_delta_gt (|measure - baseline| > value),
    drop_frac_gt (1 - measure / baseline > value).
    """

    kind: Literal["frame"] = "frame"
    code: str = Field(pattern=r"^[A-Z][A-Z_]+$")
    measure: str
    op: Literal["lt", "gt", "abs_delta_gt", "drop_frac_gt"]
    value: float
    min_duration_s: float = Field(default=0.0, ge=0)
    severity: Severity = "warn"
    unknown_in_views: tuple[View, ...] = ()
    experimental: bool = False

    @property
    def needs_baseline(self) -> bool:
        """True if the rule compares against the calibration baseline."""
        return self.op in ("abs_delta_gt", "drop_frac_gt")

    @model_validator(mode="after")
    def _measure(self) -> FrameRule:
        if self.measure not in FRAME_MEASURES:
            raise ValueError(f"unknown measure {self.measure!r}")
        return self


class RepRule(_Frozen):
    """Rule evaluated once per completed rep (parameters come from the exercise)."""

    kind: Literal["rep"]
    code: Literal["INCOMPLETE_ROM", "TOO_FAST"]
    severity: Severity = "warn"


Rule = Annotated[FrameRule | RepRule, Field(discriminator="kind")]


def min_jerk_rep(peak: float, n: int = 64) -> list[float]:
    """Synthetic reference rep: minimum-jerk rise 0 -> peak then fall back to 0 (n samples)."""
    tau = np.linspace(0.0, 2.0, n)
    s = np.where(tau <= 1.0, tau, 2.0 - tau)
    profile = 10 * s**3 - 15 * s**4 + 6 * s**5
    return [round(float(v), 4) for v in peak * profile]


class ReferenceConfig(_Frozen):
    """Reference rep curve for DTW similarity; synthetic until a recorded rep replaces it."""

    source: Literal["synthetic", "recorded"]
    curve: tuple[float, ...] | None = None
    n: int = Field(default=64, ge=8)

    @model_validator(mode="after")
    def _curve(self) -> ReferenceConfig:
        if self.source == "recorded" and not self.curve:
            raise ValueError("a recorded reference needs a curve")
        return self


class ExerciseConfig(_Frozen):
    """One exercise: primary measure, rep FSM, rules and cue texts (defaults, not clinical)."""

    id: str = Field(pattern=r"^[a-z_]+$")
    name: str
    view: View
    allowed_views: tuple[View, ...] = ()
    experimental: bool = False
    primary_angle: str
    unit: Literal["deg", "ratio"] = "deg"
    target: float
    fsm: FsmConfig
    min_rep_s: float = Field(gt=0)
    max_peak_vel_dps: float = Field(gt=0)
    rom_fraction: float = Field(default=0.9, gt=0, le=1)
    rules: tuple[Rule, ...]
    cue_texts: dict[str, str]
    reference: ReferenceConfig

    @model_validator(mode="after")
    def _check(self) -> ExerciseConfig:
        if self.primary_angle not in PRIMARY_MEASURES:
            raise ValueError(f"primary_angle must be one of {PRIMARY_MEASURES}")
        if self.allowed_views and self.view not in self.allowed_views:
            raise ValueError("view must be in allowed_views")
        missing = [r.code for r in self.rules if r.code not in self.cue_texts]
        if missing:
            raise ValueError(f"cue_texts missing for {missing}")
        if self.fsm.start_below + self.fsm.min_excursion > self.target:
            raise ValueError("target must be >= start_below + min_excursion")
        return self

    @property
    def views(self) -> tuple[View, ...]:
        """Views a session may use for this exercise."""
        return self.allowed_views or (self.view,)

    @property
    def peak_threshold(self) -> float:
        """Primary value above which the FSM enters the peak phase."""
        return self.target - self.fsm.peak_above_offset

    @property
    def frame_rules(self) -> tuple[FrameRule, ...]:
        """Per-frame rules."""
        return tuple(r for r in self.rules if isinstance(r, FrameRule))

    @property
    def rep_rules(self) -> tuple[RepRule, ...]:
        """Per-rep rules."""
        return tuple(r for r in self.rules if isinstance(r, RepRule))

    def reference_curve(self) -> list[float]:
        """Reference rep curve (recorded, or synthetic minimum-jerk peaking at target)."""
        if self.reference.curve:
            return list(self.reference.curve)
        return min_jerk_rep(self.target, self.reference.n)


def load_exercise(path: Path) -> ExerciseConfig:
    """Load and validate one exercise YAML file."""
    with Path(path).open() as fh:
        return ExerciseConfig.model_validate(yaml.safe_load(fh))


def load_exercises(directory: Path) -> dict[str, ExerciseConfig]:
    """Load every *.yaml in `directory`, keyed by exercise id (error if there are none)."""
    if not Path(directory).is_dir() or not any(Path(directory).glob("*.yaml")):
        raise FileNotFoundError(f"no exercise configs (*.yaml) found in {directory}")
    out: dict[str, ExerciseConfig] = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        ex = load_exercise(path)
        if ex.id != path.stem:
            raise ValueError(f"{path}: id {ex.id!r} must match the file name")
        out[ex.id] = ex
    return out


def apply_overrides(
    exercise: ExerciseConfig,
    target: float | None = None,
    min_rep_s: float | None = None,
    thresholds: dict[str, float] | None = None,
) -> ExerciseConfig:
    """Per-user copy of an exercise (therapist settings), re-validated.

    thresholds maps frame-rule codes to new `value`s; unknown codes raise ValueError.
    """
    data = exercise.model_dump()
    if target is not None:
        data["target"] = target
    if min_rep_s is not None:
        data["min_rep_s"] = min_rep_s
    codes = {r["code"] for r in data["rules"] if r["kind"] == "frame"}
    unknown = sorted(set(thresholds or {}) - codes)
    if unknown:
        raise ValueError(f"no frame rule {unknown} in {exercise.id}")
    for rule in data["rules"]:
        if rule["kind"] == "frame" and rule["code"] in (thresholds or {}):
            rule["value"] = float((thresholds or {})[rule["code"]])
    return ExerciseConfig.model_validate(data)
