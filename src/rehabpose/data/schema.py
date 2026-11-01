"""Annotation schema v1 for labelled videos (docs/05_data_plan.md §6), pydantic models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rehabpose.core.angles import joint_angle

RepLabel = Literal["correct", "incomplete_rom", "too_fast", "compensated"]
COMPENSATION_CODES = (
    "TRUNK_LEAN",
    "TRUNK_ROTATION",
    "SHOULDER_HIKE",
    "ELBOW_BEND",
    "SHOULDER_DRIFT",
    "ELBOW_AWAY",
    "PLANE_DRIFT",
)
ANGLE_TOLERANCE_DEG = 0.5


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RepAnnotation(_Model):
    """One repetition: frame indices (start <= peak <= end) and a label."""

    idx: int = Field(ge=0)
    start_frame: int = Field(ge=0)
    peak_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    label: RepLabel
    flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _order(self) -> RepAnnotation:
        if not self.start_frame <= self.peak_frame <= self.end_frame:
            raise ValueError("need start_frame <= peak_frame <= end_frame")
        bad = [f for f in self.flags if f not in COMPENSATION_CODES]
        if bad:
            raise ValueError(f"unknown flags {bad}")
        return self


class CompensationSegment(_Model):
    """A frame range showing one compensation code (segments may overlap)."""

    code: str
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)

    @model_validator(mode="after")
    def _check(self) -> CompensationSegment:
        if self.code not in COMPENSATION_CODES:
            raise ValueError(f"unknown compensation code {self.code!r}")
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame < start_frame")
        return self


class AngleAnnotation(_Model):
    """3-click angle: proximal point, vertex, distal point (pixels); angle in degrees."""

    frame: int = Field(ge=0)
    measure: str
    points_px: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    angle_deg: float = Field(ge=0, le=180)

    @staticmethod
    def angle_from_points(
        points: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ) -> float:
        """Angle at the middle point (degrees)."""
        a, b, c = (np.asarray(p, dtype=np.float64) for p in points)
        return float(joint_angle(a, b, c))

    @model_validator(mode="after")
    def _consistent(self) -> AngleAnnotation:
        expected = self.angle_from_points(self.points_px)
        if abs(expected - self.angle_deg) > ANGLE_TOLERANCE_DEG:
            raise ValueError(f"angle_deg {self.angle_deg} != {expected:.2f} from points_px")
        return self


class Annotation(_Model):
    """All labels for one video (schema_version 1)."""

    schema_version: Literal[1] = 1
    video_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    exercise_id: str
    side: Literal["L", "R"]
    view: Literal["sagittal", "frontal"]
    fps: float = Field(gt=0)
    annotator: str = Field(min_length=1)
    scripted_error: str | None = None
    conditions: dict[str, str] = Field(default_factory=dict)
    reps: list[RepAnnotation] = Field(default_factory=list)
    compensation_segments: list[CompensationSegment] = Field(default_factory=list)
    angle_annotations: list[AngleAnnotation] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _reps(self) -> Annotation:
        idx = [r.idx for r in self.reps]
        if idx != list(range(len(self.reps))):
            raise ValueError("rep idx must be 0..n-1 in order")
        for a, b in zip(self.reps, self.reps[1:], strict=False):
            if b.start_frame <= a.end_frame:
                raise ValueError(f"reps {a.idx} and {b.idx} overlap")
        return self


def load_annotation(path: Path) -> Annotation:
    """Read and validate an annotation JSON file."""
    return Annotation.model_validate_json(Path(path).read_text())


def save_annotation(annotation: Annotation, path: Path) -> Path:
    """Validate and write an annotation as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Annotation.model_validate(annotation.model_dump())
    path.write_text(annotation.model_dump_json(indent=2) + "\n")
    return path


def json_schema() -> dict[str, Any]:
    """JSON Schema of the annotation format."""
    return Annotation.model_json_schema()


def schema_json() -> str:
    """JSON Schema as a formatted string."""
    return json.dumps(json_schema(), indent=2)
