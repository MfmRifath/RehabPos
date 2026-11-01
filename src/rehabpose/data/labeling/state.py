"""Pure labelling state (no OpenCV): marks, rep labels, segments, 3-click angles, undo."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Literal, get_args

from rehabpose.data.schema import (
    COMPENSATION_CODES,
    AngleAnnotation,
    Annotation,
    CompensationSegment,
    RepAnnotation,
    RepLabel,
)

REP_LABELS: tuple[RepLabel, ...] = get_args(RepLabel)
# 3-click order: upper_arm_trunk_angle = elbow, shoulder (vertex), hip;
#                elbow_interior = shoulder, elbow (vertex), wrist.
ANGLE_MEASURES = ("upper_arm_trunk_angle", "elbow_interior")

HELP = (
    "space play/pause | a/d -1/+1 | A/D -10/+10 | s start p peak e end | 1-4 label last rep | "
    "k code c segment | n measure, click x3 angle | u undo | w save | q quit"
)


@dataclass
class LabelState:
    """Editable labels for one video; `to_annotation()` validates against schema v1."""

    video_id: str
    subject: str
    exercise_id: str
    side: Literal["L", "R"]
    view: Literal["sagittal", "frontal"]
    fps: float
    n_frames: int
    annotator: str
    frame: int = 0
    playing: bool = False
    marks: dict[str, int] = field(default_factory=dict)
    reps: list[RepAnnotation] = field(default_factory=list)
    segments: list[CompensationSegment] = field(default_factory=list)
    open_segment: tuple[str, int] | None = None
    angles: list[AngleAnnotation] = field(default_factory=list)
    clicks: list[tuple[float, float]] = field(default_factory=list)
    code_idx: int = 0
    measure_idx: int = 0
    message: str = ""
    dirty: bool = False
    _history: list[dict[str, object]] = field(default_factory=list, repr=False)

    # --- navigation ---------------------------------------------------------------------------
    def goto(self, frame: int) -> None:
        """Jump to a frame (clamped)."""
        self.frame = max(0, min(self.n_frames - 1, frame))

    def seek(self, delta: int) -> None:
        """Move by `delta` frames."""
        self.goto(self.frame + delta)

    # --- editing ------------------------------------------------------------------------------
    def _snapshot(self) -> None:
        keys = ("marks", "reps", "segments", "open_segment", "angles", "clicks")
        self._history.append({k: copy.deepcopy(getattr(self, k)) for k in keys})
        self._history = self._history[-200:]
        self.dirty = True

    def undo(self) -> None:
        """Revert the last edit."""
        if not self._history:
            self.message = "nothing to undo"
            return
        for k, v in self._history.pop().items():
            setattr(self, k, v)
        self.message = "undone"

    @property
    def code(self) -> str:
        """Currently selected compensation code."""
        return COMPENSATION_CODES[self.code_idx]

    @property
    def measure(self) -> str:
        """Currently selected angle measure for 3-click annotation."""
        return ANGLE_MEASURES[self.measure_idx]

    def mark(self, kind: Literal["start", "peak", "end"]) -> None:
        """Mark rep start/peak/end at the current frame; `end` completes the rep."""
        self._snapshot()
        self.marks[kind] = self.frame
        self.message = f"{kind} @ {self.frame}"
        if kind != "end":
            return
        if "start" not in self.marks or "peak" not in self.marks:
            self.message = "mark start and peak before end"
            self.marks.pop("end")
            return
        s, p, e = self.marks["start"], self.marks["peak"], self.marks["end"]
        if not s <= p <= e:
            self.message = f"need start <= peak <= end (got {s}, {p}, {e})"
            self.marks.clear()
            return
        if any(not (e < r.start_frame or s > r.end_frame) for r in self.reps):
            self.message = "rep overlaps an existing rep"
            self.marks.clear()
            return
        reps = [
            *self.reps,
            RepAnnotation(idx=0, start_frame=s, peak_frame=p, end_frame=e, label="correct"),
        ]
        reps.sort(key=lambda r: r.start_frame)
        self.reps = [r.model_copy(update={"idx": i}) for i, r in enumerate(reps)]
        self.marks.clear()
        self.message = f"rep {s}-{e} added (label correct)"

    def _last_rep_index(self) -> int | None:
        if not self.reps:
            return None
        containing = [
            i for i, r in enumerate(self.reps) if r.start_frame <= self.frame <= r.end_frame
        ]
        return containing[0] if containing else len(self.reps) - 1

    def label_rep(self, label: RepLabel) -> None:
        """Label the rep under the cursor (or the last rep)."""
        i = self._last_rep_index()
        if i is None:
            self.message = "no rep to label"
            return
        self._snapshot()
        self.reps[i] = self.reps[i].model_copy(update={"label": label})
        self.message = f"rep {i} -> {label}"

    def cycle_code(self) -> None:
        """Select the next compensation code."""
        self.code_idx = (self.code_idx + 1) % len(COMPENSATION_CODES)
        self.message = f"code {self.code}"

    def toggle_segment(self) -> None:
        """Open a compensation segment at the cursor, or close the open one."""
        self._snapshot()
        if self.open_segment is None:
            self.open_segment = (self.code, self.frame)
            self.message = f"{self.code} segment from {self.frame}"
            return
        code, start = self.open_segment
        lo, hi = sorted((start, self.frame))
        self.segments.append(CompensationSegment(code=code, start_frame=lo, end_frame=hi))
        self.open_segment = None
        self.message = f"{code} segment {lo}-{hi}"

    def cycle_measure(self) -> None:
        """Select the next angle measure (resets pending clicks)."""
        self.measure_idx = (self.measure_idx + 1) % len(ANGLE_MEASURES)
        self.clicks = []
        self.message = f"measure {self.measure}"

    def click(self, x: float, y: float) -> None:
        """Add an angle click (proximal, vertex, distal); the 3rd click stores the angle."""
        self._snapshot()
        self.clicks.append((float(x), float(y)))
        if len(self.clicks) < 3:
            self.message = f"click {len(self.clicks)}/3"
            return
        pts = (self.clicks[0], self.clicks[1], self.clicks[2])
        deg = AngleAnnotation.angle_from_points(pts)
        self.clicks = []
        if deg != deg:  # NaN: coincident points
            self.message = "degenerate angle (repeated point); discarded"
            return
        self.angles.append(
            AngleAnnotation(
                frame=self.frame, measure=self.measure, points_px=pts, angle_deg=round(deg, 3)
            )
        )
        self.message = f"{self.measure} = {deg:.1f} deg @ {self.frame}"

    # --- keys ---------------------------------------------------------------------------------
    def handle_key(self, key: int) -> bool:
        """Apply one key press; returns False when the user quits."""
        if key < 0:
            return True
        ch = chr(key & 0xFF)
        actions = {
            " ": lambda: setattr(self, "playing", not self.playing),
            "a": lambda: self.seek(-1),
            "d": lambda: self.seek(1),
            "A": lambda: self.seek(-10),
            "D": lambda: self.seek(10),
            "s": lambda: self.mark("start"),
            "p": lambda: self.mark("peak"),
            "e": lambda: self.mark("end"),
            "k": self.cycle_code,
            "c": self.toggle_segment,
            "n": self.cycle_measure,
            "u": self.undo,
        }
        if ch == "q" or key == 27:
            return False
        if ch in "1234":
            self.label_rep(REP_LABELS[int(ch) - 1])
        elif ch in actions:
            actions[ch]()
        return True

    # --- (de)serialization --------------------------------------------------------------------
    def to_annotation(self, notes: str = "") -> Annotation:
        """Validated Annotation (an open segment is not included)."""
        flags_by_rep = [
            sorted(
                {
                    s.code
                    for s in self.segments
                    if s.start_frame <= r.end_frame and s.end_frame >= r.start_frame
                }
            )
            for r in self.reps
        ]
        reps = [
            r.model_copy(update={"flags": f}) for r, f in zip(self.reps, flags_by_rep, strict=True)
        ]
        return Annotation(
            video_id=self.video_id,
            subject=self.subject,
            exercise_id=self.exercise_id,
            side=self.side,
            view=self.view,
            fps=self.fps,
            annotator=self.annotator,
            reps=reps,
            compensation_segments=list(self.segments),
            angle_annotations=list(self.angles),
            notes=notes,
        )

    @classmethod
    def from_annotation(cls, ann: Annotation, n_frames: int) -> LabelState:
        """Resume editing an existing annotation."""
        return cls(
            video_id=ann.video_id,
            subject=ann.subject,
            exercise_id=ann.exercise_id,
            side=ann.side,
            view=ann.view,
            fps=ann.fps,
            n_frames=n_frames,
            annotator=ann.annotator,
            reps=list(ann.reps),
            segments=list(ann.compensation_segments),
            angles=list(ann.angle_annotations),
        )
