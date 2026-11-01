"""Stateful per-session engine: Frame -> FeedbackFrame (+ RepRecord when a rep completes)."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np

from rehabpose.core import angles
from rehabpose.core.dtw import dtw_similarity
from rehabpose.core.exercise import ExerciseConfig, View
from rehabpose.core.feedback import Cue, CueSelector
from rehabpose.core.filters import FrameSmoother, OneEuroParams
from rehabpose.core.landmarks import Frame, Side, side_joints
from rehabpose.core.reps import Phase, RepCandidate, RepCounter
from rehabpose.core.rules import Calibrator, Flag, RepLabel, RuleEngine, label_rep
from rehabpose.core.tempo import VelocityEstimator

DISPLAY_MEASURES = ("upper_arm_trunk_angle", "elbow_flexion", "trunk_tilt")
CALIBRATION_MEASURES = ("trunk_tilt", "hike_ratio", "shoulder_width")


@dataclass(frozen=True)
class EngineConfig:
    """Engine parameters (seconds unless noted); thresholds live in the exercise YAML."""

    smoothing: bool = True
    experimental_rules: bool = False  # e.g. PLANE_DRIFT (monocular depth)
    one_euro: OneEuroParams = field(default_factory=OneEuroParams)
    space: angles.Space = "2d"
    visibility_threshold: float = angles.VISIBILITY_THRESHOLD
    calibration_s: float = 1.0
    max_calibration_s: float = 3.0
    unknown_reset_s: float = 0.5
    setup_persist_s: float = 0.5
    view_persist_s: float = 1.0
    rep_flag_hold_s: float = 2.0
    debounce_s: float = 0.5
    speak_interval_s: float = 3.0
    velocity_window_s: float = 0.1
    frontal_ratio: float = 0.45  # shoulder width / torso length above this -> frontal
    sagittal_ratio: float = 0.25  # below this -> sagittal


@dataclass(frozen=True, slots=True)
class RepRecord:
    """One validated repetition (times s, angles deg or primary units, velocity units/s)."""

    idx: int
    t_start: float
    t_peak: float
    t_end: float
    peak_angle: float
    duration_s: float
    peak_vel_dps: float
    flags: tuple[str, ...]
    label: RepLabel
    dtw_similarity: float

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (flags as a list)."""
        d = asdict(self)
        d["flags"] = list(self.flags)
        return d


@dataclass(frozen=True, slots=True)
class Event:
    """A flag raised (rising edge) at time t (s)."""

    t: float
    code: str
    severity: str
    confidence: float
    source: str


@dataclass(frozen=True, slots=True)
class FeedbackFrame:
    """Engine output for one input frame."""

    frame_idx: int
    ts_ms: int
    angles: dict[str, float | None]
    phase: Phase
    rep_count: int
    last_rep: RepRecord | None
    completed_rep: RepRecord | None
    flags: tuple[Flag, ...]
    cue: Cue | None
    visibility_ok: bool
    view_ok: bool
    calibrated: bool
    proc_ms: float


def detect_view(points: angles.FloatArray, cfg: EngineConfig) -> View | None:
    """Frontal/sagittal from the 2D shoulder-width / torso-length ratio; None if ambiguous."""
    sw = float(angles.shoulder_width(points))
    torso = float(
        np.linalg.norm(
            angles.mid_point(points, 11, 12) - angles.mid_point(points, 23, 24),
        )
    )
    if torso <= 0:
        return None
    ratio = sw / torso
    if ratio >= cfg.frontal_ratio:
        return "frontal"
    if ratio <= cfg.sagittal_ratio:
        return "sagittal"
    return None


class FlagModel(Protocol):
    """Optional learned flagger (implemented in rehabpose.ml.online; core stays torch-free)."""

    def update(self, t_s: float, frame: Frame | None) -> list[Flag]:
        """Feed one raw (unsmoothed) frame; return currently raised model flags."""
        ...

    def reset(self) -> None:
        """Forget streaming state."""
        ...


class SessionEngine:
    """Processes frames of one session (exercise + side + camera view) in time order.

    With a `flag_model`, model flags are fused with rule flags by OR: a code raised by either
    source is active (rule flags win when both raise the same code).
    """

    def __init__(
        self,
        exercise: ExerciseConfig,
        side: Side,
        view: View | None = None,
        config: EngineConfig | None = None,
        auto_start: bool = True,
        flag_model: FlagModel | None = None,
    ) -> None:
        self.exercise = exercise
        self.side = side
        side_joints(side)  # validate
        self.view: View = view or exercise.view
        if self.view not in exercise.views:
            raise ValueError(f"{exercise.id} does not support the {self.view} view")
        self.cfg = config or EngineConfig()
        self.flag_model = flag_model
        self.reps: list[RepRecord] = []
        self.events: list[Event] = []
        self._reference = np.asarray(exercise.reference_curve())
        self._measure_names = tuple(
            dict.fromkeys(
                [
                    *(m for m in (exercise.primary_angle,) if m != "reach_ratio"),
                    *(r.measure for r in exercise.frame_rules if r.measure != "reach_ratio"),
                    *DISPLAY_MEASURES,
                    *CALIBRATION_MEASURES,
                ]
            )
        )
        self._smoother = FrameSmoother(self.cfg.one_euro)
        self._counter = RepCounter(exercise.fsm, exercise.peak_threshold, self.cfg.unknown_reset_s)
        self._rules = RuleEngine(exercise, self.view, self.cfg.experimental_rules)
        self._calib = Calibrator(self.cfg.calibration_s, self.cfg.max_calibration_s)
        self._velocity = VelocityEstimator(self.cfg.velocity_window_s)
        self._cues = CueSelector(self.cfg.debounce_s, self.cfg.speak_interval_s)
        self.active = False
        self._prev_codes: set[str] = set()
        if auto_start:
            self.start_set()

    # --- control ------------------------------------------------------------------------------
    def start_set(self) -> None:
        """Begin a new set: reset filters, FSM, calibration and cue state (reps accumulate)."""
        self._smoother.reset()
        self._counter.reset()
        self._rules.reset()
        self._calib.reset()
        self._velocity.reset()
        self._cues.reset()
        if self.flag_model is not None:
            self.flag_model.reset()
        self._rep_codes: set[str] = set()
        self._rep_flags: list[Flag] = []
        self._rep_flags_until = -np.inf
        self._invisible_since: float | None = None
        self._view_bad_since: float | None = None
        self._prev_codes = set()
        self.active = True

    def pause(self) -> None:
        """Stop counting; a rep in progress is discarded."""
        self._counter.reset()
        self.active = False

    def end_set(self) -> None:
        """End the current set."""
        self.pause()

    def calibrate(self) -> None:
        """Restart baseline calibration from the next frame."""
        self._calib.reset()

    # --- processing ---------------------------------------------------------------------------
    def _measures(self, frame: Frame) -> tuple[dict[str, float | None], dict[str, float]]:
        cfg = self.cfg
        m = angles.compute_measures(
            frame, self.side, cfg.space, cfg.visibility_threshold, self._measure_names
        )
        j = side_joints(self.side)
        if angles.is_visible(frame.visibility, (j.wrist,), cfg.visibility_threshold):
            m["wrist_x"] = float(frame.pixel_xy[j.wrist, 0])
        if self.exercise.primary_angle == "reach_ratio":
            rest = self._calib.baseline.get("wrist_x")
            ok = angles.is_visible(
                frame.visibility, angles.required_landmarks("reach_ratio", self.side)
            )
            m["reach_ratio"] = (
                float(angles.reach_ratio(frame.pixel_xy, self.side, np.array([rest, 0.0])))
                if ok and rest is not None
                else None
            )
        near = (j.shoulder, j.hip)
        side_on = self.view == "sagittal"
        if side_on and angles.is_visible(frame.visibility, near, cfg.visibility_threshold):
            pts = frame.pixel_xy if cfg.space == "2d" else frame.world_xyz
            m["trunk_tilt"] = float(angles.trunk_tilt_side(pts, self.side))
        vis = frame.visibility
        conf = {
            name: float(vis[list(angles.required_landmarks(name, self.side))].min())
            for name in m
            if name != "wrist_x"
        }
        if side_on and m.get("trunk_tilt") is not None:
            conf["trunk_tilt"] = float(vis[list(near)].min())
        return m, conf

    def process_missing(self, ts_ms: int, frame_idx: int = 0) -> FeedbackFrame:
        """Feed a frame where no person was detected."""
        return self._step(None, ts_ms, frame_idx)

    def process(self, frame: Frame) -> FeedbackFrame:
        """Feed one frame (timestamps must increase)."""
        return self._step(frame, frame.ts_ms, frame.frame_idx)

    def _step(self, frame: Frame | None, ts_ms: int, frame_idx: int) -> FeedbackFrame:
        t0 = time.perf_counter()
        t = ts_ms / 1000.0
        cfg, ex = self.cfg, self.exercise
        measures: dict[str, float | None] = {}
        conf: dict[str, float] = {}
        detected_view: View | None = None
        raw_frame = frame
        if frame is not None:
            if cfg.smoothing:
                frame = self._smoother(frame)
            measures, conf = self._measures(frame)
            detected_view = detect_view(frame.pixel_xy, cfg)

        primary = measures.get(ex.primary_angle)
        visibility_ok = frame is not None and angles.is_visible(
            frame.visibility,
            angles.required_landmarks(ex.primary_angle, self.side),
            cfg.visibility_threshold,
        )
        view_ok = detected_view is None or detected_view == self.view
        completed: RepRecord | None = None
        flags: list[Flag] = []

        if self.active:
            self._calib.update(t, {k: measures.get(k) for k in (*CALIBRATION_MEASURES, "wrist_x")})
            vel = self._velocity.update(t, primary) if primary is not None else 0.0
            if primary is None and self._counter.phase == "rest":
                self._velocity.reset()
            cand = self._counter.update(t, primary, vel)

            frame_flags, _ = self._rules.evaluate(t, measures, conf, self._calib.baseline)
            if self.flag_model is not None:
                rule_codes = {f.code for f in frame_flags}
                frame_flags += [
                    f for f in self.flag_model.update(t, raw_frame) if f.code not in rule_codes
                ]
            if self._counter.phase != "rest" or cand is not None:
                self._rep_codes.update(f.code for f in frame_flags)
            elif self._counter.phase == "rest":
                self._rep_codes.clear()
            flags.extend(frame_flags)

            if cand is not None:
                completed = self._complete(cand)
            if t < self._rep_flags_until:
                flags.extend(self._rep_flags)
            flags.extend(self._setup_flags(t, visibility_ok, view_ok))

        texts = ex.cue_texts
        if self.view == "sagittal" and frame is not None and not visibility_ok:
            # A person is detected but the exercising arm is hidden: they likely face the wrong way.
            arm = "right" if self.side == "R" else "left"
            texts = {**texts, "SETUP_VISIBILITY": f"Turn so your {arm} arm faces the camera"}
        cue = self._cues.update(t, flags, texts) if self.active else None
        self._record_events(t, flags)
        return FeedbackFrame(
            frame_idx=frame_idx,
            ts_ms=ts_ms,
            angles={
                k: measures.get(k) for k in dict.fromkeys((ex.primary_angle, *DISPLAY_MEASURES))
            },
            phase=self._counter.phase,
            rep_count=len(self.reps),
            last_rep=self.reps[-1] if self.reps else None,
            completed_rep=completed,
            flags=tuple(flags),
            cue=cue,
            visibility_ok=visibility_ok,
            view_ok=view_ok,
            calibrated=self._calib.done,
            proc_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def _complete(self, cand: RepCandidate) -> RepRecord:
        peak_vel = float(np.abs(cand.velocities).max()) if cand.velocities.size else 0.0
        rep_flags = self._rules.evaluate_rep(cand.peak_value, cand.duration_s, peak_vel)
        codes = self._rep_codes | {f.code for f in rep_flags}
        self._rep_codes = set()
        self._rep_flags = rep_flags
        self._rep_flags_until = cand.t_end + self.cfg.rep_flag_hold_s
        rec = RepRecord(
            idx=len(self.reps),
            t_start=round(cand.t_start, 3),
            t_peak=round(cand.t_peak, 3),
            t_end=round(cand.t_end, 3),
            peak_angle=round(cand.peak_value, 2),
            duration_s=round(cand.duration_s, 3),
            peak_vel_dps=round(peak_vel, 2),
            flags=tuple(sorted(codes)),
            label=label_rep(codes),
            dtw_similarity=round(dtw_similarity(cand.values, self._reference), 4),
        )
        self.reps.append(rec)
        return rec

    def _setup_flags(self, t: float, visibility_ok: bool, view_ok: bool) -> list[Flag]:
        cfg = self.cfg
        out: list[Flag] = []
        if visibility_ok:
            self._invisible_since = None
        else:
            self._invisible_since = (
                self._invisible_since if self._invisible_since is not None else t
            )
            if t - self._invisible_since >= cfg.setup_persist_s:
                out.append(Flag("SETUP_VISIBILITY", "error", 1.0))
        if view_ok:
            self._view_bad_since = None
        else:
            self._view_bad_since = self._view_bad_since if self._view_bad_since is not None else t
            if t - self._view_bad_since >= cfg.view_persist_s:
                out.append(Flag(f"SETUP_VIEW_{self.view.upper()}", "warn", 1.0))
        if visibility_ok and not self._calib.done:
            out.append(Flag("CALIBRATING", "info", 1.0))
        return out

    def _record_events(self, t: float, flags: list[Flag]) -> None:
        codes = {f.code for f in flags}
        for f in flags:
            if f.code not in self._prev_codes and f.code != "CALIBRATING":
                self.events.append(
                    Event(round(t, 3), f.code, f.severity, round(f.confidence, 3), f.source)
                )
        self._prev_codes = codes

    # --- summary ------------------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Session summary: counts, labels, mean ROM / similarity, flag counts."""
        labels = Counter(r.label for r in self.reps)
        n = len(self.reps)
        return {
            "exercise_id": self.exercise.id,
            "side": self.side,
            "view": self.view,
            "rep_count": n,
            "labels": dict(labels),
            "mean_rom": round(float(np.mean([r.peak_angle for r in self.reps])), 2) if n else None,
            "target": self.exercise.target,
            "mean_similarity": (
                round(float(np.mean([r.dtw_similarity for r in self.reps])), 4) if n else None
            ),
            "quality_score": round(labels.get("correct", 0) / n, 4) if n else None,
            "flag_counts": dict(Counter(c for r in self.reps for c in r.flags)),
            "events": len(self.events),
        }
