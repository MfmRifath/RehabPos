"""Labelled sets (synthetic corpus or own recordings) and rep / streaming windows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from rehabpose.core.features import (
    Space,
    canonical_points,
    features_from_canonical,
    frames_to_arrays,
)
from rehabpose.core.landmarks import Frame, Side
from rehabpose.ml.transforms import AugmentConfig, augment, augment_batch, resample_time

FloatArray = NDArray[np.float64]
Task = Literal["rep", "stream"]

REP_LABELS: tuple[str, ...] = ("correct", "incomplete_rom", "too_fast", "compensated")
STREAM_CODES: tuple[str, ...] = (
    "TRUNK_LEAN",
    "SHOULDER_HIKE",
    "ELBOW_BEND",
    "SHOULDER_DRIFT",
    "ELBOW_AWAY",
    "TRUNK_ROTATION",
)
DEFAULT_EXERCISES = ("shoulder_flexion", "shoulder_abduction", "elbow_flexion")
TARGETS = {
    "shoulder_flexion": 150.0,
    "shoulder_abduction": 130.0,
    "elbow_flexion": 130.0,
    "shoulder_external_rotation": 60.0,
}


@dataclass
class RepTruth:
    """Ground-truth rep: inclusive frame range, label and compensation codes."""

    start: int
    end: int
    label: str
    codes: tuple[str, ...] = ()


@dataclass
class LabelledSet:
    """One recorded or synthetic set with frame-level ground truth."""

    subject: str
    exercise_id: str
    side: Side
    fps: float
    frames: list[Frame | None]
    reps: list[RepTruth]
    frame_codes: NDArray[np.bool_]  # (T, len(STREAM_CODES))
    angle_truth: FloatArray | None = None  # (T,) primary measure ground truth (synthetic only)
    source: str = "synthetic"


def _compensation(exercise: str, code: str, rng: np.random.Generator) -> dict[str, float | str]:
    u = rng.uniform
    table: dict[tuple[str, str], dict[str, float | str]] = {
        ("shoulder_flexion", "TRUNK_LEAN"): {
            "trunk_lean_deg": u(12, 20),
            "trunk_lean_plane": "sagittal",
        },
        ("shoulder_flexion", "ELBOW_BEND"): {"elbow_flexion_deg": u(30, 50)},
        ("shoulder_abduction", "TRUNK_LEAN"): {"trunk_lean_deg": -u(10, 16)},
        ("shoulder_abduction", "SHOULDER_HIKE"): {"hike_m": u(0.04, 0.06)},
        ("shoulder_abduction", "ELBOW_BEND"): {
            "elbow_flexion_deg": u(30, 50),
            "rotation_deg": 90.0,
        },
        ("elbow_flexion", "SHOULDER_DRIFT"): {"elevation_deg": u(30, 45)},
        ("elbow_flexion", "TRUNK_LEAN"): {
            "trunk_lean_deg": u(12, 20),
            "trunk_lean_plane": "sagittal",
        },
        ("shoulder_external_rotation", "ELBOW_AWAY"): {
            "elevation_deg": u(25, 35),
            "plane_deg": 90.0,
        },
        ("shoulder_external_rotation", "TRUNK_ROTATION"): {"trunk_rotation_deg": u(30, 45)},
    }
    return table[(exercise, code)]


COMPENSATION_OPTIONS: dict[str, tuple[str, ...]] = {
    "shoulder_flexion": ("TRUNK_LEAN", "ELBOW_BEND"),
    "shoulder_abduction": ("TRUNK_LEAN", "SHOULDER_HIKE", "ELBOW_BEND"),
    "elbow_flexion": ("SHOULDER_DRIFT", "TRUNK_LEAN"),
    "shoulder_external_rotation": ("ELBOW_AWAY", "TRUNK_ROTATION"),
}


def synthetic_corpus(
    n_subjects: int = 16,
    sets_per_exercise: int = 2,
    reps_per_set: int = 8,
    exercises: Sequence[str] = DEFAULT_EXERCISES,
    seed: int = 0,
    fps: float = 30.0,
    label_probs: Sequence[float] = (0.45, 0.2, 0.15, 0.2),
    noise_m: float = 0.0,
) -> list[LabelledSet]:
    """SYNTHETIC corpus: per-subject body size, camera scale, noise, tempo and side.

    noise_m adds gaussian noise (metres) to world landmarks (default: exact).

    Labels come from the generator (not from the rule engine): correct (full ROM, slow),
    incomplete_rom (peak 55-85 % of target), too_fast (2.4-3.5 s reps), compensated (one
    exercise-specific compensation ramping in with the movement).
    """
    from rehabpose.synth import RepSpec, rep_sequence_with_truth

    rng = np.random.default_rng(seed)
    sets: list[LabelledSet] = []
    for si in range(n_subjects):
        subject = f"S{si + 1:02d}"
        body = rng.uniform(0.85, 1.15)
        px_per_m = rng.uniform(300, 450)
        noise = rng.uniform(0.5, 2.0)
        tempo = rng.uniform(0.85, 1.2)
        side: Side = "L" if rng.random() < 0.5 else "R"
        for ex in exercises:
            target = TARGETS[ex]
            for _ in range(sets_per_exercise):
                specs, truth = [], []
                for _ in range(reps_per_set):
                    kind = REP_LABELS[int(rng.choice(4, p=list(label_probs)))]
                    comp: dict[str, float | str] = {}
                    codes: tuple[str, ...] = ()
                    if kind == "correct":
                        peak, dur = target * rng.uniform(0.95, 1.03), rng.uniform(7, 10) * tempo
                    elif kind == "incomplete_rom":
                        peak, dur = target * rng.uniform(0.55, 0.85), rng.uniform(6, 9) * tempo
                    elif kind == "too_fast":
                        peak, dur = target * rng.uniform(0.95, 1.0), rng.uniform(2.4, 3.5)
                    else:
                        code = str(rng.choice(COMPENSATION_OPTIONS[ex]))
                        comp, codes = _compensation(ex, code, rng), (code,)
                        peak, dur = target * rng.uniform(0.95, 1.0), rng.uniform(7, 10) * tempo
                    comp["body_scale"] = float(body)
                    specs.append(
                        RepSpec(
                            float(peak),
                            float(dur),
                            hold_s=float(rng.choice([0.0, 0.5])),
                            pause_s=float(rng.uniform(0.6, 2.0)),
                            compensation=comp,
                        )
                    )
                    truth.append((kind, codes))
                frames, values, ramps, owner = rep_sequence_with_truth(
                    ex,
                    specs,
                    fps=fps,
                    rest_s=1.5,
                    side=side,
                    noise_px=float(noise),
                    seed=int(rng.integers(1 << 31)),
                    px_per_m=float(px_per_m),
                    noise_m=noise_m,
                    rest_value=0.0,
                )
                own = np.asarray(owner)
                reps, fcodes = [], np.zeros((len(frames), len(STREAM_CODES)), dtype=bool)
                for k, (kind, codes) in enumerate(truth):
                    idx = np.flatnonzero(own == k)
                    reps.append(RepTruth(int(idx[0]), int(idx[-1]), kind, codes))
                    for c in codes:
                        fcodes[idx[ramps[idx] >= 0.5], STREAM_CODES.index(c)] = True
                sets.append(LabelledSet(subject, ex, side, fps, frames, reps, fcodes, values))
    return sets


@dataclass
class WindowSet:
    """Windows of canonical points with labels and subject groups."""

    task: Task
    points: list[FloatArray]  # each (L, 13, D)
    vis: list[FloatArray]  # each (L, 13)
    y: NDArray[Any]  # (N,) int for rep, (N, C) float for stream
    groups: NDArray[np.str_]
    fps: float
    classes: tuple[str, ...]
    meta: list[dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.points)

    def subset(self, idx: Iterable[int]) -> WindowSet:
        """Windows at the given indices."""
        ii = list(idx)
        return WindowSet(
            self.task,
            [self.points[i] for i in ii],
            [self.vis[i] for i in ii],
            self.y[ii],
            self.groups[ii],
            self.fps,
            self.classes,
            [self.meta[i] for i in ii] if self.meta else [],
        )

    def class_counts(self) -> dict[str, int]:
        """Number of windows per class (rep) or positive windows per code (stream)."""
        if self.task == "rep":
            c = Counter(int(v) for v in self.y)
            return {name: c.get(i, 0) for i, name in enumerate(self.classes)}
        return {name: int(self.y[:, i].sum()) for i, name in enumerate(self.classes)}


def _canonical(s: LabelledSet, space: Space) -> tuple[FloatArray, FloatArray]:
    pts, vis = frames_to_arrays(s.frames, space)
    return canonical_points(pts, vis, s.side)


def rep_windows(sets: Sequence[LabelledSet], space: Space = "2d") -> WindowSet:
    """One window per ground-truth rep (variable length; resampled at featurization)."""
    pts_l, vis_l, y, g, meta = [], [], [], [], []
    for s in sets:
        p, v = _canonical(s, space)
        for r in s.reps:
            pts_l.append(p[r.start : r.end + 1])
            vis_l.append(v[r.start : r.end + 1])
            y.append(REP_LABELS.index(r.label))
            g.append(s.subject)
            meta.append({"exercise": s.exercise_id, "codes": r.codes})
    fps = sets[0].fps if sets else 30.0
    return WindowSet(
        "rep", pts_l, vis_l, np.asarray(y, dtype=np.int64), np.asarray(g), fps, REP_LABELS, meta
    )


def stream_codes_for(exercises: Iterable[str]) -> tuple[str, ...]:
    """Compensation codes that can occur for the given exercises (STREAM_CODES order)."""
    present = {c for e in exercises for c in COMPENSATION_OPTIONS.get(e, ())}
    return tuple(c for c in STREAM_CODES if c in present)


def stream_windows(
    sets: Sequence[LabelledSet],
    length: int = 45,
    stride: int = 5,
    space: Space = "2d",
    codes: Sequence[str] | None = None,
) -> WindowSet:
    """Causal windows ending at frame t (stride `stride`); label = codes active at t.

    `codes` defaults to the compensations possible for the exercises present in `sets`.
    """
    codes = tuple(codes) if codes is not None else stream_codes_for(s.exercise_id for s in sets)
    cols = [STREAM_CODES.index(c) for c in codes]
    pts_l, vis_l, y, g, meta = [], [], [], [], []
    for s in sets:
        p, v = _canonical(s, space)
        for end in range(length - 1, len(s.frames), stride):
            pts_l.append(p[end - length + 1 : end + 1])
            vis_l.append(v[end - length + 1 : end + 1])
            y.append(s.frame_codes[end, cols].astype(np.float32))
            g.append(s.subject)
            meta.append({"exercise": s.exercise_id})
    fps = sets[0].fps if sets else 30.0
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1, len(codes))
    return WindowSet("stream", pts_l, vis_l, y_arr, np.asarray(g), fps, codes, meta)


def featurize(
    ws: WindowSet,
    rep_length: int = 64,
    rng: np.random.Generator | None = None,
    aug: AugmentConfig | None = None,
    idx: Sequence[int] | NDArray[np.int64] | None = None,
) -> FloatArray:
    """(N, L, F) float32 features (optionally augmented); rep windows resampled to rep_length."""
    ii = list(range(len(ws)) if idx is None else idx)
    if ws.task == "stream" and ii:  # equal-length windows: vectorized path
        p = np.stack([ws.points[i] for i in ii])
        v = np.stack([ws.vis[i] for i in ii])
        if aug is not None and rng is not None:
            p, v = augment_batch(p, v, rng, aug)
        return np.asarray(features_from_canonical(p, v, ws.fps), dtype=np.float32)
    out = []
    for i in ii:
        p, v = ws.points[i], ws.vis[i]
        if aug is not None and rng is not None:
            p, v = augment(p, v, rng, aug)
        f = features_from_canonical(p, v, ws.fps)
        if ws.task == "rep":
            f = resample_time(f, rep_length)
        out.append(f)
    return np.asarray(out, dtype=np.float32)
