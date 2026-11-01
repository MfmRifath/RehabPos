"""Offline analysis: video or landmark parquet -> engine -> annotated MP4 + CSV/JSON outputs."""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rehabpose import DISCLAIMER, __version__
from rehabpose.core.engine import Event, FeedbackFrame, RepRecord, SessionEngine
from rehabpose.core.exercise import ExerciseConfig, View
from rehabpose.core.landmarks import Frame, Side
from rehabpose.data.extract import extract_video, load_frames, read_meta
from rehabpose.data.video_io import VideoWriter, iter_frames
from rehabpose.render import draw_overlay

log = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Paths and summary written by `analyze`."""

    out_dir: Path
    summary: dict[str, Any]
    files: dict[str, Path] = field(default_factory=dict)
    reps: list[RepRecord] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)


def run_engine(
    frames: list[Frame | None], fps: float, engine: SessionEngine
) -> list[FeedbackFrame]:
    """Feed all frames (None = no person) through the engine in order."""
    out = []
    for i, fr in enumerate(frames):
        if fr is None:
            out.append(engine.process_missing(int(round(i * 1000.0 / fps)), i))
        else:
            out.append(engine.process(fr))
    return out


def _write_video(
    path: Path,
    frames: list[Frame | None],
    feedback: list[FeedbackFrame],
    engine: SessionEngine,
    fps: float,
    source: Path | None,
    size: tuple[int, int],
) -> None:
    ex, side = engine.exercise, engine.side
    with VideoWriter(path, fps, size) as wr:
        if source is not None:
            for (idx, _, img), fb in zip(iter_frames(source), feedback, strict=False):
                wr.write(draw_overlay(img, frames[idx], fb, ex, side))
        else:
            canvas = np.full((size[1], size[0], 3), 40, dtype=np.uint8)
            for fr, fb in zip(frames, feedback, strict=True):
                wr.write(draw_overlay(canvas, fr, fb, ex, side))


def analyze(
    landmarks_path: Path,
    exercise: ExerciseConfig,
    side: Side,
    out_dir: Path,
    view: View | None = None,
    video: Path | None = None,
    render_video: bool = True,
) -> AnalysisResult:
    """Analyze a landmark parquet file; writes reps.csv, events.csv, summary.json, annotated.mp4.

    The annotated video is drawn over `video` (or the parquet's source video when it exists),
    otherwise over a blank canvas.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = read_meta(landmarks_path)
    frames = load_frames(landmarks_path)
    fps = float(meta.get("fps") or 30.0)

    engine = SessionEngine(exercise, side, view)
    feedback = run_engine(frames, fps, engine)
    files: dict[str, Path] = {}

    reps_df = pd.DataFrame([r.to_dict() for r in engine.reps], columns=_REP_COLUMNS)
    reps_df["flags"] = reps_df["flags"].map(lambda f: ";".join(f) if isinstance(f, list) else "")
    files["reps"] = out_dir / "reps.csv"
    reps_df.to_csv(files["reps"], index=False)

    events_df = pd.DataFrame(
        [asdict(e) for e in engine.events],
        columns=["t", "code", "severity", "confidence", "source"],
    )
    files["events"] = out_dir / "events.csv"
    events_df.to_csv(files["events"], index=False)

    proc = np.array([fb.proc_ms for fb in feedback]) if feedback else np.zeros(1)
    source = video or (Path(meta["source"]) if meta.get("source") else None)
    if source is not None and not source.is_file():
        source = None
    first = next((f for f in frames if f is not None), None)
    size = (
        (int(meta["width"]), int(meta["height"]))
        if meta.get("width")
        else ((first.img_w, first.img_h) if first else (1280, 720))
    )
    summary = {
        **engine.summary(),
        "exercise_name": exercise.name,
        "experimental": exercise.experimental,
        "input": str(landmarks_path),
        "source_video": str(source) if source else None,
        "pose": {k: meta.get(k) for k in ("pose_backend", "model")},
        "n_frames": len(frames),
        "n_detected": sum(f is not None for f in frames),
        "fps": fps,
        "engine_ms": {
            "mean": round(float(proc.mean()), 4),
            "p95": round(float(np.percentile(proc, 95)), 4),
        },
        "reference_curve": exercise.reference.source,
        "versions": {"rehabpose": __version__, "python": platform.python_version()},
        "disclaimer": DISCLAIMER,
    }
    files["summary"] = out_dir / "summary.json"
    files["summary"].write_text(json.dumps(summary, indent=2))

    if render_video:
        files["video"] = out_dir / "annotated.mp4"
        _write_video(files["video"], frames, feedback, engine, fps, source, size)
    log.info("Analysis written to %s (%d reps)", out_dir, len(engine.reps))
    return AnalysisResult(out_dir, summary, files, list(engine.reps), list(engine.events))


_REP_COLUMNS = [
    "idx", "t_start", "t_peak", "t_end", "peak_angle", "duration_s", "peak_vel_dps",
    "flags", "label", "dtw_similarity",
]  # fmt: skip


def extract_for_analysis(
    video: Path, out_dir: Path, model: str, models_dir: Path, download: bool
) -> Path:
    """Run MediaPipe on a video and save landmarks.parquet into out_dir."""
    from rehabpose.pose import create_estimator
    from rehabpose.pose.model_files import VARIANTS

    if model not in VARIANTS:
        raise ValueError(f"model must be one of {VARIANTS}")
    estimator = create_estimator(model, models_dir, download=download)
    try:
        out = Path(out_dir) / "landmarks.parquet"
        extract_video(video, estimator, out, meta={"pose_backend": "mediapipe", "model": model})
    finally:
        estimator.close()
    return out
