"""Offline analysis outputs and the `rehabpose analyze` CLI."""

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from rehabpose import DISCLAIMER
from rehabpose.cli import app
from rehabpose.core.exercise import load_exercises
from rehabpose.data import video_io
from rehabpose.offline import analyze
from rehabpose.pose.model_files import model_path
from rehabpose.settings import get_settings
from tests.fixtures.synthetic import RepSpec, rep_sequence, write_frames_parquet, write_frames_video

EX = load_exercises(Path(__file__).resolve().parents[2] / "configs" / "exercises")
LEAN = {"trunk_lean_deg": 15.0, "trunk_lean_plane": "sagittal"}


@pytest.fixture
def flexion_parquet(tmp_path: Path) -> Path:
    frames = rep_sequence(
        "shoulder_flexion",
        [RepSpec(150), RepSpec(150, compensation=LEAN), RepSpec(100)],
        img_w=640,
        img_h=360,
        px_per_m=200,
        dropouts=[(0.2, 0.3)],
    )
    return write_frames_parquet(tmp_path / "lm.parquet", frames)


def test_analyze_writes_all_outputs(flexion_parquet: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    res = analyze(flexion_parquet, EX["shoulder_flexion"], "R", out)
    assert {p.name for p in out.iterdir()} == {
        "reps.csv",
        "events.csv",
        "summary.json",
        "annotated.mp4",
    }

    reps = pd.read_csv(out / "reps.csv")
    assert list(reps["label"]) == ["correct", "compensated", "incomplete_rom"]
    assert list(reps["flags"].fillna("")) == ["", "TRUNK_LEAN", "INCOMPLETE_ROM"]
    assert {
        "t_start",
        "t_peak",
        "t_end",
        "peak_angle",
        "duration_s",
        "peak_vel_dps",
        "dtw_similarity",
    } <= set(reps.columns)

    events = pd.read_csv(out / "events.csv")
    assert "TRUNK_LEAN" in set(events["code"])

    summary = json.loads((out / "summary.json").read_text())
    assert summary == res.summary
    assert summary["rep_count"] == 3 and summary["disclaimer"] == DISCLAIMER
    assert summary["reference_curve"] == "synthetic"
    assert summary["n_frames"] == summary["n_detected"] + 3  # 0.1 s dropout at 30 fps
    assert summary["engine_ms"]["p95"] > 0

    info = video_io.probe(out / "annotated.mp4")
    assert (info.width, info.height) == (640, 360)
    assert info.frame_count == summary["n_frames"]


def test_analyze_without_video(flexion_parquet: Path, tmp_path: Path) -> None:
    res = analyze(flexion_parquet, EX["shoulder_flexion"], "R", tmp_path / "r", render_video=False)
    assert "video" not in res.files and not (tmp_path / "r" / "annotated.mp4").exists()


def test_analyze_empty_session_csvs(tmp_path: Path) -> None:
    from tests.fixtures.synthetic import SynthPose, make_frame

    frames = [make_frame(SynthPose(), "sagittal", ts_ms=i * 33, frame_idx=i) for i in range(30)]
    p = write_frames_parquet(tmp_path / "still.parquet", frames)
    res = analyze(p, EX["shoulder_flexion"], "R", tmp_path / "o", render_video=False)
    assert res.summary["rep_count"] == 0 and res.summary["mean_rom"] is None
    assert pd.read_csv(tmp_path / "o" / "reps.csv").empty


def test_cli_analyze_parquet(flexion_parquet: Path, tmp_path: Path) -> None:
    out = tmp_path / "cli"
    r = CliRunner().invoke(
        app,
        [
            "analyze",
            str(flexion_parquet),
            "--exercise",
            "shoulder_flexion",
            "--side",
            "R",
            "--out-dir",
            str(out),
            "--no-video",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "3 reps" in r.output and (out / "summary.json").is_file()


@pytest.mark.parametrize(
    "args",
    [
        ["--exercise", "squat"],
        ["--exercise", "shoulder_flexion", "--side", "X"],
        ["--exercise", "shoulder_flexion", "--view", "top"],
        ["--exercise", "shoulder_flexion", "--view", "frontal"],  # not allowed for flexion
    ],
)
def test_cli_analyze_bad_args(flexion_parquet: Path, tmp_path: Path, args: list[str]) -> None:
    r = CliRunner().invoke(
        app, ["analyze", str(flexion_parquet), *args, "--out-dir", str(tmp_path / "x")]
    )
    assert r.exit_code != 0


@pytest.mark.mediapipe
def test_cli_analyze_real_video_counts_reps(tmp_path: Path) -> None:
    """Real MediaPipe on a rendered mannequin abduction set (SYNTHETIC sanity check only)."""
    if not model_path("full", get_settings().models_dir).is_file():
        pytest.skip("pose_landmarker_full.task not present")
    frames = rep_sequence(
        "shoulder_abduction",
        [RepSpec(130, duration_s=7.0, pause_s=1.0)] * 3,
        rest_value=20.0,  # arm off the torso: MediaPipe sees the elbow at rest
        img_w=640,
        img_h=480,
        px_per_m=330,
        noise_px=0.0,
    )
    vid = write_frames_video(tmp_path / "abd.mp4", frames)
    out = tmp_path / "run"
    r = CliRunner().invoke(
        app,
        [
            "analyze",
            str(vid),
            "--exercise",
            "shoulder_abduction",
            "--side",
            "R",
            "--out-dir",
            str(out),
        ],
    )
    assert r.exit_code == 0, r.output
    summary = json.loads((out / "summary.json").read_text())
    print(
        f"real MediaPipe on mannequin: reps={summary['rep_count']} "
        f"labels={summary['labels']} flags={summary['flag_counts']} mean_rom={summary['mean_rom']}"
    )
    assert summary["rep_count"] == 3
    assert (out / "landmarks.parquet").is_file() and (out / "annotated.mp4").is_file()


def test_draw_overlay_shapes_and_cue_banner() -> None:
    import numpy as np

    from rehabpose.core.engine import SessionEngine
    from rehabpose.render import SEVERITY_COLOR, draw_overlay
    from tests.fixtures.synthetic import SynthPose, make_frame

    ex = EX["shoulder_flexion"]
    eng = SessionEngine(ex, "R")
    frame = make_frame(SynthPose(elevation_deg=90), "sagittal", img_w=640, img_h=360, px_per_m=200)
    fb = eng.process(frame)
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    out = draw_overlay(canvas, frame, fb, ex, "R")
    assert out.shape == canvas.shape and out.any() and not canvas.any()  # input untouched
    assert draw_overlay(canvas, None, fb, ex, "R").shape == canvas.shape

    from dataclasses import replace

    from rehabpose.core.feedback import Cue

    warn = replace(fb, cue=Cue("TRUNK_LEAN", "Keep your trunk upright", "warn", True))
    banner = draw_overlay(canvas, frame, warn, ex, "R")
    assert tuple(banner[330, 630]) == SEVERITY_COLOR["warn"]  # right edge of the banner
