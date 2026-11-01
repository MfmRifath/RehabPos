"""Annotation schema v1 and skeleton joint maps."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from rehabpose.core import angles
from rehabpose.data.joint_maps import (
    KINECT_V2,
    PLUG_IN_GAIT,
    UPPER13,
    project_world,
    skeleton_to_frames,
    to_mediapipe_world,
)
from rehabpose.data.schema import (
    AngleAnnotation,
    Annotation,
    json_schema,
    load_annotation,
    save_annotation,
)
from tests.fixtures.datasets import kinect_skeleton, pig_markers
from tests.fixtures.synthetic import SynthPose, world_skeleton

# --- schema ------------------------------------------------------------------------------------


def _ann(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "video_id": "P01_x",
        "subject": "P01",
        "exercise_id": "shoulder_flexion",
        "side": "R",
        "view": "sagittal",
        "fps": 30.0,
        "annotator": "A1",
        "reps": [
            {"idx": 0, "start_frame": 10, "peak_frame": 40, "end_frame": 80, "label": "correct"},
            {
                "idx": 1,
                "start_frame": 90,
                "peak_frame": 120,
                "end_frame": 160,
                "label": "compensated",
                "flags": ["TRUNK_LEAN"],
            },
        ],  # fmt: skip
        "compensation_segments": [{"code": "TRUNK_LEAN", "start_frame": 100, "end_frame": 130}],
        "angle_annotations": [
            {
                "frame": 40,
                "measure": "upper_arm_trunk_angle",
                "points_px": [[10, 0], [0, 0], [0, 10]],
                "angle_deg": 90.0,
            },
        ],  # fmt: skip
    }
    base.update(kw)
    return base


def test_annotation_roundtrip(tmp_path: Path) -> None:
    ann = Annotation.model_validate(_ann())
    path = save_annotation(ann, tmp_path / "labels" / "a.json")
    assert load_annotation(path) == ann
    assert ann.schema_version == 1


def _rep(idx: int, s: int, p: int, e: int, label: str = "correct", **kw: Any) -> dict[str, Any]:
    return {"idx": idx, "start_frame": s, "peak_frame": p, "end_frame": e, "label": label, **kw}


ANGLE_45 = {"frame": 1, "measure": "m", "points_px": [[10, 0], [0, 0], [0, 10]], "angle_deg": 45}


@pytest.mark.parametrize(
    ("patch", "msg"),
    [
        ({"reps": [_rep(0, 50, 40, 80)]}, "start_frame"),
        ({"reps": [_rep(0, 10, 40, 80), _rep(1, 70, 90, 99)]}, "overlap"),
        ({"reps": [_rep(3, 1, 2, 3)]}, "0..n-1"),
        ({"reps": [_rep(0, 1, 2, 3, "great")]}, "label"),
        ({"reps": [_rep(0, 1, 2, 3, flags=["X"])]}, "unknown flags"),
        (
            {"compensation_segments": [{"code": "KNEE", "start_frame": 1, "end_frame": 2}]},
            "unknown",
        ),
        (
            {"compensation_segments": [{"code": "TRUNK_LEAN", "start_frame": 5, "end_frame": 2}]},
            "end",
        ),
        ({"angle_annotations": [ANGLE_45]}, "angle_deg"),
        ({"side": "B"}, "side"),
        ({"fps": 0}, "fps"),
        ({"extra": 1}, "Extra"),
    ],
)
def test_annotation_validation(patch: dict[str, Any], msg: str) -> None:
    with pytest.raises(ValidationError, match=msg):
        Annotation.model_validate(_ann(**patch))


def test_angle_from_points_and_json_schema() -> None:
    assert AngleAnnotation.angle_from_points(((1, 0), (0, 0), (-1, 0))) == pytest.approx(180)
    schema = json_schema()
    assert {"reps", "compensation_segments", "angle_annotations"} <= set(schema["properties"])


# --- joint maps --------------------------------------------------------------------------------

POSES = [
    SynthPose(elevation_deg=e, plane_deg=p, elbow_flexion_deg=f)
    for e, p, f in [(0, 90, 0), (45, 90, 20), (90, 0, 0), (130, 90, 10), (150, 0, 60)]
]


@pytest.mark.parametrize(
    ("jmap", "make"), [(KINECT_V2, kinect_skeleton), (PLUG_IN_GAIT, pig_markers)]
)
def test_mapped_world_preserves_arm_angles(jmap, make) -> None:  # type: ignore[no-untyped-def]
    world = to_mediapipe_world(make(POSES), jmap)
    assert world.shape == (len(POSES), 33, 4)
    assert set(np.flatnonzero(world[0, :, 3])) == set(UPPER13)
    for t, pose in enumerate(POSES):
        w = world[t, :, :3]
        assert angles.upper_arm_trunk_angle(w, "R") == pytest.approx(pose.elevation_deg, abs=0.5)
        assert angles.elbow_flexion(w, "R") == pytest.approx(pose.elbow_flexion_deg, abs=0.5)
        # hip-centred, MediaPipe axes: shoulders above hips (y down), +x = subject's left
        np.testing.assert_allclose((w[23] + w[24]) / 2, 0, atol=1e-9)
        assert w[11, 1] < 0 and w[11, 0] > w[12, 0]


def test_kinect_axes_match_mediapipe_world() -> None:
    world = to_mediapipe_world(kinect_skeleton([SynthPose(elevation_deg=40)]), KINECT_V2)[0]
    truth = world_skeleton(SynthPose(elevation_deg=40))
    mid = (truth[23] + truth[24]) / 2
    for i in (11, 12, 13, 14, 15, 16, 23, 24):
        np.testing.assert_allclose(world[i, :3], truth[i] - mid, atol=1e-9)


def test_untracked_joints_get_zero_visibility() -> None:
    skel = kinect_skeleton(POSES[:2])
    skel[0, KINECT_V2.joints.index("WristRight")] = 0.0
    skel[1, KINECT_V2.joints.index("ElbowLeft")] = np.nan
    world = to_mediapipe_world(skel, KINECT_V2)
    assert world[0, 16, 3] == 0 and world[0, 15, 3] == 1
    assert world[1, 13, 3] == 0 and np.isfinite(world).all()


def test_wrong_shape_rejected() -> None:
    with pytest.raises(ValueError, match="expected"):
        to_mediapipe_world(np.zeros((3, 20, 3)), KINECT_V2)


def test_skeleton_to_frames_frontal_2d_angles() -> None:
    abd = [SynthPose(elevation_deg=e, plane_deg=90) for e in (0, 60, 120)]
    frames = skeleton_to_frames(kinect_skeleton(abd), KINECT_V2, fps=30.0, view="frontal")
    assert [f.ts_ms for f in frames] == [0, 33, 67]
    for f, p in zip(frames, abd, strict=True):
        assert angles.upper_arm_trunk_angle(f.pixel_xy, "R") == pytest.approx(
            p.elevation_deg, abs=0.5
        )
        assert f.visibility[16] == 1.0 and f.visibility[25] == 0.0


def test_project_world_views() -> None:
    w = np.zeros((33, 4))
    w[:, 3] = 1
    w[0, :3] = [0.1, -0.2, -0.3]
    front = project_world(w, "frontal", 1000, 1000, 100)
    side = project_world(w, "sagittal", 1000, 1000, 100)
    assert front[0, 0] == pytest.approx(0.51) and side[0, 0] == pytest.approx(0.53)
    assert front[0, 1] == side[0, 1] == pytest.approx(0.53)
