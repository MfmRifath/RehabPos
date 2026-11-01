import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rehabpose.core import angles
from rehabpose.core.landmarks import LM
from tests.fixtures.synthetic import SynthPose, make_frame

TOL = 0.5  # degrees (phase 2 acceptance)


def test_angle_between_basics() -> None:
    assert angles.angle_between(np.array([1.0, 0]), np.array([0, 1.0])) == pytest.approx(90)
    assert angles.angle_between(np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])) == pytest.approx(180)
    assert angles.angle_between(np.array([1.0, 1]), np.array([2.0, 2])) == pytest.approx(
        0, abs=1e-9
    )
    assert np.isnan(angles.angle_between(np.zeros(2), np.array([1.0, 0])))
    with pytest.raises(ValueError):
        angles.angle_between(np.ones(4), np.ones(4))


def test_joint_angle_vectorized() -> None:
    a = np.array([[1.0, 0], [0, 1]])
    b = np.zeros((2, 2))
    c = np.array([[0.0, 1], [0, 1]])
    np.testing.assert_allclose(angles.joint_angle(a, b, c), [90.0, 0.0], atol=1e-9)


@pytest.mark.parametrize("side", ["L", "R"])
@pytest.mark.parametrize("elevation", [0, 15, 45, 90, 120, 150, 175])
def test_flexion_2d_sagittal_and_3d(side: str, elevation: float) -> None:
    fr = make_frame(SynthPose(side=side, elevation_deg=elevation), "sagittal")  # type: ignore[arg-type]
    assert angles.upper_arm_trunk_angle(fr.pixel_xy, side) == pytest.approx(elevation, abs=TOL)  # type: ignore[arg-type]
    assert angles.upper_arm_trunk_angle(fr.world_xyz, side) == pytest.approx(elevation, abs=TOL)  # type: ignore[arg-type]


@pytest.mark.parametrize("elevation", [0, 30, 90, 130, 170])
def test_abduction_2d_frontal(elevation: float) -> None:
    fr = make_frame(SynthPose(elevation_deg=elevation, plane_deg=90), "frontal")
    assert angles.upper_arm_trunk_angle(fr.pixel_xy, "R") == pytest.approx(elevation, abs=TOL)


@pytest.mark.parametrize("flex", [0, 20, 60, 90, 130, 150])
def test_elbow_flexion_2d_sagittal_and_3d(flex: float) -> None:
    fr = make_frame(SynthPose(elevation_deg=10, elbow_flexion_deg=flex), "sagittal")
    assert angles.elbow_flexion(fr.pixel_xy, "R") == pytest.approx(flex, abs=TOL)
    assert angles.elbow_interior(fr.world_xyz, "R") == pytest.approx(180 - flex, abs=TOL)


@pytest.mark.parametrize(
    ("lean", "plane", "view"),
    [(8, "frontal", "frontal"), (-12, "frontal", "frontal"), (15, "sagittal", "sagittal")],
)
def test_trunk_tilt(lean: float, plane: str, view: str) -> None:
    pose = SynthPose(trunk_lean_deg=lean, trunk_lean_plane=plane, elevation_deg=60)  # type: ignore[arg-type]
    fr = make_frame(pose, view)  # type: ignore[arg-type]
    assert angles.trunk_tilt(fr.pixel_xy) == pytest.approx(abs(lean), abs=TOL)
    assert angles.trunk_tilt(fr.world_xyz) == pytest.approx(abs(lean), abs=TOL)
    # arm angle relative to the trunk is unaffected by lean (3D)
    assert angles.upper_arm_trunk_angle(fr.world_xyz, "R") == pytest.approx(60, abs=TOL)


@pytest.mark.parametrize("side", ["L", "R"])
@pytest.mark.parametrize("rot", [-30, 0, 30, 60, 90])
def test_forearm_rotation_3d(side: str, rot: float) -> None:
    pose = SynthPose(side=side, elbow_flexion_deg=90, rotation_deg=rot)  # type: ignore[arg-type]
    fr = make_frame(pose, "frontal")
    assert angles.forearm_rotation(fr.world_xyz, side) == pytest.approx(rot, abs=TOL)  # type: ignore[arg-type]


def test_hike_ratio_drops_when_shoulder_hikes() -> None:
    base = angles.hike_ratio(make_frame(SynthPose(), "frontal").pixel_xy, "R")
    hiked = angles.hike_ratio(make_frame(SynthPose(hike_m=0.04), "frontal").pixel_xy, "R")
    assert base > 0
    # ear-shoulder gap shrinks by 0.04 m out of 0.18 m; shoulder width barely changes
    assert hiked / base == pytest.approx((0.18 - 0.04) / 0.18, rel=0.02)


def test_shoulder_width_shrinks_with_trunk_rotation() -> None:
    w0 = angles.shoulder_width(make_frame(SynthPose(), "frontal").pixel_xy)
    w30 = angles.shoulder_width(make_frame(SynthPose(trunk_rotation_deg=30), "frontal").pixel_xy)
    assert w30 / w0 == pytest.approx(np.cos(np.radians(30)), rel=1e-6)


def test_reach_ratio() -> None:
    rest = make_frame(SynthPose(), "sagittal")
    reach = make_frame(SynthPose(elevation_deg=90), "sagittal")
    rest_w = rest.pixel_xy[LM.RIGHT_WRIST]
    assert angles.reach_ratio(rest.pixel_xy, "R", rest_w) == pytest.approx(0, abs=1e-9)
    # arm horizontal and straight: wrist moves forward by the full arm length
    assert angles.reach_ratio(reach.pixel_xy, "R", rest_w) == pytest.approx(1.0, abs=1e-6)


def test_sequence_broadcasting() -> None:
    frames = [make_frame(SynthPose(elevation_deg=e), "sagittal") for e in (10, 50, 90)]
    pts = np.stack([f.pixel_xy for f in frames])
    np.testing.assert_allclose(angles.upper_arm_trunk_angle(pts, "R"), [10, 50, 90], atol=TOL)


def test_compute_measures_and_visibility_gate() -> None:
    fr = make_frame(SynthPose(elevation_deg=70, elbow_flexion_deg=20), "sagittal")
    m = angles.compute_measures(fr, "R")
    assert set(m) == set(angles.MEASURES)
    assert m["upper_arm_trunk_angle"] == pytest.approx(70, abs=TOL)
    assert m["elbow_flexion"] == pytest.approx(20, abs=TOL)

    occluded = make_frame(SynthPose(elevation_deg=70), "sagittal", occluded=[LM.RIGHT_WRIST])
    m = angles.compute_measures(occluded, "R")
    assert m["elbow_interior"] is None and m["elbow_flexion"] is None
    assert m["upper_arm_trunk_angle"] is not None  # does not need the wrist


def test_noisy_angles_stay_close() -> None:
    rng = np.random.default_rng(1)
    errs = [
        angles.upper_arm_trunk_angle(
            make_frame(SynthPose(elevation_deg=90), "sagittal", noise_px=1.0, rng=rng).pixel_xy, "R"
        )
        - 90
        for _ in range(200)
    ]
    assert np.mean(np.abs(errs)) < 1.0  # 1 px noise at ~120 px segments


coords = st.floats(-500, 500, allow_nan=False, width=64)
vec = st.tuples(coords, coords, coords)


@settings(max_examples=200, deadline=None)
@given(a=vec, b=vec, c=vec, t=vec, s=st.floats(0.01, 100), dim=st.sampled_from([2, 3]))
def test_angle_translation_scale_invariance(a, b, c, t, s, dim) -> None:  # type: ignore[no-untyped-def]
    a, b, c, t = (np.array(v)[:dim] for v in (a, b, c, t))
    if min(np.linalg.norm(a - b), np.linalg.norm(c - b)) < 1e-3:
        return
    base = angles.joint_angle(a, b, c)
    moved = angles.joint_angle(s * a + t, s * b + t, s * c + t)
    assert moved == pytest.approx(base, abs=1e-6)


@settings(max_examples=100, deadline=None)
@given(
    el=st.floats(0, 180), flex=st.floats(0, 150), plane=st.floats(0, 90),
    t=vec, s=st.floats(0.2, 5),
)  # fmt: skip
def test_measures_invariant_to_skeleton_translation_scale(el, flex, plane, t, s) -> None:  # type: ignore[no-untyped-def]
    w = make_frame(SynthPose(elevation_deg=el, plane_deg=plane, elbow_flexion_deg=flex)).world_xyz
    w2 = s * w + np.array(t)
    assert angles.upper_arm_trunk_angle(w, "R") == pytest.approx(el, abs=TOL)
    assert angles.elbow_flexion(w, "R") == pytest.approx(flex, abs=TOL)
    assert angles.upper_arm_trunk_angle(w2, "R") == pytest.approx(el, abs=1e-6)
    assert angles.elbow_flexion(w2, "R") == pytest.approx(flex, abs=1e-6)


def test_body_scale_floors_collapsed_shoulder_width() -> None:
    frontal = make_frame(SynthPose(), "frontal").pixel_xy
    sagittal = make_frame(SynthPose(), "sagittal").pixel_xy
    assert angles.body_scale(frontal) == pytest.approx(angles.shoulder_width(frontal))
    assert angles.shoulder_width(sagittal) == pytest.approx(0, abs=1e-6)
    assert angles.body_scale(sagittal) == pytest.approx(0.7 * 0.5 * 400)  # 0.7 x torso px
