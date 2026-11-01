import numpy as np
import pytest

from rehabpose.core import angles
from rehabpose.core.features import (
    StreamingFeatures,
    canonical_points,
    frames_to_arrays,
    n_features,
    sequence_features,
)
from tests.fixtures.synthetic import RepSpec, SynthPose, make_frame, rep_sequence


def test_feature_sizes() -> None:
    assert n_features("2d") == 45 and n_features("3d") == 58
    fr = rep_sequence("shoulder_flexion", [RepSpec(150, duration_s=3.0)], noise_px=0.0)
    assert sequence_features(fr, "R", 30.0).shape == (len(fr), 45)
    assert sequence_features(fr, "R", 30.0, "3d").shape == (len(fr), 58)


@pytest.mark.parametrize("space", ["2d", "3d"])
def test_angle_features_match_core_angles(space: str) -> None:
    fr = make_frame(SynthPose(elevation_deg=70, elbow_flexion_deg=30), "sagittal")
    f = sequence_features([fr], "R", 30.0, space)[0]  # type: ignore[arg-type]
    k = 26 if space == "2d" else 39
    pts = fr.pixel_xy if space == "2d" else fr.world_xyz
    assert f[k] * 180 == pytest.approx(float(angles.upper_arm_trunk_angle(pts, "R")), abs=1e-6)
    assert f[k + 1] * 180 == pytest.approx(float(angles.elbow_flexion(pts, "R")), abs=1e-6)
    assert f[k + 2] * 180 == pytest.approx(float(angles.trunk_tilt(pts)), abs=1e-6)


def test_left_side_is_canonicalised_to_right() -> None:
    """A left-arm pose seen frontally gives the same features as the mirrored right-arm pose."""
    left = make_frame(SynthPose(side="L", elevation_deg=80, plane_deg=90), "frontal")
    right = make_frame(SynthPose(side="R", elevation_deg=80, plane_deg=90), "frontal")
    fl = sequence_features([left], "L", 30.0)[0]
    fr = sequence_features([right], "R", 30.0)[0]
    np.testing.assert_allclose(fl, fr, atol=1e-9)


def test_translation_and_scale_invariance() -> None:
    a = make_frame(SynthPose(elevation_deg=45), "sagittal", px_per_m=400)
    b = make_frame(SynthPose(elevation_deg=45, body_scale=1.2), "sagittal", px_per_m=300)
    np.testing.assert_allclose(
        sequence_features([a], "R", 30.0), sequence_features([b], "R", 30.0), atol=1e-9
    )


def test_invisible_joints_and_missing_frames_are_zero() -> None:
    occ = make_frame(SynthPose(elevation_deg=45), "sagittal", occluded=[16])
    f = sequence_features([occ, None], "R", 30.0)
    assert f[0, 26 + 1] == 0.0  # elbow flexion unknown without the wrist
    assert f[0, 26 + 3 + 8] == 0.0  # wrist visibility (canonical position 8)
    assert not f[1].any()  # missing frame: no spurious velocity either
    pts, vis = frames_to_arrays([None])
    assert pts.shape == (1, 33, 2) and not vis.any()


def test_streaming_equals_batch() -> None:
    frames = rep_sequence(
        "shoulder_abduction", [RepSpec(130, duration_s=3.0)], dropouts=[(0.5, 0.6)], side="L"
    )
    batch = sequence_features(frames, "L", 30.0)
    sf = StreamingFeatures("L", 30.0)
    stream = np.stack([sf.update(f) for f in frames])
    np.testing.assert_allclose(stream, batch, atol=1e-9)
    sf.reset()
    assert not sf.update(frames[5])[-3:].any()  # velocity restarts at 0


def test_canonical_points_centered_and_scaled() -> None:
    fr = make_frame(SynthPose(), "frontal")
    p, v = canonical_points(*frames_to_arrays([fr]), "R")
    hip_mid = (p[0, 11] + p[0, 12]) / 2
    np.testing.assert_allclose(hip_mid, 0.0, atol=1e-12)
    assert np.linalg.norm(p[0, 3] - p[0, 4]) == pytest.approx(1.0)  # shoulder width = unit
    assert v.shape == (1, 13)
