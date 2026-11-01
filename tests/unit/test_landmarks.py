import numpy as np
import pytest

from rehabpose.core.landmarks import LM, Frame, side_joints
from tests.fixtures.synthetic import SynthPose, make_frame


def test_side_joints_indices() -> None:
    assert side_joints("L").shoulder == 11 and side_joints("R").shoulder == 12
    assert side_joints("L").ear == 7 and side_joints("R").hip == 24
    with pytest.raises(ValueError):
        side_joints("X")  # type: ignore[arg-type]


def test_frame_validates_shapes() -> None:
    with pytest.raises(ValueError, match=r"\(33, 4\)"):
        Frame(np.zeros((32, 4)), np.zeros((33, 4)), 0, 640, 480)
    with pytest.raises(ValueError, match="positive"):
        Frame(np.zeros((33, 4)), np.zeros((33, 4)), 0, 0, 480)


def test_pixel_xy_scales_by_image_size() -> None:
    lm = np.zeros((33, 4))
    lm[LM.NOSE] = [0.5, 0.25, 0.0, 1.0]
    fr = Frame(lm, np.zeros((33, 4)), 0, 640, 480)
    np.testing.assert_allclose(fr.pixel_xy[LM.NOSE], [320.0, 120.0])
    assert fr.visibility.shape == (33,) and fr.world_xyz.shape == (33, 3)


def test_json_roundtrip() -> None:
    fr = make_frame(SynthPose(elevation_deg=40), ts_ms=1234, frame_idx=7)
    back = Frame.from_json(fr.to_json())
    np.testing.assert_allclose(back.lm, fr.lm)
    np.testing.assert_allclose(back.world, fr.world)
    assert (back.ts_ms, back.img_w, back.img_h, back.frame_idx) == (1234, 1280, 720, 7)
    d = fr.to_dict()
    assert set(d) == {"frame", "ts_ms", "img", "lm", "world"}
