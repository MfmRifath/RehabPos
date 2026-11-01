"""MediaPipe adapter: result conversion (no model needed) and a real run if a model is present."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rehabpose.pose.mediapipe_pose import result_to_frame
from rehabpose.pose.model_files import model_path
from rehabpose.settings import get_settings
from tests.fixtures.synthetic import write_synthetic_video


def _lm(i: int, vis: float | None = 0.9) -> SimpleNamespace:
    return SimpleNamespace(x=i / 100, y=0.5, z=-0.1, visibility=vis)


def test_result_to_frame_converts_first_pose() -> None:
    res = SimpleNamespace(
        pose_landmarks=[[_lm(i) for i in range(33)]],
        pose_world_landmarks=[
            [SimpleNamespace(x=0.1, y=-0.2, z=0.0, visibility=None) for _ in range(33)]
        ],
    )
    fr = result_to_frame(res, ts_ms=500, img_w=640, img_h=480)
    assert fr is not None
    assert fr.lm.shape == (33, 4) and fr.lm[10, 0] == pytest.approx(0.10)
    np.testing.assert_allclose(fr.world[:, 3], fr.lm[:, 3])  # world copies image visibility
    assert (fr.ts_ms, fr.img_w, fr.img_h) == (500, 640, 480)


def test_result_to_frame_none_without_person() -> None:
    assert (
        result_to_frame(SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[]), 0, 1, 1)
        is None
    )


def test_result_to_frame_rejects_wrong_count() -> None:
    res = SimpleNamespace(pose_landmarks=[[_lm(i) for i in range(17)]], pose_world_landmarks=[])
    with pytest.raises(ValueError, match="33"):
        result_to_frame(res, 0, 640, 480)


@pytest.mark.mediapipe
@pytest.mark.parametrize("variant", ["lite", "full", "heavy"])
def test_real_mediapipe_on_synthetic_mannequin(tmp_path: Path, variant: str) -> None:
    """Real model end-to-end on a drawn cartoon: a sanity check, not an accuracy result."""
    path = model_path(variant, get_settings().models_dir)  # type: ignore[arg-type]
    if not path.is_file():
        pytest.skip(f"model file {path} not present (download needs owner consent)")
    from rehabpose.core.angles import upper_arm_trunk_angle
    from rehabpose.data.extract import extract_video, load_frames
    from rehabpose.pose.mediapipe_pose import MediaPipePose

    vid = write_synthetic_video(tmp_path / "m.mp4", n_frames=60, style="mannequin")
    with MediaPipePose(path) as est:
        res = extract_video(vid.path, est, tmp_path / "out.parquet")
    print(f"{variant}: detected {res.n_detected}/{res.n_frames}")
    assert res.n_frames == 60
    assert res.n_detected >= 48  # >= 80 % of frames

    errs = [
        float(upper_arm_trunk_angle(fr.pixel_xy, "R") - upper_arm_trunk_angle(gt.pixel_xy, "R"))
        for fr, gt in zip(load_frames(tmp_path / "out.parquet"), vid.frames, strict=True)
        if fr is not None and fr.visibility[[12, 14, 24]].min() >= 0.6
    ]
    mae = float(np.mean(np.abs(errs)))
    print(f"{variant}: cartoon abduction angle MAE {mae:.1f} deg over {len(errs)} frames")
    assert len(errs) >= 30 and mae < 15.0  # loose: cartoon, not a human
