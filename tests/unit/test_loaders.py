"""Dataset loaders on tiny fixtures mimicking each ASSUMED layout; never touches the network."""

from pathlib import Path

import numpy as np
import pytest

from rehabpose.core import angles
from rehabpose.data.joint_maps import KINECT_V2, skeleton_to_frames
from rehabpose.data.loaders import (
    REGISTRY,
    DatasetMissingError,
    LayoutError,
    LicenseNotAcceptedError,
    get_loader,
)
from rehabpose.data.loaders.base import read_skeleton_rows
from tests.fixtures.datasets import (
    kinect_skeleton,
    make_intellirehab,
    make_kimore,
    make_own,
    make_trsp,
    make_uiprmd,
    make_videos,
    write_kinect_txt,
)
from tests.fixtures.synthetic import SynthPose

PUBLIC = sorted(set(REGISTRY) - {"own"})


def test_registry_has_all_spec_datasets() -> None:
    assert set(REGISTRY) == {
        "own", "uco", "rehab24_6", "dib_upper", "intellirehab", "uiprmd", "kimore", "trsp",
    }  # fmt: skip
    for name, cls in REGISTRY.items():
        info = cls.info
        assert info.name == name and info.license and info.citation and info.instructions
        assert info.layout_verified is (name == "own")  # public layouts are assumptions
    with pytest.raises(KeyError):
        get_loader("imagenet")


@pytest.mark.parametrize("name", PUBLIC)
def test_license_must_be_accepted(name: str, tmp_path: Path) -> None:
    with pytest.raises(LicenseNotAcceptedError, match="--i-accept-license") as exc:
        list(get_loader(name, tmp_path).sequences())
    assert REGISTRY[name].info.license in str(exc.value)


@pytest.mark.parametrize("name", PUBLIC)
def test_missing_data_gives_instructions(name: str, tmp_path: Path) -> None:
    with pytest.raises(DatasetMissingError, match="never downloads") as exc:
        list(get_loader(name, tmp_path, accept_license=True).sequences())
    assert f"raw/{name}" in str(exc.value)


def test_files_not_matching_layout_raise_layout_error(tmp_path: Path) -> None:
    (tmp_path / "raw" / "intellirehab").mkdir(parents=True)
    (tmp_path / "raw" / "intellirehab" / "weird_name.txt").write_text("1 2 3")
    with pytest.raises(LayoutError, match="did not match|matched the expected layout"):
        list(get_loader("intellirehab", tmp_path, True).sequences())


def test_intellirehab_kinect_sequences(tmp_path: Path) -> None:
    make_intellirehab(tmp_path)
    seqs = list(get_loader("intellirehab", tmp_path, True).sequences())
    assert len(seqs) == 4  # README.txt skipped
    s = seqs[0]
    assert (s.dataset, s.subject, s.exercise, s.joint_format) == (
        "intellirehab",
        "101",
        "4",
        "kinect_v2_25",
    )
    assert s.labels["correctness"] == "1" and s.labels["position"] == "stand"
    assert s.exercise_id is None  # gesture codes unmapped until verified
    assert s.skeleton is not None and s.skeleton.shape == (20, 25, 3) and s.n_frames == 20
    frames = skeleton_to_frames(s.skeleton, KINECT_V2, s.fps, "frontal")
    assert angles.upper_arm_trunk_angle(frames[-1].pixel_xy, "R") == pytest.approx(120, abs=0.5)


def test_kimore_csv_sequences(tmp_path: Path) -> None:
    make_kimore(tmp_path)
    (s,) = get_loader("kimore", tmp_path, True).sequences()
    assert (s.subject, s.exercise, s.labels["group"], s.labels["subgroup"]) == (
        "E_ID1",
        "1",
        "CG",
        "Expert",
    )
    assert s.skeleton is not None and s.skeleton.shape == (10, 25, 3)


def test_uiprmd_correct_and_incorrect(tmp_path: Path) -> None:
    make_uiprmd(tmp_path)
    seqs = list(get_loader("uiprmd", tmp_path, True).sequences())
    assert sorted(s.labels["correct"] for s in seqs) == [False, True]
    assert {s.joint_format for s in seqs} == {"unknown_22"}  # joint order not assumed
    assert {(s.subject, s.exercise) for s in seqs} == {("01", "07")}


def test_trsp_frame_labels(tmp_path: Path) -> None:
    make_trsp(tmp_path)
    (s,) = get_loader("trsp", tmp_path, True).sequences()
    assert s.subject == "S01" and s.skeleton is not None and s.skeleton.shape == (6, 25, 3)
    assert s.labels["frame_codes"] == [
        None,
        None,
        "TRUNK_LEAN",
        "TRUNK_LEAN",
        "SHOULDER_HIKE",
        "TRUNK_ROTATION",
    ]


def test_trsp_unknown_label_rejected(tmp_path: Path) -> None:
    import pandas as pd

    make_trsp(tmp_path)
    f = tmp_path / "raw" / "trsp" / "S01" / "seq1.csv"
    df = pd.read_csv(f)
    df.loc[0, "label"] = "dancing"
    df.to_csv(f, index=False)
    with pytest.raises(LayoutError, match="dancing"):
        list(get_loader("trsp", tmp_path, True).sequences())


@pytest.mark.parametrize(
    ("name", "rel", "expect"),
    [
        ("uco", "S03/flexion/cam1.mp4", {"subject": "S03", "exercise": "flexion"}),
        ("rehab24_6", "P2/ex4/take1.mp4", {"subject": "P2", "exercise": "ex4"}),
    ],
)
def test_video_loaders(name: str, rel: str, expect: dict[str, str], tmp_path: Path) -> None:
    make_videos(tmp_path, name, [rel])
    (s,) = get_loader(name, tmp_path, True).sequences()
    assert s.joint_format == "video" and s.video is not None and s.video.name == Path(rel).name
    assert {"subject": s.subject, "exercise": s.exercise} == expect
    assert s.fps == pytest.approx(30.0) and s.frames is None


def test_dib_upper_complete_flag_and_unknown_subjects(tmp_path: Path) -> None:
    make_videos(tmp_path, "dib_upper", ["ex1/Complete/a.mp4", "ex1/incomplete/b.mp4"])
    seqs = sorted(get_loader("dib_upper", tmp_path, True).sequences(), key=lambda s: s.source.name)
    assert [s.labels["complete"] for s in seqs] == [True, False]
    assert all(s.subject.startswith("unknown:") for s in seqs)


def test_layout_override_yaml(tmp_path: Path) -> None:
    from rehabpose.data.loaders.uco import UcoLoader

    make_videos(tmp_path, "uco", ["clips/s7_shoulder.mp4"])
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "uco.yaml").write_text(
        'regex: "clips/s(?P<subject>\\\\d+)_(?P<exercise>[a-z]+)\\\\.mp4$"\n'
        "exercise_map: {shoulder: shoulder_flexion}\nview: sagittal\n"
    )
    (s,) = UcoLoader(tmp_path / "raw", True, config_dir=cfg).sequences()
    assert (s.subject, s.exercise, s.exercise_id, s.view) == (
        "7",
        "shoulder",
        "shoulder_flexion",
        "sagittal",
    )
    (cfg / "uco.yaml").write_text("colour: blue\n")
    with pytest.raises(LayoutError, match="unknown keys"):
        UcoLoader(tmp_path / "raw", True, config_dir=cfg)


def test_own_loader(tmp_path: Path) -> None:
    make_own(tmp_path)
    loader = get_loader("own", tmp_path)  # no license acceptance needed for own data
    seqs = list(loader.sequences())
    assert [s.subject for s in seqs] == ["P01", "P02"]
    s = seqs[0]
    assert s.exercise_id == "shoulder_flexion" and s.joint_format == "mediapipe33"
    assert s.frames is not None and s.n_frames == len(s.frames)
    assert s.labels["reps"][0]["label"] == "correct"


def test_own_loader_missing_labels(tmp_path: Path) -> None:
    with pytest.raises(DatasetMissingError, match="rehabpose label"):
        list(get_loader("own", tmp_path).sequences())


def test_read_skeleton_rows_variants(tmp_path: Path) -> None:
    skel = kinect_skeleton([SynthPose(), SynthPose(elevation_deg=30)])
    for i, (state, header) in enumerate([(True, True), (False, False)]):
        p = write_kinect_txt(tmp_path / f"k{i}.txt", skel, with_state=state, header=header)
        np.testing.assert_allclose(read_skeleton_rows(p, 25), skel, atol=1e-5)
    # leading frame-index column
    p = tmp_path / "idx.csv"
    p.write_text(
        "\n".join(f"{i}," + ",".join(f"{v:.5f}" for v in fr.ravel()) for i, fr in enumerate(skel))
    )
    np.testing.assert_allclose(read_skeleton_rows(p, 25), skel, atol=1e-5)
    bad = tmp_path / "bad.txt"
    bad.write_text("1 2 3\n4 5 6\n")
    with pytest.raises(LayoutError, match="25x3"):
        read_skeleton_rows(bad, 25)
