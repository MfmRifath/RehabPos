"""Subject-wise splits (no leakage) and dataset jobs (stats, extract-all)."""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from rehabpose.core.landmarks import Frame
from rehabpose.data.extract import load_frames, read_meta
from rehabpose.data.loaders import get_loader
from rehabpose.data.splits import (
    Fold,
    LeakageError,
    SplitPlan,
    assert_no_leakage,
    fold_indices,
    load_plan,
    make_split_plan,
    save_plan,
)
from rehabpose.data.tasks import dataset_stats, extract_all
from tests.fixtures.datasets import make_intellirehab, make_own, make_uiprmd, make_videos

# --- splits ------------------------------------------------------------------------------------


@settings(max_examples=150, deadline=None)
@given(
    n=st.integers(2, 60),
    frac=st.floats(0.0, 0.5),
    k=st.integers(2, 10),
    seed=st.integers(0, 10_000),
)
def test_no_subject_overlap_any_plan(n: int, frac: float, k: int, seed: int) -> None:
    """Acceptance: no subject appears in more than one partition of any fold."""
    subjects = [f"S{i:03d}" for i in range(n)]
    plan = make_split_plan(subjects * 3, frac, k, seed)  # many rows per subject
    test = set(plan.test)
    vals: list[str] = []
    for f in plan.folds:
        train, val = set(f.train), set(f.val)
        assert not (train & val) and not (train & test) and not (val & test)
        assert train | val | test == set(subjects)
        vals.extend(f.val)
    assert sorted(vals) == sorted(set(subjects) - test)  # each non-test subject validated once


def test_loso_below_ten_and_group_kfold_above() -> None:
    small = make_split_plan([f"P{i}" for i in range(8)], test_fraction=0.25)
    assert small.method == "loso" and len(small.test) == 2
    assert all(len(f.val) == 1 for f in small.folds) and len(small.folds) == 6
    big = make_split_plan([f"P{i}" for i in range(40)], test_fraction=0.2, k=5)
    assert big.method == "group_kfold" and len(big.test) == 8 and len(big.folds) == 5
    assert {len(f.val) for f in big.folds} <= {6, 7}  # 32 subjects over 5 folds


def test_plan_deterministic_per_seed() -> None:
    subs = [f"P{i}" for i in range(20)]
    assert make_split_plan(subs, seed=3) == make_split_plan(list(reversed(subs)), seed=3)
    assert make_split_plan(subs, seed=3).test != make_split_plan(subs, seed=4).test


def test_rejects_unknown_subjects_and_too_few() -> None:
    with pytest.raises(ValueError, match="lack subject ids"):
        make_split_plan(["P1", "unknown:clip7"])
    with pytest.raises(ValueError, match="at least 2"):
        make_split_plan(["P1", "P1"])


def test_leakage_detected() -> None:
    bad = SplitPlan("loso", 0, ("P1",), (Fold(0, ("P1", "P2"), ("P3",)),))
    with pytest.raises(LeakageError, match="train/test"):
        assert_no_leakage(bad)


def test_fold_indices_rows() -> None:
    plan = make_split_plan([f"P{i}" for i in range(6)], test_fraction=0.2, seed=1)
    rows = [f"P{i}" for i in range(6) for _ in range(3)]
    tr, va, te = fold_indices(rows, plan, 0)
    assert sorted(np.r_[tr, va, te].tolist()) == list(range(18))
    subj = np.asarray(rows)
    assert not set(subj[tr]) & set(subj[va]) and not set(subj[tr]) & set(subj[te])
    with pytest.raises(ValueError, match="not in the plan"):
        fold_indices([*rows, "P99"], plan, 0)


def test_plan_yaml_roundtrip(tmp_path: Path) -> None:
    plan = make_split_plan([f"P{i}" for i in range(12)], seed=2)
    assert load_plan(save_plan(plan, tmp_path / "splits.yaml")) == plan
    text = (tmp_path / "splits.yaml").read_text()
    leaked = text.replace(f"- {plan.folds[0].val[0]}", f"- {plan.test[0]}", 1)
    (tmp_path / "bad.yaml").write_text(leaked)
    with pytest.raises(LeakageError):
        load_plan(tmp_path / "bad.yaml")


# --- dataset jobs ------------------------------------------------------------------------------


def test_stats_own_and_intellirehab(tmp_path: Path) -> None:
    make_own(tmp_path)
    s = dataset_stats(get_loader("own", tmp_path).sequences())
    assert (s["sequences"], s["subjects"], s["mapped_to_rehabpose"]) == (2, 2, 2)
    assert s["labels"] == {"rep:correct": 2} and s["frames_known"] > 0
    make_intellirehab(tmp_path)
    s = dataset_stats(get_loader("intellirehab", tmp_path, True).sequences())
    assert s["joint_formats"] == {"kinect_v2_25": 4}
    assert s["labels"] == {"correctness=1": 2, "correctness=2": 2}


def test_extract_all_skeleton_dataset(tmp_path: Path) -> None:
    make_intellirehab(tmp_path)
    loader = get_loader("intellirehab", tmp_path, True)
    out = tmp_path / "interim"
    assert extract_all(loader, out) == {"written": 4}
    files = sorted((out / "intellirehab").glob("*.parquet"))
    assert len(files) == 4
    frames = load_frames(files[0])
    assert len(frames) == 20 and all(f is not None for f in frames)
    meta = read_meta(files[0])
    assert meta["pose_backend"] == "kinect_v2_25" and meta["subject"] == "101"
    assert meta["license"] == "CC BY 4.0"
    assert extract_all(loader, out) == {"skipped_existing": 4}
    assert extract_all(loader, out, overwrite=True, limit=1) == {"written": 1}


def test_extract_all_unsupported_joint_format(tmp_path: Path) -> None:
    make_uiprmd(tmp_path)
    counts = extract_all(get_loader("uiprmd", tmp_path, True), tmp_path / "interim")
    assert counts == {"unsupported": 2}


class _FakeEstimator:
    instances = 0

    def __init__(self) -> None:
        type(self).instances += 1
        self.last_ts = -1

    def process(self, frame_bgr: NDArray[np.uint8], ts_ms: int) -> Frame | None:
        assert ts_ms > self.last_ts  # fresh estimator per video: timestamps restart cleanly
        self.last_ts = ts_ms
        return None

    def close(self) -> None:
        pass


def test_extract_all_videos_uses_fresh_estimator(tmp_path: Path) -> None:
    make_videos(tmp_path, "uco", ["S1/ex/a.mp4", "S2/ex/b.mp4"])
    _FakeEstimator.instances = 0
    counts = extract_all(get_loader("uco", tmp_path, True), tmp_path / "interim", _FakeEstimator)
    assert counts == {"written": 2} and _FakeEstimator.instances == 2
    assert (tmp_path / "interim" / "uco" / "S1" / "ex" / "a.parquet").is_file()
    with pytest.raises(ValueError, match="estimator factory"):
        extract_all(get_loader("uco", tmp_path, True), tmp_path / "other", None)
