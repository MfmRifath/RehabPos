"""ML data, augmentation, leakage, end-to-end experiment, export parity and online use."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from rehabpose.data.splits import fold_indices, make_split_plan
from rehabpose.ml.datasets import (
    REP_LABELS,
    STREAM_CODES,
    featurize,
    rep_windows,
    stream_codes_for,
    stream_windows,
    synthetic_corpus,
)
from rehabpose.ml.transforms import (
    AugmentConfig,
    augment,
    augment_batch,
    mirror,
    resample_time,
    time_warp,
)
from rehabpose.settings import get_settings

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def corpus() -> list:  # type: ignore[type-arg]
    return synthetic_corpus(n_subjects=6, sets_per_exercise=1, reps_per_set=6, seed=11)


# --- transforms ---------------------------------------------------------------------------------


def _window(t: int = 20) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.normal(size=(t, 13, 2)), np.ones((t, 13))


def test_mirror_is_an_involution_and_swaps_sides() -> None:
    p, v = _window()
    q, w = mirror(p, v)
    np.testing.assert_allclose(q[:, 4, 0], -p[:, 3, 0])  # right shoulder <- mirrored left
    q2, w2 = mirror(q, w)
    np.testing.assert_allclose(q2, p)
    np.testing.assert_allclose(w2, v)


def test_time_warp_keeps_length_and_endpoints() -> None:
    p, v = _window(30)
    q, w = time_warp(p, v, 0.2, np.random.default_rng(1))
    assert q.shape == p.shape and w.shape == v.shape
    np.testing.assert_allclose(q[0], p[0])
    np.testing.assert_allclose(q[-1], p[-1])
    assert not np.allclose(q, p)


def test_augment_rotation_scale_preserve_shape_and_invisible_zero() -> None:
    p, v = _window()
    v[:, 5] = 0.0
    p[:, 5] = 0.0
    cfg = AugmentConfig(jitter=0.0, scale=0.0, rotate_deg=10, time_warp=0.0, joint_dropout=0.0)
    q, w = augment(p, v, np.random.default_rng(2), cfg)
    lengths = np.linalg.norm(p.reshape(-1, 2), axis=1)
    np.testing.assert_allclose(np.linalg.norm(q.reshape(-1, 2), axis=1), lengths, atol=1e-12)
    assert not q[:, 5].any() and not w[:, 5].any()


def test_joint_dropout_and_batch_version() -> None:
    p, v = _window()
    cfg = AugmentConfig(jitter=0.0, scale=0.0, rotate_deg=0.0, time_warp=0.0, joint_dropout=1.0)
    q, w = augment(p, v, np.random.default_rng(3), cfg)
    assert not w.any() and not q.any()
    pb, vb = np.stack([p] * 4), np.stack([v] * 4)
    nothing = {
        "jitter": 0.0,
        "scale": 0.0,
        "rotate_deg": 0.0,
        "time_warp": 0.0,
        "joint_dropout": 0.0,
    }
    qb, wb = augment_batch(pb, vb, np.random.default_rng(3), AugmentConfig(mirror=1.0, **nothing))
    mp, mv = mirror(p, v)
    np.testing.assert_allclose(qb, np.stack([mp] * 4))  # batch mirror == single mirror
    np.testing.assert_allclose(wb, np.stack([mv] * 4))
    q0, _ = augment_batch(pb, vb, np.random.default_rng(0), AugmentConfig(**nothing))
    np.testing.assert_allclose(q0, pb)  # all strengths 0: identity
    qa, _ = augment_batch(pb, vb, np.random.default_rng(0), AugmentConfig())
    assert qa.shape == pb.shape and not np.allclose(qa[0], qa[1])  # per-sample randomness


def test_resample_time() -> None:
    x = np.arange(5, dtype=float)[:, None]
    np.testing.assert_allclose(resample_time(x, 9)[:, 0], np.linspace(0, 4, 9))
    np.testing.assert_allclose(resample_time(x, 5), x)


# --- corpus and windows -------------------------------------------------------------------------


def test_corpus_ground_truth(corpus: list) -> None:  # type: ignore[type-arg]
    assert len({s.subject for s in corpus}) == 6
    labels = {r.label for s in corpus for r in s.reps}
    assert labels == set(REP_LABELS)
    for s in corpus:
        assert s.frame_codes.shape == (len(s.frames), len(STREAM_CODES))
        assert s.angle_truth is not None and len(s.angle_truth) == len(s.frames)
        inside = np.zeros(len(s.frames), dtype=bool)
        for r in s.reps:
            assert 0 <= r.start < r.end < len(s.frames)
            if r.label == "compensated":
                assert len(r.codes) == 1
                inside[r.start : r.end + 1] = True
            else:
                assert r.codes == ()
        assert not s.frame_codes[~inside].any()  # codes only inside compensated reps


def test_windows_and_featurize(corpus: list) -> None:  # type: ignore[type-arg]
    rw = rep_windows(corpus)
    assert len(rw) == sum(len(s.reps) for s in corpus)
    assert featurize(rw, 64, idx=[0, 1]).shape == (2, 64, 45)
    sw = stream_windows(corpus, 30, 10)
    assert sw.classes == stream_codes_for(
        ["shoulder_flexion", "shoulder_abduction", "elbow_flexion"]
    )
    assert "ELBOW_AWAY" not in sw.classes
    assert featurize(sw, idx=range(3)).shape == (3, 30, 45)
    assert sw.y.shape == (len(sw), len(sw.classes))


@pytest.mark.parametrize("task", ["rep", "stream"])
def test_no_subject_leakage_across_partitions(corpus: list, task: str) -> None:  # type: ignore[type-arg]
    """Acceptance (leakage): every fold keeps train / val / test subjects disjoint."""
    ws = rep_windows(corpus) if task == "rep" else stream_windows(corpus, 45, 5)
    plan = make_split_plan(ws.groups.tolist(), 0.2, 5, 0)
    for k in range(len(plan.folds)):
        tr, va, te = fold_indices(ws.groups.tolist(), plan, k)
        s_tr, s_va, s_te = (set(ws.groups[i]) for i in (tr, va, te))
        assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)
        assert len(tr) + len(va) + len(te) == len(ws)


# --- end to end ---------------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("REHABPOSE_HOME", str(tmp_path))
    monkeypatch.setenv("REHABPOSE_EXERCISES_DIR", str(ROOT / "configs" / "exercises"))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def _quick(name: str, task: str, model: str) -> object:
    from rehabpose.ml.train import TrainConfig

    return TrainConfig.model_validate({
        "name": name, "task": task, "model": model,
        "data": {"n_subjects": 5, "sets_per_exercise": 1, "reps_per_set": 5, "seed": 3},
        "stream": {"length": 30, "stride": 10},
        "optim": {"epochs": 2, "patience": 2, "batch_size": 128},
        "seeds": [0], "cv": {"k": 2, "test_fraction": 0.2},
    })  # fmt: skip


def test_quick_experiment_rep_lgbm(home: Path) -> None:
    from rehabpose.ml.evaluate import run_experiment

    res = run_experiment(_quick("t_rep_lgbm", "rep", "lgbm"), use_mlflow=True)  # type: ignore[arg-type]
    text = res.report_path.read_text()
    assert "SYNTHETIC — not a result" in text and res.run_id and res.run_id in text
    assert "Baseline: rule engine" in text and "not a medical device" in text
    m = res.metrics
    assert set(m["test_seed0"]["per_class_f1"]) == set(REP_LABELS)
    assert m["engine"]["rep_count_mae"] >= 0 and "shoulder" in m["engine"]["angle_agreement"]
    assert (home / "mlruns" / "mlflow.db").is_file()
    assert res.bundle_dir is None  # LightGBM: no torch export
    assert any(p.name == "confusion.png" for p in res.figures)


def test_quick_experiment_stream_tcn_export_and_online(home: Path) -> None:
    """Export parity < 1e-4, latency <= 5 ms/window, online model fused into the engine."""
    from rehabpose.core.engine import SessionEngine
    from rehabpose.core.exercise import load_exercises
    from rehabpose.ml.evaluate import run_experiment
    from rehabpose.ml.online import TemporalFlagModel
    from rehabpose.synth import RepSpec, rep_sequence

    res = run_experiment(_quick("t_stream_tcn", "stream", "tcn"), use_mlflow=False)  # type: ignore[arg-type]
    m = res.metrics
    assert m["export_parity"]["torchscript_max_abs_diff"] < 1e-4
    assert m["export_parity"]["onnx_max_abs_diff"] < 1e-4
    print(f"latency ms: {m['latency_ms']}")
    assert m["latency_ms"]["cpu"]["p95"] <= 5.0
    assert res.bundle_dir is not None and (res.bundle_dir / "model.onnx").is_file()

    ex = load_exercises(ROOT / "configs" / "exercises")["shoulder_flexion"]
    comp = {"trunk_lean_deg": 18.0, "trunk_lean_plane": "sagittal"}
    frames = rep_sequence("shoulder_flexion", [RepSpec(150), RepSpec(150, compensation=comp)])
    for backend in ("onnx", "torchscript"):
        model = TemporalFlagModel(res.bundle_dir, "R", threshold=0.0, backend=backend)  # all codes
        eng = SessionEngine(ex, "R", flag_model=model)
        out = [eng.process(f) for f in frames if f is not None]
        model_flags = {f.code for fb in out for f in fb.flags if f.source == "model"}
        assert model_flags and model_flags <= set(model.codes)
        assert all(len({f.code for f in fb.flags}) == len(fb.flags) for fb in out)  # no duplicates
        assert any(e.source == "model" for e in eng.events)
    a = TemporalFlagModel(res.bundle_dir, "R", backend="onnx")
    b = TemporalFlagModel(res.bundle_dir, "R", backend="torchscript")
    for f in frames[:60]:
        a.update(0, f)
        b.update(0, f)
    assert a.last_proba and all(abs(a.last_proba[k] - b.last_proba[k]) < 1e-4 for k in a.codes)


def test_onnx_failure_does_not_lose_results(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: an export error once discarded a finished experiment."""
    import rehabpose.ml.export as export
    from rehabpose.ml.evaluate import run_experiment

    def boom(*a: object, **k: object) -> Path:
        raise RuntimeError("no translation")

    monkeypatch.setattr(export, "export_onnx", boom)
    res = run_experiment(_quick("t_export_fail", "stream", "tcn"), use_mlflow=False)  # type: ignore[arg-type]
    assert "no translation" in res.metrics["export_error"]
    assert "Export problem" in res.report_path.read_text()
    assert res.bundle_dir is not None and (res.bundle_dir / "model.ts").is_file()
    assert "onnx_max_abs_diff" not in res.metrics["export_parity"]


def test_online_rejects_rep_bundles(tmp_path: Path) -> None:
    import json

    from rehabpose.ml.online import TemporalFlagModel

    (tmp_path / "meta.json").write_text(json.dumps({"task": "rep"}))
    with pytest.raises(ValueError, match="needs 'stream'"):
        TemporalFlagModel(tmp_path, "R")


def test_config_files_load_and_validate() -> None:
    from rehabpose.ml.train import load_config

    paths = sorted((ROOT / "configs" / "train").glob("*.yaml"))
    assert len(paths) == 10
    for p in paths:
        cfg = load_config(p)
        assert cfg.data.source == "synthetic" and cfg.seeds == [0, 1, 2]


def test_cli_train_quick_no_mlflow(home: Path) -> None:
    from typer.testing import CliRunner

    from rehabpose.cli import app

    r = CliRunner().invoke(
        app,
        ["train", "--config", str(ROOT / "configs/train/rep_lgbm.yaml"), "--quick", "--no-mlflow"],
    )
    assert r.exit_code == 0, r.output
    assert "[SYNTHETIC] rep_lgbm" in r.output
    assert list((home / "reports" / "eval").glob("rep_lgbm-*.md"))
