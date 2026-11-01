"""Subject-wise evaluation protocol, engine metrics, latency, export bundle, MLflow + report.

Everything produced from the synthetic corpus is labelled "SYNTHETIC — not a result".
"""

from __future__ import annotations

import json
import logging
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rehabpose import __version__
from rehabpose.core.engine import SessionEngine
from rehabpose.core.exercise import ExerciseConfig, load_exercises
from rehabpose.data.splits import fold_indices, make_split_plan
from rehabpose.ml.datasets import (
    REP_LABELS,
    LabelledSet,
    WindowSet,
    featurize,
    rep_windows,
    stream_windows,
    synthetic_corpus,
)
from rehabpose.ml.metrics import bland_altman, bootstrap_ci, rep_metrics, stream_metrics
from rehabpose.ml.train import FitResult, TrainConfig, fit, predict_proba, resolve_device
from rehabpose.settings import get_settings, project_root

log = logging.getLogger(__name__)
SYNTHETIC_BANNER = (
    "**SYNTHETIC — not a result.** Trained and evaluated on generated skeletons only."
)


@dataclass
class ExperimentResult:
    """Everything written by one experiment."""

    name: str
    metrics: dict[str, Any]
    report_path: Path
    run_id: str | None = None
    bundle_dir: Path | None = None
    figures: list[Path] = field(default_factory=list)


# --- data -------------------------------------------------------------------------------------


def build_sets(cfg: TrainConfig) -> list[LabelledSet]:
    """Labelled sets for the configured source (synthetic only until data is approved)."""
    if cfg.data.source != "synthetic":
        raise NotImplementedError(
            "own-data training needs labelled recordings (phase 4 tooling); none approved yet"
        )
    d = cfg.data
    return synthetic_corpus(d.n_subjects, d.sets_per_exercise, d.reps_per_set, d.exercises, d.seed)


def windows(cfg: TrainConfig, sets: list[LabelledSet]) -> WindowSet:
    """Rep or streaming windows for the task."""
    if cfg.task == "rep":
        return rep_windows(sets, cfg.space)
    return stream_windows(sets, cfg.stream.length, cfg.stream.stride, cfg.space)


def _metrics(cfg: TrainConfig, ws: WindowSet, proba: np.ndarray) -> dict[str, Any]:
    if cfg.task == "rep":
        return rep_metrics(ws.y, proba, ws.classes)
    return stream_metrics(ws.y, proba, ws.classes, cfg.threshold)


# --- engine baselines (deterministic pipeline) -------------------------------------------------


def engine_metrics(sets: list[LabelledSet], exercises: dict[str, ExerciseConfig]) -> dict[str, Any]:
    """Rule-engine labels (rep task baseline), rep-count MAE and angle agreement vs truth."""
    count_err: list[int] = []
    y_true: list[int] = []
    y_rule: list[int] = []
    angle_pairs: dict[str, list[tuple[float, float]]] = {"shoulder": [], "elbow": []}
    for s in sets:
        ex = exercises[s.exercise_id]
        eng = SessionEngine(ex, s.side)
        measured: list[float | None] = []
        for i, f in enumerate(s.frames):
            fb = (
                eng.process(f) if f is not None else eng.process_missing(round(i * 1000 / s.fps), i)
            )
            measured.append(fb.angles.get(ex.primary_angle))
        count_err.append(abs(len(eng.reps) - len(s.reps)))
        for r in s.reps:  # match each true rep to the engine rep with the largest overlap
            best, best_ov = None, 0.0
            for er in eng.reps:
                a, b = round(er.t_start * s.fps), round(er.t_end * s.fps)
                ov = min(b, r.end) - max(a, r.start)
                if ov > best_ov:
                    best, best_ov = er, ov
            y_true.append(REP_LABELS.index(r.label))
            y_rule.append(REP_LABELS.index(best.label) if best is not None else -1)
        if s.angle_truth is not None:
            key = "elbow" if ex.primary_angle == "elbow_flexion" else "shoulder"
            for m, t in zip(measured, s.angle_truth, strict=True):
                if m is not None:
                    angle_pairs[key].append((m, float(t)))
    yt, yr = np.asarray(y_true), np.asarray(y_rule)
    proba = np.zeros((len(yr), len(REP_LABELS)))
    proba[np.arange(len(yr))[yr >= 0], yr[yr >= 0]] = 1.0
    rules = rep_metrics(yt, proba, REP_LABELS)
    rules["missed_reps"] = int((yr < 0).sum())
    rules.pop("auroc_ovr", None)  # hard labels: AUROC is not meaningful
    angles = {}
    for k, pairs in angle_pairs.items():
        if pairs:
            arr = np.asarray(pairs)
            angles[k] = bland_altman(arr[:, 0], arr[:, 1])
    return {
        "rep_count_mae": float(np.mean(count_err)),
        "rep_count_exact": float(np.mean(np.asarray(count_err) == 0)),
        "rules_rep_labels": rules,
        "angle_agreement": angles,
        "angle_pairs": angle_pairs,
    }


# --- latency ----------------------------------------------------------------------------------


def _runner(fitres: FitResult, x: np.ndarray, dev: str) -> Callable[[], None]:
    """A zero-argument callable running one forward pass on `dev`."""
    if fitres.kind == "lgbm":
        from rehabpose.ml.models.lgbm import summary_features

        feats = summary_features(x)
        return lambda: fitres.model.predict_proba(feats)
    model = fitres.model.to(dev).eval()
    xt = torch.from_numpy(x).to(dev)

    def run() -> None:
        with torch.no_grad():
            model(xt)
            if dev == "mps":
                torch.mps.synchronize()

    return run


def latency_ms(
    fitres: FitResult, cfg: TrainConfig, devices: list[str], n: int = 200
) -> dict[str, dict[str, float]]:
    """Single-window inference latency (median / p95 ms), excluding feature extraction."""
    length = cfg.rep_length if cfg.task == "rep" else cfg.stream.length
    x = np.random.default_rng(0).normal(size=(1, length, cfg.n_features)).astype(np.float32)
    out: dict[str, dict[str, float]] = {}
    for dev in devices:
        if fitres.kind == "lgbm" and dev != "cpu":
            continue
        if fitres.kind == "torch" and dev == "mps" and not torch.backends.mps.is_available():
            continue
        run = _runner(fitres, x, dev)
        for _ in range(20):
            run()
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            run()
            times.append((time.perf_counter() - t0) * 1000)
        out[dev] = {"median": statistics.median(times), "p95": float(np.percentile(times, 95))}
    if fitres.kind == "torch":
        fitres.model.cpu()
    return out


# --- MLflow -----------------------------------------------------------------------------------


def mlflow_tracking(root: Path) -> str:
    """Local MLflow: SQLite backend + file artifacts under <root>/mlruns (ADR 0004)."""
    import mlflow

    (root / "mlruns").mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{root / 'mlruns' / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    name = "rehabpose"
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(root / "mlruns" / "artifacts").as_uri())
    mlflow.set_experiment(name)
    return uri


def _flat(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v if isinstance(v, int | float | str | bool) or v is None else json.dumps(v)
    return out


def _git_commit() -> str:
    """Short commit, with "-dirty" if tracked or untracked files differ from it."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


# --- protocol ---------------------------------------------------------------------------------


def run_experiment(
    cfg: TrainConfig,
    out_dir: Path | None = None,
    use_mlflow: bool = True,
    bundle_root: Path | None = None,
    command: str = "",
) -> ExperimentResult:
    """Subject-wise CV (folds x seeds) + held-out test + engine baselines + latency + export."""
    from rehabpose.ml import report

    settings = get_settings()
    root = project_root()
    out_dir = Path(out_dir or root / "reports" / "eval")
    bundle_root = Path(bundle_root or settings.models_dir / "temporal")
    device = resolve_device(cfg.device)
    t_start = time.perf_counter()

    sets = build_sets(cfg)
    ws = windows(cfg, sets)
    plan = make_split_plan([s.subject for s in sets], cfg.cv.test_fraction, cfg.cv.k, cfg.data.seed)
    log.info(
        "%s: %d windows, %d subjects, %s with %d folds, test %s",
        cfg.name,
        len(ws),
        len(plan.subjects),
        plan.method,
        len(plan.folds),
        list(plan.test),
    )

    cv_scores: list[float] = []
    for seed in cfg.seeds:
        for fold in range(len(plan.folds)):
            tr, va, _ = fold_indices(ws.groups.tolist(), plan, fold)
            res = fit(cfg, ws.subset(tr), ws.subset(va), seed, device)
            cv_scores.append(res.best_val)
            log.info(
                "  seed %d fold %d: val macro-F1 %.3f (epoch %d)",
                seed,
                fold,
                res.best_val,
                res.best_epoch,
            )

    # final models: fold 0's train subjects; its val subjects drive early stopping; the
    # held-out test subjects are touched only here, once per seed
    tr, va, te = fold_indices(ws.groups.tolist(), plan, 0)
    trainval = np.concatenate([tr, va])
    test_ws = ws.subset(te)
    finals: list[FitResult] = []
    test_by_seed: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        res = fit(cfg, ws.subset(tr), ws.subset(va), seed, device)
        finals.append(res)
        test_by_seed.append(_metrics(cfg, test_ws, predict_proba(res, test_ws, cfg, device)))
    ref = finals[0]
    proba0 = predict_proba(ref, test_ws, cfg, device)
    test0 = _metrics(cfg, test_ws, proba0)

    def f1_at(idx: np.ndarray) -> float:
        sub = test_ws.subset(idx.tolist())
        return float(_metrics(cfg, sub, proba0[idx])["macro_f1"])

    ci = bootstrap_ci(test_ws.groups, f1_at, n=500, seed=0)

    exercises = load_exercises(settings.exercises_dir)
    test_sets = [s for s in sets if s.subject in plan.test]
    eng = engine_metrics(test_sets, exercises)
    lat = latency_ms(ref, cfg, ["cpu", "mps"])

    metrics: dict[str, Any] = {
        "synthetic": True,
        "n_windows": len(ws),
        "n_trainval_windows": int(len(trainval)),
        "n_test_windows": len(test_ws),
        "class_counts": ws.class_counts(),
        "test_class_counts": test_ws.class_counts(),
        "split": {
            "method": plan.method,
            "folds": len(plan.folds),
            "test_subjects": list(plan.test),
        },
        "cv_val_macro_f1": {
            "mean": float(np.mean(cv_scores)),
            "std": float(np.std(cv_scores)),
            "n": len(cv_scores),
        },
        "test_macro_f1_by_seed": [m["macro_f1"] for m in test_by_seed],
        "test_seed0": test0,
        "test_seed0_macro_f1_ci95": ci,
        "engine": {k: v for k, v in eng.items() if k != "angle_pairs"},
        "latency_ms": lat,
        "device": str(device),
    }

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg.name}-{stamp}"
    bundle_dir = None
    if ref.kind == "torch":
        from rehabpose.ml.export import export_onnx, export_torchscript, parity

        bundle_dir = bundle_root / run_name
        length = cfg.rep_length if cfg.task == "rep" else cfg.stream.length
        example = torch.from_numpy(featurize(test_ws, cfg.rep_length, idx=[0]))
        ts = export_torchscript(ref.model, example, bundle_dir / "model.ts")
        onnx_path: Path | None = None
        try:
            onnx_path = export_onnx(ref.model, example, bundle_dir / "model.onnx")
        except Exception as exc:  # noqa: BLE001 - a failed export must not discard the results
            log.exception("ONNX export failed for %s", cfg.name)
            metrics["export_error"] = f"ONNX: {type(exc).__name__}: {str(exc)[:300]}"
        check = torch.from_numpy(
            featurize(test_ws, cfg.rep_length, idx=list(range(min(8, len(test_ws)))))
        )
        metrics["export_parity"] = parity(ref.model, ts, None, check)
        if onnx_path is not None:
            metrics["export_parity"].update(parity(ref.model, None, onnx_path, example))
        meta = {
            "name": cfg.name,
            "run_name": run_name,
            "task": cfg.task,
            "model": cfg.model,
            "classes": list(ws.classes),
            "space": cfg.space,
            "window": length,
            "stride": cfg.stream.stride,
            "threshold": cfg.threshold,
            "fps": ws.fps,
            "n_features": cfg.n_features,
            "synthetic": True,
            "rehabpose": __version__,
        }  # noqa: E501
        (bundle_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    figures = report.figures(out_dir, run_name, cfg, test0, eng["angle_pairs"])
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "rehabpose": __version__,
        "git": _git_commit(),
    }
    run_id = None
    if use_mlflow:
        import mlflow

        mlflow_tracking(root)
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            mlflow.set_tags({"synthetic": "true", "task": cfg.task, "model": cfg.model})
            mlflow.log_params({k: v for k, v in _flat(cfg.model_dump()).items()})
            mlflow.log_metrics(
                {
                    "cv_val_macro_f1_mean": metrics["cv_val_macro_f1"]["mean"],
                    "test_macro_f1_seed0": test0["macro_f1"],
                    "test_macro_f1_mean": float(np.mean(metrics["test_macro_f1_by_seed"])),
                    "rep_count_mae": eng["rep_count_mae"],
                    **{f"latency_{d}_p95_ms": v["p95"] for d, v in lat.items()},
                }
            )
            for fig in figures:
                mlflow.log_artifact(str(fig), "figures")
            if bundle_dir is not None:
                mlflow.log_artifacts(str(bundle_dir), "bundle")
    metrics["elapsed_s"] = round(time.perf_counter() - t_start, 1)
    report_path = report.write_report(
        out_dir, run_name, cfg, metrics, figures, run_id, env, command, bundle_dir
    )
    if use_mlflow and run_id is not None:
        import mlflow

        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(report_path), "report")
    return ExperimentResult(cfg.name, metrics, report_path, run_id, bundle_dir, figures)
