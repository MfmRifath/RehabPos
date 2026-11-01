"""Ablations (SYNTHETIC): smoothing, 2D vs 3D angles, window length, pose backend, pretraining."""

from __future__ import annotations

import importlib.util
import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rehabpose import DISCLAIMER
from rehabpose.core.engine import EngineConfig, SessionEngine
from rehabpose.core.exercise import load_exercises
from rehabpose.ml.metrics import bland_altman
from rehabpose.settings import get_settings, project_root

log = logging.getLogger(__name__)
ABLATIONS = ("smoothing", "space", "window", "backend", "pretraining")


def _engine_eval(sets: list[Any], cfg: EngineConfig) -> dict[str, Any]:
    exercises = load_exercises(get_settings().exercises_dir)
    errs: list[int] = []
    pairs: dict[str, list[tuple[float, float]]] = {"shoulder": [], "elbow": []}
    for s in sets:
        ex = exercises[s.exercise_id]
        eng = SessionEngine(ex, s.side, config=cfg)
        for i, f in enumerate(s.frames):
            fb = (
                eng.process(f) if f is not None else eng.process_missing(round(i * 1000 / s.fps), i)
            )
            m = fb.angles.get(ex.primary_angle)
            if m is not None:
                pairs["elbow" if ex.primary_angle == "elbow_flexion" else "shoulder"].append(
                    (m, float(s.angle_truth[i]))
                )
        errs.append(abs(len(eng.reps) - len(s.reps)))
    out: dict[str, Any] = {"rep_count_mae": float(np.mean(errs))}
    for k, v in pairs.items():
        if v:
            a = np.asarray(v)
            out[f"{k}_angle"] = bland_altman(a[:, 0], a[:, 1])
    return out


def _corpus(quick: bool, noise_m: float = 0.0) -> list[Any]:
    from rehabpose.ml.datasets import synthetic_corpus

    return synthetic_corpus(3 if quick else 6, 1, 4 if quick else 8, seed=100, noise_m=noise_m)


def ablate_smoothing(quick: bool) -> dict[str, Any]:
    """Engine with One Euro smoothing on vs off."""
    sets = _corpus(quick)
    return {
        mode: _engine_eval(sets, EngineConfig(smoothing=mode == "on")) for mode in ("on", "off")
    }


def ablate_space(quick: bool) -> dict[str, Any]:
    """Engine angles from 2D pixels vs 3D world landmarks (world noise 1 cm added)."""
    sets = _corpus(quick, noise_m=0.01)
    return {
        "note": "synthetic world landmarks with 1 cm gaussian noise; real MediaPipe depth "
        "is far noisier",
        **{space: _engine_eval(sets, EngineConfig(space=space)) for space in ("2d", "3d")},
    }


def ablate_window(quick: bool) -> dict[str, Any]:
    """Stream TCN test macro-F1 for window lengths 30 / 45 / 60 (one seed, one split)."""
    from rehabpose.data.splits import fold_indices, make_split_plan
    from rehabpose.ml.datasets import stream_windows, synthetic_corpus
    from rehabpose.ml.metrics import stream_metrics
    from rehabpose.ml.train import (
        OptimConfig,
        StreamConfig,
        TrainConfig,
        fit,
        predict_proba,
        resolve_device,
    )

    sets = synthetic_corpus(6 if quick else 12, 1, 4 if quick else 8, seed=7)
    out: dict[str, Any] = {}
    for length in (30, 45, 60):
        cfg = TrainConfig(
            name=f"window{length}",
            task="stream",
            model="tcn",
            seeds=[0],
            stream=StreamConfig(length=length, stride=5),
            optim=OptimConfig(epochs=2 if quick else 10, batch_size=256, patience=3),
        )
        ws = stream_windows(sets, length, 5)
        plan = make_split_plan(ws.groups.tolist(), 0.25, 3, 0)
        tr, va, te = fold_indices(ws.groups.tolist(), plan, 0)
        dev = resolve_device("auto")
        res = fit(cfg, ws.subset(tr), ws.subset(va), 0, dev)
        test = ws.subset(te)
        m = stream_metrics(test.y, predict_proba(res, test, cfg, dev), test.classes, 0.5)
        out[str(length)] = {
            "test_macro_f1": m["macro_f1"],
            "mean_auroc": m["mean_auroc"],
            "best_epoch": res.best_epoch,
        }
    return out


def ablate_backend(quick: bool) -> dict[str, Any]:
    """Pose backends on a rendered mannequin abduction set: detection, ms/frame, angle error."""
    from rehabpose.core import angles
    from rehabpose.data.video_io import iter_frames
    from rehabpose.pose.mediapipe_pose import MediaPipePose
    from rehabpose.pose.model_files import ModelVariant, model_path
    from rehabpose.synth import RepSpec, rep_sequence_with_truth, write_frames_video

    tmp = project_root() / "reports" / "ablations" / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    reps = [RepSpec(130, duration_s=6.0, pause_s=1.0)] * (2 if quick else 3)
    frames, values, _, _ = rep_sequence_with_truth(
        "shoulder_abduction",
        reps,
        img_w=640,
        img_h=480,
        px_per_m=330,
        noise_px=0.0,
        rest_value=20.0,
    )
    video = write_frames_video(tmp / "abduction.mp4", frames)
    out: dict[str, Any] = {"video": "rendered cartoon mannequin (frontal abduction), not a human"}
    variants: tuple[ModelVariant, ...] = ("lite", "full", "heavy")
    for variant in variants:
        path = model_path(variant, get_settings().models_dir)
        if not path.is_file():
            out[f"mediapipe_{variant}"] = "model file missing"
            continue
        times, errs, detected = [], [], 0
        with MediaPipePose(path) as est:
            for idx, ts, img in iter_frames(video):
                t0 = time.perf_counter()
                fr = est.process(img, ts)
                times.append((time.perf_counter() - t0) * 1000)
                if fr is None:
                    continue
                detected += 1
                if min(fr.visibility[[12, 14, 24]]) >= 0.6:
                    errs.append(
                        float(angles.upper_arm_trunk_angle(fr.pixel_xy, "R")) - float(values[idx])
                    )
        e = np.asarray(errs)
        out[f"mediapipe_{variant}"] = {
            "frames": len(times),
            "detected": detected,
            "ms_per_frame_median": float(np.median(times)),
            "ms_per_frame_p95": float(np.percentile(times, 95)),
            "angle_mae_deg": float(np.abs(e).mean()) if e.size else None,
            "angle_frames": int(e.size),
        }
    for extra, module in (
        ("rtmpose (rtmlib)", "rtmlib"),
        ("yolo-pose (ultralytics, AGPL)", "ultralytics"),
    ):
        installed = importlib.util.find_spec(module) is not None
        out[extra] = (
            "installed but no adapter yet (phase 9)"
            if installed
            else "not installed (optional extra)"
        )
    (tmp / "abduction.mp4").unlink(missing_ok=True)
    tmp.rmdir()
    return out


def ablate_pretraining(quick: bool) -> dict[str, Any]:
    """Needs approved public skeleton datasets (none yet)."""
    return {
        "status": "NOT RUN",
        "reason": "no public skeleton dataset has been downloaded/approved (phase 4)",
    }


RUNNERS: dict[str, Callable[[bool], dict[str, Any]]] = {
    "smoothing": ablate_smoothing,
    "space": ablate_space,
    "window": ablate_window,
    "backend": ablate_backend,
    "pretraining": ablate_pretraining,
}


def run_ablations(
    names: list[str], quick: bool = False, out_dir: Path | None = None, command: str = ""
) -> Path:
    """Run ablations and write reports/ablations/ablations-<stamp>.md (+ .json)."""
    from rehabpose.ml.evaluate import _git_commit

    out_dir = Path(out_dir or project_root() / "reports" / "ablations")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for name in names:
        t0 = time.perf_counter()
        log.info("ablation %s ...", name)
        results[name] = RUNNERS[name](quick)
        results[name]["elapsed_s"] = round(time.perf_counter() - t0, 1)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    lines = [
        f"# Ablations ({stamp})",
        "",
        "> **SYNTHETIC — not a result.** Generated skeletons / a rendered cartoon only.",
        "",
        f"Commit: `{_git_commit()}`. Command: `{command or 'n/a'}`"
        + (" (quick mode: reduced sizes)" if quick else ""),
        "",
    ]
    for name, res in results.items():
        lines += [f"## {name}", "", "```json", json.dumps(res, indent=2, default=str), "```", ""]
    lines += ["---", "", DISCLAIMER, ""]
    path = out_dir / f"ablations-{stamp}.md"
    path.write_text("\n".join(lines))
    (out_dir / f"ablations-{stamp}.json").write_text(json.dumps(results, indent=2, default=str))
    return path
