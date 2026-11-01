"""Pose + engine and WebSocket latency benchmarks; writes reports/benchmarks.md (+ .json).

Every number comes from a run on this machine. Anything not run reads NOT YET MEASURED.
"""

from __future__ import annotations

import json
import multiprocessing
import platform
import resource
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from rehabpose import DISCLAIMER
from rehabpose.core.engine import SessionEngine
from rehabpose.core.exercise import load_exercises
from rehabpose.core.landmarks import Side
from rehabpose.data.video_io import iter_frames
from rehabpose.pose.base import PoseEstimator
from rehabpose.pose.model_files import ModelVariant, model_path
from rehabpose.settings import Settings, get_settings

NOT_MEASURED = "NOT YET MEASURED"
LIBRARIES = ("mediapipe", "torch", "numpy", "opencv-python", "fastapi", "uvicorn", "websockets")


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def machine_info() -> dict[str, Any]:
    """Hardware, OS and library versions of this machine."""
    info: dict[str, Any] = {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "libraries": {name: metadata.version(name) for name in LIBRARIES},
    }
    if sys.platform == "darwin":
        info |= {
            "hw_model": _run(["sysctl", "-n", "hw.model"]),
            "chip": _run(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "memory_gb": round(int(_run(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30),
            "macos": _run(["sw_vers", "-productVersion"]),
        }
    else:
        info["platform"] = platform.platform()
    return info


def peak_rss_mb() -> float:
    """Peak resident memory of this process (macOS reports bytes, Linux kilobytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (2**20 if sys.platform == "darwin" else 2**10), 1)


def _stats(values: Sequence[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"median": float("nan"), "p95": float("nan")}
    return {"median": round(float(np.median(a)), 3), "p95": round(float(np.percentile(a, 95)), 3)}


def bench_one(
    video: Path,
    estimator: PoseEstimator,
    exercise_id: str,
    side: Side,
    view: str | None,
    exercises_dir: Path,
) -> dict[str, Any]:
    """Per-frame pose ms, engine ms and end-to-end FPS over one video."""
    ex = load_exercises(exercises_dir)[exercise_id]
    engine = SessionEngine(ex, side, view)  # type: ignore[arg-type]
    pose_ms: list[float] = []
    engine_ms: list[float] = []
    detected = 0
    for idx, ts, img in iter_frames(video):
        t0 = time.perf_counter()
        frame = estimator.process(img, ts)
        t1 = time.perf_counter()
        if frame is None:
            engine.process_missing(ts, idx)
        else:
            detected += 1
            engine.process(frame)
        t2 = time.perf_counter()
        pose_ms.append((t1 - t0) * 1000)
        engine_ms.append((t2 - t1) * 1000)
    total_s = (sum(pose_ms) + sum(engine_ms)) / 1000
    return {
        "frames": len(pose_ms),
        "detected": detected,
        "reps": len(engine.reps),
        "pose_ms": _stats(pose_ms),
        "engine_ms": _stats(engine_ms),
        "fps": round(len(pose_ms) / total_s, 1) if total_s > 0 else None,
    }


def _bench_variant(
    video: str, variant: ModelVariant, exercise_id: str, side: Side, view: str | None
) -> dict[str, Any]:
    """Runs in a fresh process so that peak memory belongs to this model alone."""
    from rehabpose.pose import create_estimator

    settings = get_settings()
    estimator = create_estimator(variant, settings.models_dir)
    try:
        out = bench_one(Path(video), estimator, exercise_id, side, view, settings.exercises_dir)
    finally:
        estimator.close()
    out["peak_rss_mb"] = peak_rss_mb()
    return out


def bench_pose(
    video: Path, variants: Sequence[ModelVariant], exercise_id: str, side: Side, view: str | None
) -> dict[str, Any]:
    """Benchmark each model variant in its own process; missing models are reported, not fetched."""
    results: dict[str, Any] = {}
    context = multiprocessing.get_context("spawn")
    for variant in variants:
        if not model_path(variant, get_settings().models_dir).is_file():
            results[variant] = "model file missing"
            continue
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
            future = pool.submit(_bench_variant, str(video), variant, exercise_id, side, view)
            results[variant] = future.result()
    return results


def _server_settings(tmp: Path, base: Settings) -> Settings:
    return Settings(
        db_url=f"sqlite:///{tmp / 'bench.db'}",
        data_dir=tmp / "data",
        exercises_dir=base.exercises_dir,
        frontend_dist=tmp / "no-frontend",
        models_dir=tmp / "models",
    )


def bench_ws(reps: int = 6) -> dict[str, Any]:
    """Replay SYNTHETIC landmarks over a real local server; one frame at a time."""
    import httpx
    import uvicorn
    from websockets.sync.client import connect

    from rehabpose.api.main import create_app
    from rehabpose.synth import RepSpec, rep_sequence

    frames = [f for f in rep_sequence("shoulder_flexion", [RepSpec(150)] * reps) if f is not None]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = int(s.getsockname()[1])
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(_server_settings(Path(tmp), get_settings()))
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 15
        while not server.started and time.time() < deadline:
            time.sleep(0.02)
        server_ms: list[float] = []
        rtt_ms: list[float] = []
        reps_seen = 0
        try:
            r = httpx.post(
                f"http://127.0.0.1:{port}/api/sessions",
                json={"user_id": "bench", "exercise_id": "shoulder_flexion", "side": "R"},
            )
            sid = r.json()["id"]
            with connect(f"ws://127.0.0.1:{port}/ws/session/{sid}") as ws:
                for f in frames:
                    msg = {"v": 1, "type": "landmarks", "sid": sid, **f.to_dict()}
                    payload = json.dumps(msg)
                    t0 = time.perf_counter()
                    ws.send(payload)
                    while True:
                        reply = json.loads(ws.recv(timeout=10))
                        if reply["type"] == "rep_completed":
                            reps_seen += 1
                        if reply["type"] == "feedback":
                            break
                    rtt_ms.append((time.perf_counter() - t0) * 1000)
                    server_ms.append(float(reply["server_ms"]))
        finally:
            server.should_exit = True
            thread.join(timeout=10)
    return {
        "messages": len(frames),
        "reps": reps_seen,
        "server_ms": _stats(server_ms),
        "round_trip_ms": _stats(rtt_ms),
        "p99_server_ms": round(float(np.percentile(server_ms, 99)), 3) if server_ms else None,
    }


def render(data: dict[str, Any]) -> str:
    """Markdown for reports/benchmarks.md."""
    m = data["machine"]
    lines = [
        "# RehabPose benchmarks",
        "",
        f"Last updated {data['updated']}. Regenerate with the commands below; numbers are from "
        "this machine only.",
        "",
        "## Machine",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in m.items() if k != "libraries"]
    lines.append("- libraries: " + ", ".join(f"{k} {v}" for k, v in m["libraries"].items()))
    lines += ["", "## Pose + engine (Python, CPU delegate)", ""]
    pose = data.get("pose")
    if not pose:
        lines.append(NOT_MEASURED)
    else:
        kind = "SYNTHETIC rendered cartoon, not a human" if pose["synthetic"] else "user video"
        lines += [
            f"`{pose['command']}`",
            "",
            f"Video: {pose['video']} ({kind}). Peak memory is for a whole process, per model.",
            "",
            "| model | frames | detected | pose ms median / p95 | engine ms median / p95 "
            "| end-to-end FPS | peak MB |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, r in pose["models"].items():
            if isinstance(r, str):
                lines.append(f"| {name} | {r} | | | | | |")
                continue
            p, e = r["pose_ms"], r["engine_ms"]
            lines.append(
                f"| {name} | {r['frames']} | {r['detected']} | {p['median']} / {p['p95']} "
                f"| {e['median']} / {e['p95']} | {r['fps']} | {r['peak_rss_mb']} |"
            )
    lines += ["", "## Server latency (WebSocket replay, SYNTHETIC landmarks)", ""]
    ws = data.get("ws")
    if not ws:
        lines.append(NOT_MEASURED)
    else:
        s, t = ws["server_ms"], ws["round_trip_ms"]
        lines += [
            f"`{ws['command']}`",
            "",
            f"{ws['messages']} messages, one at a time, over localhost; {ws['reps']} reps counted.",
            "",
            "| measure | median ms | p95 ms |",
            "|---|---|---|",
            f"| server processing | {s['median']} | {s['p95']} |",
            f"| client round trip | {t['median']} | {t['p95']} |",
        ]
    lines += [
        "",
        "## Not measured here",
        "",
        f"- Live browser FPS on a real camera: {NOT_MEASURED} (read from the app's HUD).",
        f"- Docker image build and run: {NOT_MEASURED}.",
        f"- Accuracy on real people: {NOT_MEASURED}.",
        "",
        "---",
        "",
        DISCLAIMER,
        "",
    ]
    return "\n".join(lines)


def update_report(out: Path, section: str, value: dict[str, Any]) -> Path:
    """Merge one section into benchmarks.json and re-render benchmarks.md."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    store = out.with_suffix(".json")
    data: dict[str, Any] = json.loads(store.read_text()) if store.is_file() else {}
    data[section] = value
    data["machine"] = machine_info()
    data["updated"] = datetime.now(UTC).isoformat(timespec="seconds")
    store.write_text(json.dumps(data, indent=2) + "\n")
    out.write_text(render(data))
    return out
