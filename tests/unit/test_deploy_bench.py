"""CI and Docker config checks (no daemon or network) and benchmark code with mocks."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from rehabpose.bench.benchmark import (
    NOT_MEASURED,
    bench_one,
    bench_ws,
    machine_info,
    peak_rss_mb,
    render,
    update_report,
)
from rehabpose.core.landmarks import Frame
from rehabpose.settings import Settings, get_settings
from tests.fixtures.synthetic import RepSpec, rep_sequence, write_frames_video

ROOT = Path(__file__).resolve().parents[2]
BANNED = ("nvidia", "cuda", "onnxruntime-gpu", "tensorrt")


def test_ci_workflow() -> None:
    wf = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    py = wf["jobs"]["python"]
    assert py["strategy"]["matrix"]["os"] == ["macos-latest", "ubuntu-latest"]
    runs = " ".join(s.get("run", "") for s in py["steps"])
    for needed in (
        "uv sync --frozen",
        "ruff check",
        "ruff format --check",
        "mypy src",
        "--cov-fail-under=80",
        "--cov=rehabpose.core",
    ):
        assert needed in runs, needed
    assert (
        next(s for s in py["steps"] if "setup-uv" in s.get("uses", ""))["with"]["python-version"]
        == "3.12"
    )
    fe = " ".join(s.get("run", "") for s in wf["jobs"]["frontend"]["steps"])
    for needed in ("npm ci", "npm run lint", "npm test", "npm run build"):
        assert needed in fe, needed


def test_dockerfile_and_compose() -> None:
    text = (ROOT / "docker/Dockerfile").read_text()
    assert text.count("FROM ") == 2 and "USER rehabpose" in text and "HEALTHCHECK" in text
    assert "uv sync --frozen --no-dev" in text and "/api/health" in text
    assert not any(b in text.lower() for b in BANNED)
    assert text.index("USER rehabpose") < text.index('CMD ["rehabpose", "serve"]')
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]["rehabpose"]
    assert compose["ports"] == ["127.0.0.1:8000:8000"] and "./data:/app/data" in compose["volumes"]
    ignored = (ROOT / ".dockerignore").read_text().splitlines()
    assert {"data", "mlruns", "*.task", "*.mp4"} <= set(ignored)


def test_linux_lock_has_cpu_torch_for_both_architectures() -> None:
    import tomllib

    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    cpu = next(p for p in lock["package"] if p["name"] == "torch" and "+cpu" in p["version"])
    urls = [w["url"] for w in cpu["wheels"] if "cp312" in w["url"]]
    assert any("x86_64" in u for u in urls) and any("aarch64" in u for u in urls)
    assert not any(p["name"].startswith("nvidia") for p in lock["package"])


@pytest.fixture
def models_client(tmp_path: Path) -> Iterator[tuple[TestClient, Settings]]:
    from rehabpose.api.main import create_app

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>ui</html>")
    s = Settings(
        db_url=f"sqlite:///{tmp_path / 't.db'}", data_dir=tmp_path / "data", frontend_dist=dist,
        models_dir=tmp_path / "models", exercises_dir=ROOT / "configs" / "exercises",
    )  # fmt: skip
    s.models_dir.mkdir()
    with TestClient(create_app(s)) as c:
        yield c, s


def test_pose_model_route_serves_only_pose_models(
    models_client: tuple[TestClient, Settings],
) -> None:
    c, s = models_client
    assert c.get("/models/pose_landmarker_full.task").status_code == 404
    (s.models_dir / "pose_landmarker_full.task").write_bytes(b"model")
    (s.models_dir / "secret.task").write_bytes(b"x")
    r = c.get("/models/pose_landmarker_full.task")
    assert r.status_code == 200 and r.content == b"model"
    assert c.get("/models/secret.task").status_code == 422
    assert c.get("/models/..%2Frehabpose.db").status_code in (404, 422)
    assert c.get("/").text == "<html>ui</html>"  # the static UI still works


class _Truth:
    def __init__(self, frames: list[Frame | None]) -> None:
        self.by_ts = {f.ts_ms: f for f in frames if f is not None}

    def process(self, frame_bgr: NDArray[np.uint8], ts_ms: int) -> Frame | None:
        return self.by_ts.get(ts_ms)

    def close(self) -> None:
        pass


def test_bench_one_with_mock_estimator(tmp_path: Path) -> None:
    frames = rep_sequence(
        "shoulder_flexion", [RepSpec(150, duration_s=3.0)], img_w=320, img_h=240, px_per_m=120
    )
    video = write_frames_video(tmp_path / "v.mp4", frames, style="stick")
    fixed = {round(i * 1000 / 30): f for i, f in enumerate(frames) if f is not None}
    est = _Truth(list(fixed.values()))
    est.by_ts = fixed
    r = bench_one(video, est, "shoulder_flexion", "R", None, ROOT / "configs" / "exercises")
    assert r["frames"] == len(frames) and r["detected"] == len(frames) and r["reps"] == 1
    assert r["pose_ms"]["median"] >= 0 and r["engine_ms"]["p95"] > 0 and r["fps"] > 0


def test_machine_info_and_memory() -> None:
    m = machine_info()
    assert m["python"] and set(m["libraries"]) >= {"mediapipe", "torch"}
    assert peak_rss_mb() > 10


def test_report_merges_sections_and_marks_unmeasured(tmp_path: Path) -> None:
    out = tmp_path / "reports" / "benchmarks.md"
    ws = {"command": "c", "messages": 2, "reps": 0, "server_ms": {"median": 0.1, "p95": 0.2},
          "round_trip_ms": {"median": 1.0, "p95": 2.0}}  # fmt: skip
    update_report(out, "ws", ws)
    text = out.read_text()
    assert NOT_MEASURED in text.split("## Server latency")[0]  # pose not run yet
    assert "| server processing | 0.1 | 0.2 |" in text and "Live browser FPS" in text
    pose = {"command": "p", "video": "x", "synthetic": True, "models": {
        "lite": {"frames": 3, "detected": 2, "reps": 0, "pose_ms": {"median": 1, "p95": 2},
                 "engine_ms": {"median": 3, "p95": 4}, "fps": 50.0, "peak_rss_mb": 100.0},
        "heavy": "model file missing"}}  # fmt: skip
    update_report(out, "pose", pose)
    text = out.read_text()
    assert (
        "| lite | 3 | 2 | 1 / 2 | 3 / 4 | 50.0 | 100.0 |" in text
        and "| heavy | model file missing" in text
    )
    assert "SYNTHETIC rendered cartoon" in text and "| server processing | 0.1 | 0.2 |" in text
    assert "not a medical device" in text
    data = json.loads(out.with_suffix(".json").read_text())
    assert render(data) == text


def test_bench_ws_replay_on_a_real_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REHABPOSE_EXERCISES_DIR", str(ROOT / "configs" / "exercises"))
    get_settings.cache_clear()
    try:
        r = bench_ws(reps=1)
    finally:
        get_settings.cache_clear()
    assert r["reps"] == 1 and r["messages"] > 100
    assert 0 < r["server_ms"]["median"] <= r["round_trip_ms"]["median"]


def test_cli_bench_needs_a_source() -> None:
    from typer.testing import CliRunner

    from rehabpose.cli import app

    assert CliRunner().invoke(app, ["bench"]).exit_code != 0
    assert CliRunner().invoke(app, ["bench", "--synthetic", "--models", "huge"]).exit_code != 0
