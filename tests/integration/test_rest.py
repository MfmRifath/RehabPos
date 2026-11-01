"""REST endpoints: health, exercises, sessions, targets, progress, reports, upload jobs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rehabpose import DISCLAIMER
from rehabpose.settings import Settings
from tests.fixtures.synthetic import RepSpec, rep_sequence, write_frames_video
from tests.fixtures.ws_replay import landmark_messages
from tests.integration.conftest import TruthEstimator, new_session


def _live_set(client: TestClient, sid: str, reps: list[RepSpec]) -> None:
    frames = rep_sequence("shoulder_flexion", reps)
    with client.websocket_connect(f"/ws/session/{sid}") as ws:
        for m in landmark_messages(frames, sid):
            ws.send_text(json.dumps(m))
            while ws.receive_json()["type"] != "feedback":
                pass


def test_health_and_exercises(client: TestClient) -> None:
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["exercises"] == 5 and h["torch_device"] in ("mps", "cpu")
    ex = {e["id"]: e for e in client.get("/api/exercises").json()}
    flex = ex["shoulder_flexion"]
    assert flex["target"] == 150 and flex["view"] == "sagittal" and flex["fsm"]["start_below"] == 30
    assert {r["code"] for r in flex["rules"]} >= {"ELBOW_BEND", "TRUNK_LEAN", "TOO_FAST"}
    assert ex["elbow_flexion"]["allowed_views"] == ["sagittal", "frontal"]
    assert ex["shoulder_external_rotation"]["experimental"] is True


def test_session_validation(client: TestClient) -> None:
    base = {"user_id": "u1", "exercise_id": "shoulder_flexion", "side": "R"}
    assert client.post("/api/sessions", json={**base, "exercise_id": "squat"}).status_code == 404
    assert client.post("/api/sessions", json={**base, "view": "frontal"}).status_code == 422
    assert client.post("/api/sessions", json={**base, "side": "X"}).status_code == 422
    assert client.post("/api/sessions", json={**base, "user_id": "bad id!"}).status_code == 422
    assert client.get("/api/sessions/nope").status_code == 404
    r = client.post(
        "/api/sessions", json={**base, "exercise_id": "elbow_flexion", "view": "frontal"}
    )
    assert r.status_code == 201 and r.json()["view"] == "frontal"
    assert r.json()["model_version"].endswith("|browser-mediapipe|rules")


def test_end_is_idempotent_and_summarizes(client: TestClient) -> None:
    sid = new_session(client)
    _live_set(client, sid, [RepSpec(150), RepSpec(100)])
    first = client.post(f"/api/sessions/{sid}/end").json()
    assert first["rep_count"] == 2 and first["mean_rom"] == pytest.approx(125, abs=2)
    assert first["quality_score"] == 0.5
    assert client.post(f"/api/sessions/{sid}/end").json() == first


def test_targets_get_put_validate(client: TestClient) -> None:
    assert client.get("/api/users/u1/targets").status_code == 404
    new_session(client)
    assert client.get("/api/users/u1/targets").json() == {"user_id": "u1", "targets": []}
    body = [
        {
            "exercise_id": "shoulder_flexion",
            "target_rom": 120,
            "min_rep_s": 3,
            "thresholds": {"TRUNK_LEAN": 6},
        }
    ]
    out = client.put("/api/users/u1/targets", json=body).json()
    assert (
        out["targets"][0]["thresholds"] == {"TRUNK_LEAN": 6.0}
        and out["targets"][0]["target_rom"] == 120
    )
    body2 = [{"exercise_id": "shoulder_flexion", "target_rom": 130}]
    assert client.put("/api/users/u1/targets", json=body2).json()["targets"][0]["target_rom"] == 130
    bad = [{"exercise_id": "shoulder_flexion", "thresholds": {"NOPE": 1}}]
    assert client.put("/api/users/u1/targets", json=bad).status_code == 422
    assert (
        client.put(
            "/api/users/u1/targets", json=[{"exercise_id": "shoulder_flexion", "target_rom": 30}]
        ).status_code
        == 422
    )
    assert client.put("/api/users/u1/targets", json=[{"exercise_id": "squat"}]).status_code == 404


def test_progress_aggregates_ended_sessions(client: TestClient) -> None:
    for reps in ([RepSpec(150)], [RepSpec(140), RepSpec(150)]):
        sid = new_session(client)
        _live_set(client, sid, reps)
        client.post(f"/api/sessions/{sid}/end")
    new_session(client)  # open session: excluded
    p = client.get("/api/users/u1/progress").json()
    flex = p["exercises"]["shoulder_flexion"]
    assert flex["sessions"] == 2 and flex["total_reps"] == 3
    (day,) = flex["series"]
    assert day["sessions"] == 2 and day["reps"] == 3 and day["mean_rom"] is not None
    assert client.get("/api/users/ghost/progress").status_code == 404


def test_reports_json_and_pdf(client: TestClient, tmp_path: Path) -> None:
    sid = new_session(client)
    lean = {"trunk_lean_deg": 15.0, "trunk_lean_plane": "sagittal"}
    _live_set(client, sid, [RepSpec(150), RepSpec(150, compensation=lean), RepSpec(100)])
    client.post(f"/api/sessions/{sid}/end")
    rep = client.get(f"/api/sessions/{sid}/report.json").json()
    assert rep["disclaimer"] == DISCLAIMER
    assert rep["summary"]["rep_count"] == 3 and rep["summary"]["labels"] == {
        "correct": 1,
        "compensated": 1,
        "incomplete_rom": 1,
    }
    assert rep["flags_summary"] == {"TRUNK_LEAN": 1, "INCOMPLETE_ROM": 1}
    assert rep["target"] == {"rom": 150.0, "unit": "deg", "min_rep_s": 2.0}
    r = client.get(f"/api/sessions/{sid}/report.pdf")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF") and len(r.content) > 10_000
    (tmp_path / "report.pdf").write_bytes(r.content)
    print(f"report.pdf: {len(r.content)} bytes")
    empty = new_session(client)
    assert client.get(f"/api/sessions/{empty}/report.pdf").content.startswith(b"%PDF")


def _wait_job(client: TestClient, job_id: str, timeout: float = 60) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job timed out")


def test_upload_job_end_to_end(client: TestClient, settings: Settings, tmp_path: Path) -> None:
    reps = [RepSpec(150, duration_s=5.0), RepSpec(100, duration_s=8.0)]
    frames = rep_sequence("shoulder_flexion", reps, img_w=320, img_h=240, px_per_m=120)
    TruthEstimator.truth = {round(i * 1000 / 30): f for i, f in enumerate(frames) if f is not None}
    video = write_frames_video(tmp_path / "set.mp4", frames, style="stick")
    with video.open("rb") as fh:
        r = client.post(
            "/api/videos",
            files={"file": ("set.mp4", fh, "video/mp4")},
            data={"user_id": "u2", "exercise_id": "shoulder_flexion", "side": "R"},
        )
    assert r.status_code == 202, r.text
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "done", job["error"]
    assert job["progress"] == 1.0 and set(job["result"]) >= {"report_pdf", "annotated_video"}
    sid = job["session_id"]
    detail = client.get(f"/api/sessions/{sid}").json()
    assert detail["source"] == "upload" and detail["ended_at"] and detail["rep_count"] == 2
    # 5 s min-jerk rep over 150 deg peaks at ~112 deg/s (> 90): too_fast;
    # slow 100 deg rep (~47 deg/s peak) < 0.9 x 150: incomplete_rom
    assert [rp["label"] for rp in detail["reps"]] == ["too_fast", "incomplete_rom"]
    vid = client.get(job["result"]["annotated_video"])
    assert (
        vid.status_code == 200
        and vid.headers["content-type"] == "video/mp4"
        and len(vid.content) > 1000
    )
    assert not list((settings.data_dir / "uploads").iterdir())  # raw upload deleted (privacy)
    assert "mediapipe-full" in detail["model_version"]


def test_upload_rejections_and_missing_job(client: TestClient, tmp_path: Path) -> None:
    r = client.post(
        "/api/videos",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        data={"user_id": "u2", "exercise_id": "shoulder_flexion", "side": "R"},
    )
    assert r.status_code == 415
    assert client.get("/api/jobs/nope").status_code == 404
    sid = new_session(client)
    assert client.get(f"/api/sessions/{sid}/annotated.mp4").status_code == 404


def test_upload_job_failure_is_reported(client: TestClient, tmp_path: Path) -> None:
    r = client.post(
        "/api/videos",
        files={"file": ("broken.mp4", b"not really a video", "video/mp4")},
        data={"user_id": "u2", "exercise_id": "shoulder_flexion", "side": "R"},
    )
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "error" and "cannot open" in job["error"].lower()
