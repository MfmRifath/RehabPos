"""WebSocket protocol v1: replay synthetic landmark JSONL, backpressure, errors, control."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.fixtures.synthetic import RepSpec, rep_sequence
from tests.fixtures.ws_replay import landmark_messages, read_jsonl, write_jsonl
from tests.integration.conftest import new_session

LEAN = {"trunk_lean_deg": 15.0, "trunk_lean_plane": "sagittal"}


def replay(client: TestClient, sid: str, msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lock-step replay: send one frame, read until its feedback arrives."""
    out: list[dict[str, Any]] = []
    with client.websocket_connect(f"/ws/session/{sid}") as ws:
        for m in msgs:
            ws.send_text(json.dumps(m))
            while True:
                reply = ws.receive_json()
                out.append(reply)
                if reply["type"] in ("feedback", "error"):
                    break
    return out


@pytest.fixture
def flexion_jsonl(tmp_path: Path) -> Path:
    frames = rep_sequence(
        "shoulder_flexion",
        [RepSpec(150), RepSpec(150, compensation=LEAN), RepSpec(100), RepSpec(150)],
        dropouts=[(0.5, 0.6)],
    )
    return write_jsonl(tmp_path / "flexion.jsonl", landmark_messages(frames, "placeholder"))


def test_replay_counts_reps_flags_and_persists(client: TestClient, flexion_jsonl: Path) -> None:
    """Acceptance: WS replay -> rep count + flags; server p95 <= 10 ms per message."""
    sid = new_session(client)
    msgs = read_jsonl(flexion_jsonl, sid)
    out = replay(client, sid, msgs)

    feedback = [m for m in out if m["type"] == "feedback"]
    reps = [m["rep"] for m in out if m["type"] == "rep_completed"]
    assert len(feedback) == len(msgs)
    assert [r["label"] for r in reps] == ["correct", "compensated", "incomplete_rom", "correct"]
    assert reps[1]["flags"] == ["TRUNK_LEAN"] and reps[2]["flags"] == ["INCOMPLETE_ROM"]
    assert feedback[-1]["rep"]["count"] == 4 and feedback[-1]["dropped"] == 0
    assert {f["phase"] for f in feedback} == {"rest", "ascending", "peak", "descending"}
    assert any(f["cue"] and f["cue"]["code"] == "TRUNK_LEAN" for f in feedback)
    assert any(m["type"] == "setup_status" and m["calibrated"] for m in out)
    assert reps[0]["t_start"] > 0  # relative to the first frame

    ms = np.array([f["server_ms"] for f in feedback])
    p50, p95, p99 = np.percentile(ms, [50, 95, 99])
    print(
        f"server_ms over {ms.size} msgs: p50={p50:.3f} p95={p95:.3f} "
        f"p99={p99:.3f} max={ms.max():.3f}"
    )
    assert p95 <= 10.0

    detail = client.get(f"/api/sessions/{sid}").json()
    assert [r["label"] for r in detail["reps"]] == [r["label"] for r in reps]
    assert "TRUNK_LEAN" in {e["code"] for e in detail["events"]}
    ended = client.post(f"/api/sessions/{sid}/end").json()
    assert ended["rep_count"] == 4 and ended["quality_score"] == 0.5 and ended["ended_at"]


def test_latest_slot_unit() -> None:
    from rehabpose.api.ws import LatestSlot

    slot: LatestSlot[int] = LatestSlot()
    assert slot.take() is None
    for i in range(5):
        slot.put(i)
    assert slot.take() == 4 and slot.dropped == 4 and slot.take() is None
    slot.put(9)
    assert slot.take() == 9 and slot.dropped == 4


@pytest.fixture
def live_server(settings: Any) -> Iterator[str]:
    """Real uvicorn server in a thread (TestClient alternates strictly, hiding backpressure)."""
    import socket
    import threading
    import time

    import uvicorn

    from rehabpose.api.main import create_app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn did not start"
    yield f"127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def test_backpressure_processes_latest_frame(
    live_server: str, flexion_jsonl: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server that is behind skips stale frames but always answers the newest one."""
    import time

    import httpx
    from websockets.sync.client import connect

    from rehabpose.api.ws import LiveSession

    original = LiveSession.handle_landmarks

    def slow(self: LiveSession, msg: Any, t_recv: float) -> list[str]:
        time.sleep(0.005)  # simulate a server that is behind (blocks its event loop)
        return original(self, msg, t_recv)

    monkeypatch.setattr(LiveSession, "handle_landmarks", slow)
    r = httpx.post(
        f"http://{live_server}/api/sessions",
        json={"user_id": "u1", "exercise_id": "shoulder_flexion", "side": "R"},
    )
    sid = r.json()["id"]
    msgs = read_jsonl(flexion_jsonl, sid)[:200]
    last = msgs[-1]["frame"]
    with connect(f"ws://{live_server}/ws/session/{sid}") as ws:
        for m in msgs:  # burst without reading replies
            ws.send(json.dumps(m))
        feedback: list[dict[str, Any]] = []
        while not feedback or feedback[-1]["frame"] != last:
            reply = json.loads(ws.recv(timeout=10))
            if reply["type"] == "feedback":
                feedback.append(reply)
    frames = [f["frame"] for f in feedback]
    dropped = feedback[-1]["dropped"]
    print(f"burst of {len(msgs)} at 5 ms/frame: processed {len(feedback)}, dropped {dropped}")
    assert frames == sorted(frames) and frames[-1] == last  # newest frame always processed
    assert len(feedback) + dropped == len(msgs)
    assert dropped > 0


def _next_error(ws: Any) -> dict[str, Any]:
    while (reply := ws.receive_json())["type"] != "error":
        pass
    return reply


def test_errors_keep_connection_open(client: TestClient, flexion_jsonl: Path) -> None:
    sid = new_session(client)
    good = read_jsonl(flexion_jsonl, sid)[:2]
    with client.websocket_connect(f"/ws/session/{sid}") as ws:
        ws.send_text("not json")
        assert _next_error(ws)["code"] == "bad_message"
        ws.send_text(json.dumps(dict(good[0], lm=good[0]["lm"][:10])))
        err = _next_error(ws)
        assert err["code"] == "bad_message" and "lm" in err["message"]
        ws.send_text(json.dumps(dict(good[0], v=2)))
        assert _next_error(ws)["code"] == "bad_message"
        ws.send_text(json.dumps(dict(good[0], sid="other")))
        assert _next_error(ws)["code"] == "sid_mismatch"
        ws.send_text(json.dumps(good[1]))
        assert ws.receive_json()["type"] == "feedback"  # connection still usable
        ws.send_text(json.dumps(good[0]))  # older timestamp
        assert _next_error(ws)["code"] == "non_monotonic_ts"


def test_unknown_or_ended_session_is_rejected(client: TestClient) -> None:
    with client.websocket_connect("/ws/session/nope") as ws:
        assert ws.receive_json() == {
            "v": 1,
            "type": "error",
            "code": "unknown_session",
            "message": "session nope",
        }
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 4404
    sid = new_session(client)
    client.post(f"/api/sessions/{sid}/end")
    with client.websocket_connect(f"/ws/session/{sid}") as ws:
        assert ws.receive_json()["code"] == "session_ended"


def test_pause_stops_counting_and_start_set_resumes(client: TestClient, tmp_path: Path) -> None:
    sid = new_session(client)
    frames = rep_sequence("shoulder_flexion", [RepSpec(150)] * 2)
    msgs = landmark_messages(frames, sid)
    half = len(msgs) // 2
    with client.websocket_connect(f"/ws/session/{sid}") as ws:
        ws.send_text(json.dumps({"v": 1, "type": "control", "action": "pause"}))
        counts = []
        for i, m in enumerate(msgs):
            if i == half:
                ws.send_text(json.dumps({"v": 1, "type": "control", "action": "start_set"}))
            ws.send_text(json.dumps(m))
            while (reply := ws.receive_json())["type"] != "feedback":
                pass
            counts.append(reply["rep"]["count"])
            if i < half:
                assert reply["phase"] == "rest" and reply["cue"] is None
    assert counts[half - 1] == 0 and counts[-1] == 1  # only the rep after start_set counts


def test_user_targets_change_live_labels(client: TestClient) -> None:
    sid0 = new_session(client)  # creates user u1
    client.post(f"/api/sessions/{sid0}/end")
    r = client.put(
        "/api/users/u1/targets", json=[{"exercise_id": "shoulder_flexion", "target_rom": 110}]
    )
    assert r.status_code == 200
    sid = new_session(client)
    frames = rep_sequence("shoulder_flexion", [RepSpec(110)])
    out = replay(client, sid, landmark_messages(frames, sid))
    (rep,) = [m["rep"] for m in out if m["type"] == "rep_completed"]
    assert rep["label"] == "correct"  # with the default target 150 this would be incomplete_rom
