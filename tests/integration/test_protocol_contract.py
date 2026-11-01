"""Protocol v1 contract shared with the frontend (frontend/src/ws/protocol.fixtures.json).

Python builds the server messages and validates the client messages; Vitest parses the same
server messages and checks its builder reproduces the client messages. Regenerate with
REHABPOSE_UPDATE_FIXTURES=1 after an intentional protocol change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rehabpose.api.schemas import (
    CLIENT_MSG,
    CueOut,
    ErrorMsg,
    FeedbackMsg,
    FlagOut,
    Quality,
    RepCompletedMsg,
    RepInfo,
    RepOut,
    SetupStatusMsg,
)

FIXTURE = Path(__file__).resolve().parents[2] / "frontend" / "src" / "ws" / "protocol.fixtures.json"


def _client_messages() -> list[dict[str, Any]]:
    lm = [[round(i / 100, 2), 0.5, 0.0, 0.9] for i in range(33)]
    world = [[0.1, -0.2, 0.3, 0.9] for _ in range(33)]
    return [
        {
            "v": 1,
            "type": "landmarks",
            "sid": "s1",
            "frame": 3,
            "ts_ms": 1235,
            "img": {"w": 640, "h": 480},
            "lm": lm,
            "world": world,
            "client_infer_ms": 9.0,
        },
        {"v": 1, "type": "control", "action": "pause"},
    ]


def _server_messages() -> list[dict[str, Any]]:
    rep = RepOut(
        idx=0,
        t_start=1.0,
        t_peak=3.0,
        t_end=6.0,
        peak_angle=150.0,
        duration_s=5.0,
        peak_vel=70.0,
        similarity=0.9,
        label="compensated",
        flags=["TRUNK_LEAN"],
    )
    msgs = [
        FeedbackMsg(
            frame=3,
            ts_ms=1235,
            angles={"upper_arm_trunk_angle": 97.5, "trunk_tilt": None},
            phase="ascending",
            rep=RepInfo(count=1, last_peak=150.0, last_duration_s=5.0, last_similarity=0.9),
            flags=[FlagOut(code="TRUNK_LEAN", severity="warn", confidence=0.9, source="rule")],
            cue=CueOut(
                code="TRUNK_LEAN", text="Keep your trunk upright", speak=True, severity="warn"
            ),
            quality=Quality(visibility_ok=True, view_ok=True),
            server_ms=0.2,
            dropped=0,
        ),
        RepCompletedMsg(rep=rep),
        SetupStatusMsg(
            calibrated=False, visibility_ok=True, view_ok=False, hints=["Face the camera"]
        ),
        ErrorMsg(code="bad_message", message="lm: List should have at least 33 items"),
    ]
    return [json.loads(m.model_dump_json()) for m in msgs]


def test_protocol_fixture_matches_python_models() -> None:
    expected = {"client": _client_messages(), "server": _server_messages()}
    if os.environ.get("REHABPOSE_UPDATE_FIXTURES"):
        FIXTURE.write_text(json.dumps(expected, indent=1) + "\n")
    current = json.loads(FIXTURE.read_text())
    hint = "protocol drift: regenerate with REHABPOSE_UPDATE_FIXTURES=1 and update protocol.ts"
    assert current == expected, hint
    for msg in current["client"]:
        CLIENT_MSG.validate_python(msg)
