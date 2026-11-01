"""Progress aggregation, streaks, badges, export, delete cascade, retention, final report."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader
from sqlmodel import Session, select

from rehabpose import DISCLAIMER
from rehabpose.api.db import Event, Rep, SessionRow, Target, User, make_engine
from rehabpose.api.privacy import purge_expired_outputs
from rehabpose.api.progress import aggregate, current_streak, local_day, longest_streak
from rehabpose.settings import Settings
from tests.fixtures.synthetic import RepSpec, rep_sequence
from tests.fixtures.ws_replay import landmark_messages
from tests.integration.conftest import new_session

NOW = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
FLEX, ABD = "shoulder_flexion", "shoulder_abduction"


def row(
    day_offset: int,
    reps: int = 10,
    rom: float | None = 120.0,
    quality: float | None = 1.0,
    exercise: str = FLEX,
    hour: int = 10,
) -> SessionRow:
    start = (NOW + timedelta(days=day_offset)).replace(hour=hour)
    return SessionRow(
        id=f"s{day_offset}{exercise}{hour}", user_id="u", exercise_id=exercise, side="R",
        view="sagittal", source="live", started_at=start, ended_at=start, rep_count=reps,
        mean_rom=rom if reps else None, quality_score=quality if reps else None,
    )  # fmt: skip


def d(*offsets: int) -> list[date]:
    return [date(2026, 3, 10) + timedelta(days=o) for o in offsets]


# --- pure logic ---------------------------------------------------------------------------------


def test_streaks() -> None:
    today = date(2026, 3, 10)
    assert current_streak(d(0, -1, -2), today) == 3
    assert current_streak(d(-1, -2), today) == 2  # yesterday still counts
    assert current_streak(d(-2, -3), today) == 0
    assert current_streak([], today) == 0
    assert longest_streak(d(-9, -8, -7, -3, -2)) == 3
    assert longest_streak(d(0, 0, 0)) == 1  # duplicates are one day
    assert longest_streak([]) == 0


def test_local_day_uses_the_callers_time_zone() -> None:
    late = datetime(2026, 3, 9, 20, 0, tzinfo=UTC)  # 01:30 on the 10th at UTC+5:30
    assert local_day(late, 0) == date(2026, 3, 9)
    assert local_day(late, 330) == date(2026, 3, 10)
    assert local_day(late.replace(tzinfo=None), 330) == date(2026, 3, 10)  # naive = UTC


def test_aggregate_daily_values_are_rep_weighted() -> None:
    p = aggregate("u", [row(0, 30, 100.0, 1.0), row(0, 10, 140.0, 0.0, hour=11)], NOW)
    (day,) = p["exercises"][FLEX]["series"]
    assert day == {
        "date": "2026-03-10",
        "sessions": 2,
        "reps": 40,
        "mean_rom": 110.0,
        "quality": 0.75,
    }
    assert p["overall"] == {
        "sessions": 2, "total_reps": 40, "active_days": 1, "current_streak": 1, "longest_streak": 1,
    }  # fmt: skip


def test_aggregate_separates_exercises_and_totals_overall() -> None:
    p = aggregate("u", [row(-1, 5, exercise=FLEX), row(0, 7, exercise=ABD)], NOW)
    assert set(p["exercises"]) == {FLEX, ABD}
    assert p["exercises"][FLEX]["total_reps"] == 5 and p["exercises"][ABD]["total_reps"] == 7
    assert p["overall"]["total_reps"] == 12 and p["overall"]["current_streak"] == 2


def test_session_without_reps_does_not_extend_a_streak() -> None:
    p = aggregate("u", [row(-1, 5), row(0, 0)], NOW)
    assert p["overall"]["active_days"] == 1 and p["overall"]["sessions"] == 2
    assert p["exercises"][FLEX]["series"][1]["mean_rom"] is None


def test_empty_user() -> None:
    p = aggregate("u", [], NOW)
    assert p["exercises"] == {} and p["overall"]["total_reps"] == 0
    assert not any(b["earned"] for b in p["badges"])


def _badges(rows: list[SessionRow]) -> dict[str, bool]:
    return {b["id"]: b["earned"] for b in aggregate("u", rows, NOW)["badges"]}


@pytest.mark.parametrize(("reps", "earned"), [(49, False), (50, True)])
def test_badge_first_50(reps: int, earned: bool) -> None:
    assert _badges([row(0, reps)])["first_50"] is earned


def test_badge_7_day_streak_stays_earned_after_the_streak_ends() -> None:
    assert not _badges([row(-i, 1) for i in range(6)])["streak_7"]
    assert _badges([row(-i, 1) for i in range(7)])["streak_7"]
    old = [row(-30 - i, 1) for i in range(7)]
    p = aggregate("u", old, NOW)
    assert p["overall"]["current_streak"] == 0 and p["overall"]["longest_streak"] == 7
    assert _badges(old)["streak_7"]


def test_badge_rom_personal_best() -> None:
    assert not _badges([row(0, 5, 120.0)])["rom_pb"]  # first session has nothing to beat
    assert not _badges([row(-1, 5, 120.0), row(0, 5, 110.0)])["rom_pb"]
    assert _badges([row(-1, 5, 120.0), row(0, 5, 130.0)])["rom_pb"]
    # stays earned when a later session is lower; each exercise has its own best
    assert _badges([row(-2, 5, 120.0), row(-1, 5, 130.0), row(0, 5, 100.0)])["rom_pb"]
    assert not _badges([row(-1, 5, 150.0, exercise=FLEX), row(0, 5, 100.0, exercise=ABD)])["rom_pb"]
    pb = aggregate("u", [row(-1, 5, 120.0), row(0, 5, 130.0)], NOW)["exercises"][FLEX]
    assert pb["rom_personal_best"] == {"value": 130.0, "date": "2026-03-10"}


# --- API ----------------------------------------------------------------------------------------


def _live_set(client: TestClient, sid: str, reps: list[RepSpec]) -> None:
    frames = rep_sequence(FLEX, reps)
    with client.websocket_connect(f"/ws/session/{sid}") as ws:
        for m in landmark_messages(frames, sid):
            ws.send_text(json.dumps(m))
            while ws.receive_json()["type"] != "feedback":
                pass


def _finished_session(client: TestClient, settings: Settings) -> str:
    sid = new_session(client)
    lean = {"trunk_lean_deg": 15.0, "trunk_lean_plane": "sagittal"}
    _live_set(client, sid, [RepSpec(150), RepSpec(150, compensation=lean)])
    client.post(f"/api/sessions/{sid}/end")
    folder = settings.data_dir / "sessions" / sid
    folder.mkdir(parents=True)
    (folder / "annotated.mp4").write_bytes(b"video")
    return sid


def test_progress_endpoint(client: TestClient, settings: Settings) -> None:
    _finished_session(client, settings)
    p = client.get("/api/users/u1/progress", params={"tz_offset_minutes": 330}).json()
    assert p["tz_offset_minutes"] == 330 and p["overall"]["total_reps"] == 2
    assert p["overall"]["current_streak"] == 1 and {b["id"] for b in p["badges"]} == {
        "first_50",
        "streak_7",
        "rom_pb",
    }
    assert (
        client.get("/api/users/u1/progress", params={"tz_offset_minutes": 9999}).status_code == 422
    )


def test_export_contains_everything(client: TestClient, settings: Settings) -> None:
    sid = _finished_session(client, settings)
    client.put("/api/users/u1/targets", json=[{"exercise_id": FLEX, "target_rom": 140}])
    r = client.get("/api/users/u1/export")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    data = r.json()
    assert data["user"]["id"] == "u1" and data["disclaimer"] == DISCLAIMER
    assert data["targets"][0]["target_rom"] == 140
    (s,) = data["sessions"]
    assert s["id"] == sid and len(s["reps"]) == 2 and s["files"] == ["annotated.mp4"]
    assert "TRUNK_LEAN" in {e["code"] for e in s["events"]} and s["reps"][1]["flags"] == [
        "TRUNK_LEAN"
    ]
    assert client.get("/api/users/ghost/export").status_code == 404


def test_delete_user_cascades_and_removes_files(client: TestClient, settings: Settings) -> None:
    """Acceptance: deleting a user removes all their rows and files, and nobody else's."""
    sid = _finished_session(client, settings)
    client.put("/api/users/u1/targets", json=[{"exercise_id": FLEX, "target_rom": 140}])
    other = client.post(
        "/api/sessions", json={"user_id": "u2", "exercise_id": FLEX, "side": "L"}
    ).json()["id"]
    r = client.delete("/api/users/u1")
    assert r.status_code == 200
    assert r.json()["deleted"] == {"sessions": 1, "reps": 2, "events": 1, "targets": 1, "files": 1}
    assert not (settings.data_dir / "sessions" / sid).exists()
    with Session(make_engine(settings.db_url)) as db:
        assert db.get(User, "u1") is None
        assert db.exec(select(SessionRow).where(SessionRow.user_id == "u1")).all() == []
        assert db.exec(select(Target).where(Target.user_id == "u1")).all() == []
        assert db.exec(select(Rep).where(Rep.session_id == sid)).all() == []
        assert db.exec(select(Event).where(Event.session_id == sid)).all() == []
        assert db.get(SessionRow, other) is not None  # other users untouched
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    assert client.delete("/api/users/u1").status_code == 404


def test_delete_session(client: TestClient, settings: Settings) -> None:
    sid = _finished_session(client, settings)
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    assert not (settings.data_dir / "sessions" / sid).exists()
    assert client.get("/api/users/u1/progress").json()["overall"]["sessions"] == 0
    assert client.delete(f"/api/sessions/{sid}").status_code == 404


def test_delete_refused_while_live_session_is_connected(client: TestClient) -> None:
    sid = new_session(client)
    with client.websocket_connect(f"/ws/session/{sid}") as ws:
        frame = landmark_messages(rep_sequence(FLEX, [RepSpec(150)])[:1], sid)[0]
        ws.send_text(json.dumps(frame))
        assert ws.receive_json()["type"] == "feedback"
        assert client.delete("/api/users/u1").status_code == 409
        assert client.delete(f"/api/sessions/{sid}").status_code == 409
    assert client.delete(f"/api/sessions/{sid}").status_code == 204  # released on disconnect


def test_delete_refused_while_a_job_is_active(client: TestClient) -> None:
    sid = new_session(client)
    jobs = client.app.state.rp.jobs  # type: ignore[attr-defined]
    from rehabpose.api.jobs import Job

    jobs._jobs["j1"] = Job(id="j1", session_id=sid, status="running")
    assert client.delete("/api/users/u1").status_code == 409
    jobs._jobs["j1"].status = "done"
    assert client.delete("/api/users/u1").status_code == 200


def test_retention_purges_only_old_session_files(client: TestClient, settings: Settings) -> None:
    sid = _finished_session(client, settings)
    engine = make_engine(settings.db_url)
    later = datetime.now(UTC) + timedelta(days=10)
    assert purge_expired_outputs(engine, settings.data_dir, None, later) == 0
    assert purge_expired_outputs(engine, settings.data_dir, 30, later) == 0
    assert (settings.data_dir / "sessions" / sid / "annotated.mp4").exists()
    assert purge_expired_outputs(engine, settings.data_dir, 5, later) == 1
    assert not (settings.data_dir / "sessions" / sid).exists()
    assert client.get(f"/api/sessions/{sid}").json()["rep_count"] == 2  # data and report stay


def _pdf_text(content: bytes) -> str:
    return " ".join(" ".join(p.extract_text() for p in PdfReader(BytesIO(content)).pages).split())


def test_final_pdf_content(client: TestClient, settings: Settings) -> None:
    sid = _finished_session(client, settings)
    r = client.get(f"/api/sessions/{sid}/report.pdf")
    text = _pdf_text(r.content)
    assert " ".join(DISCLAIMER.split()) in text
    for expected in (
        "DTW similarity",
        "synthetic reference curve",
        "TRUNK_LEAN",
        "Flags",
        "Model/version",
        "Active time",
    ):
        assert expected in text, expected
    rep = client.get(f"/api/sessions/{sid}/report.json").json()
    assert rep["reference_rep"] == "synthetic" and rep["summary"]["active_duration_s"] > 0
    assert rep["note"] is None and rep["events"][0]["code"] == "TRUNK_LEAN"


def test_sample_report_is_generated_and_labelled(tmp_path: Path) -> None:
    from rehabpose.api.samples import SAMPLE_NOTE, write_sample_report
    from rehabpose.core.exercise import load_exercises

    root = Path(__file__).resolve().parents[2]
    ex = load_exercises(root / "configs" / "exercises")[FLEX]
    j, pdf = write_sample_report(ex, tmp_path)
    data = json.loads(j.read_text())
    assert data["note"] == SAMPLE_NOTE and data["summary"]["rep_count"] == 6
    assert data["summary"]["labels"] == {
        "correct": 2,
        "compensated": 2,
        "incomplete_rom": 1,
        "too_fast": 1,
    }
    assert "SYNTHETIC sample" in _pdf_text(pdf.read_bytes())
    committed = root / "reports" / "samples"
    assert (committed / "sample_report.pdf").is_file() and (
        committed / "sample_report.json"
    ).is_file()
