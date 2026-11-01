"""Sessions: create, read, end, reports and the annotated video."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from sqlmodel import col, select

from rehabpose.api.db import Event, Rep, SessionRow, User, utcnow
from rehabpose.api.deps import DbDep, StateDep
from rehabpose.api.reports import build_report, render_pdf
from rehabpose.api.schemas import EventOut, RepOut, SessionCreate, SessionDetail, SessionOut
from rehabpose.api.state import get_session_row, model_version

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def create_session_row(body: SessionCreate, db: DbDep, state: StateDep) -> SessionRow:
    """Validate and insert a session (creating the pseudonymous user if needed)."""
    ex = state.exercise(body.exercise_id)
    view = body.view or ex.view
    if view not in ex.views:
        raise HTTPException(
            422, f"{ex.id} does not support the {view} view (allowed {list(ex.views)})"
        )
    if db.get(User, body.user_id) is None:
        db.add(User(id=body.user_id, display_name=body.display_name))
        db.flush()  # no ORM relationship: insert the user before the session (FK order)
    pose = (
        "browser-mediapipe" if body.source == "live" else f"mediapipe-{state.settings.pose_model}"
    )
    row = SessionRow(
        id=uuid.uuid4().hex[:16], user_id=body.user_id, exercise_id=ex.id, side=body.side,
        view=view, source=body.source, model_version=model_version(pose),
    )  # fmt: skip
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.post("", status_code=201)
def create_session(body: SessionCreate, db: DbDep, state: StateDep) -> SessionOut:
    """Create a session; connect to /ws/session/{id} next for live mode."""
    return SessionOut.model_validate(create_session_row(body, db, state))


@router.get("/{sid}")
def get_session(sid: str, db: DbDep) -> SessionDetail:
    """Session with reps and events."""
    row = get_session_row(db, sid)
    reps = db.exec(select(Rep).where(Rep.session_id == sid).order_by(col(Rep.idx))).all()
    events = db.exec(select(Event).where(Event.session_id == sid).order_by(col(Event.t))).all()
    return SessionDetail(
        **SessionOut.model_validate(row).model_dump(),
        reps=[RepOut.model_validate(r) for r in reps],
        events=[EventOut.model_validate(e) for e in events],
    )


@router.post("/{sid}/end")
def end_session(sid: str, db: DbDep) -> SessionOut:
    """Close a session and store its summary (idempotent)."""
    row = get_session_row(db, sid)
    if row.ended_at is None:
        reps = db.exec(select(Rep).where(Rep.session_id == sid)).all()
        row.rep_count = len(reps)
        row.mean_rom = round(sum(r.peak_angle for r in reps) / len(reps), 2) if reps else None
        correct = sum(r.label == "correct" for r in reps)
        row.quality_score = round(correct / len(reps), 4) if reps else None
        row.ended_at = utcnow()
        db.add(row)
        db.commit()
        db.refresh(row)
    return SessionOut.model_validate(row)


@router.get("/{sid}/report.json")
def report_json(sid: str, db: DbDep, state: StateDep) -> dict[str, Any]:
    """Structured session report (includes the disclaimer)."""
    row = get_session_row(db, sid)
    return build_report(db, row, state.exercise_for(db, row.user_id, row.exercise_id))


@router.get("/{sid}/report.pdf")
def report_pdf(sid: str, db: DbDep, state: StateDep) -> Response:
    """PDF session report."""
    report = report_json(sid, db, state)
    return Response(
        render_pdf(report),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="rehabpose_{sid}.pdf"'},
    )


@router.get("/{sid}/annotated.mp4")
def annotated_video(sid: str, db: DbDep, state: StateDep) -> FileResponse:
    """Annotated video of an upload session (404 until its job is done)."""
    get_session_row(db, sid)
    path = state.settings.data_dir / "sessions" / sid / "annotated.mp4"
    if not path.is_file():
        raise HTTPException(404, "no annotated video for this session")
    return FileResponse(path, media_type="video/mp4")
