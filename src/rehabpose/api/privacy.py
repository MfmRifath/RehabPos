"""User data export, deletion and retention of per-session files."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from sqlalchemy.engine import Engine
from sqlmodel import Session, col, select

from rehabpose import DISCLAIMER
from rehabpose.api.db import Event, Rep, SessionRow, Target, User, utcnow
from rehabpose.api.progress import as_utc
from rehabpose.api.state import AppState

log = logging.getLogger(__name__)


def session_dir(data_dir: Path, sid: str) -> Path:
    """Folder of a session's files (landmarks, annotated video, CSV/JSON outputs)."""
    return Path(data_dir) / "sessions" / sid


def session_files(data_dir: Path, sid: str) -> list[str]:
    """Names of the files stored for a session."""
    folder = session_dir(data_dir, sid)
    return sorted(p.name for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []


def delete_session_files(data_dir: Path, sid: str) -> int:
    """Remove a session's folder; returns the number of files removed."""
    n = len(session_files(data_dir, sid))
    shutil.rmtree(session_dir(data_dir, sid), ignore_errors=True)
    return n


def require_idle(state: AppState, session_ids: list[str]) -> None:
    """409 if a live connection or an upload job is using any of these sessions."""
    busy = state.jobs.active_session_ids() if state.jobs is not None else set()
    live = {sid for sid, n in state.live_connections.items() if n > 0}
    in_use = sorted((busy | live) & set(session_ids))
    if in_use:
        raise HTTPException(409, f"sessions in use, end them first: {in_use}")


def export_user(
    db: Session, user: User, data_dir: Path, now: datetime | None = None
) -> dict[str, Any]:
    """Everything stored about a user, as JSON-ready data (files are listed, not embedded)."""
    sessions = db.exec(
        select(SessionRow).where(SessionRow.user_id == user.id).order_by(col(SessionRow.started_at))
    ).all()
    targets = db.exec(select(Target).where(Target.user_id == user.id)).all()
    out: list[dict[str, Any]] = []
    for s in sessions:
        reps = db.exec(select(Rep).where(Rep.session_id == s.id).order_by(col(Rep.idx))).all()
        events = db.exec(select(Event).where(Event.session_id == s.id).order_by(col(Event.t))).all()
        out.append(
            {
                **s.model_dump(mode="json"),
                "reps": [r.model_dump(mode="json") for r in reps],
                "events": [e.model_dump(mode="json", exclude={"id"}) for e in events],
                "files": session_files(data_dir, s.id),
            }
        )
    return {
        "export_version": 1,
        "exported_at": (now or utcnow()).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
        "user": user.model_dump(mode="json"),
        "targets": [t.model_dump(mode="json") for t in targets],
        "sessions": out,
    }


def delete_user(db: Session, user: User, data_dir: Path) -> dict[str, int]:
    """Delete a user: database rows cascade through foreign keys, then session files go."""
    sids = list(db.exec(select(SessionRow.id).where(SessionRow.user_id == user.id)).all())
    counts = {
        "sessions": len(sids),
        "reps": len(db.exec(select(Rep).where(col(Rep.session_id).in_(sids))).all()),
        "events": len(db.exec(select(Event).where(col(Event.session_id).in_(sids))).all()),
        "targets": len(db.exec(select(Target).where(Target.user_id == user.id)).all()),
    }
    db.delete(user)
    db.commit()
    counts["files"] = sum(delete_session_files(data_dir, sid) for sid in sids)
    return counts


def purge_expired_outputs(
    engine: Engine, data_dir: Path, retention_days: int | None, now: datetime | None = None
) -> int:
    """Delete files of sessions that ended over `retention_days` ago; rows and reports stay.

    None keeps files until the session or user is deleted. Returns the sessions purged.
    """
    if retention_days is None:
        return 0
    cutoff = as_utc(now or utcnow()) - timedelta(days=retention_days)
    purged = 0
    with Session(engine) as db:
        rows = db.exec(select(SessionRow).where(col(SessionRow.ended_at).is_not(None))).all()
        for r in rows:
            expired = r.ended_at is not None and as_utc(r.ended_at) < cutoff
            if expired and session_dir(data_dir, r.id).is_dir():
                delete_session_files(data_dir, r.id)
                purged += 1
    if purged:
        log.info(
            "retention: removed files of %d sessions older than %d days", purged, retention_days
        )
    return purged
