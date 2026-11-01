"""SQLite persistence (SQLModel): users, targets, sessions, reps, events."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, event
from sqlalchemy.engine import Engine
from sqlmodel import Field, Session, SQLModel, create_engine

from rehabpose.core.engine import RepRecord


def utcnow() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(UTC)


class User(SQLModel, table=True):
    """Pseudonymous user (no real names or emails required)."""

    id: str = Field(primary_key=True)
    display_name: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class Target(SQLModel, table=True):
    """Per-user, per-exercise overrides set by a therapist."""

    user_id: str = Field(foreign_key="user.id", primary_key=True, ondelete="CASCADE")
    exercise_id: str = Field(primary_key=True)
    target_rom: float | None = None
    min_rep_s: float | None = None
    thresholds: dict[str, float] = Field(default_factory=dict, sa_type=JSON)


class SessionRow(SQLModel, table=True):
    """One exercise session (live or upload)."""

    __tablename__ = "session"

    id: str = Field(primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True, ondelete="CASCADE")
    exercise_id: str
    side: str
    view: str
    source: str  # live | upload
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None
    rep_count: int = 0
    mean_rom: float | None = None
    quality_score: float | None = None
    model_version: str = ""


class Rep(SQLModel, table=True):
    """One completed repetition (times s, angles deg or primary units)."""

    session_id: str = Field(foreign_key="session.id", primary_key=True, ondelete="CASCADE")
    idx: int = Field(primary_key=True)
    t_start: float
    t_peak: float
    t_end: float
    peak_angle: float
    duration_s: float
    peak_vel: float
    similarity: float
    label: str
    flags: list[str] = Field(default_factory=list, sa_type=JSON)


class Event(SQLModel, table=True):
    """A flag raised during a session (t in s since the session's first frame)."""

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="session.id", index=True, ondelete="CASCADE")
    t: float
    code: str
    severity: str
    confidence: float
    source: str


def rep_row(sid: str, r: RepRecord, t0: float = 0.0) -> Rep:
    """DB row for an engine rep (times relative to the session's first frame, seconds)."""
    return Rep(
        session_id=sid,
        idx=r.idx,
        t_start=round(r.t_start - t0, 3),
        t_peak=round(r.t_peak - t0, 3),
        t_end=round(r.t_end - t0, 3),
        peak_angle=r.peak_angle,
        duration_s=r.duration_s,
        peak_vel=r.peak_vel_dps,
        similarity=r.dtw_similarity,
        label=r.label,
        flags=list(r.flags),
    )


def _enable_sqlite_fk(dbapi_conn: Any, _record: Any) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


def make_engine(db_url: str) -> Engine:
    """Create the engine (SQLite: parent dir created, foreign keys on, thread-safe)."""
    kwargs: dict[str, Any] = {}
    if db_url.startswith("sqlite"):
        path = db_url.split("///", 1)[-1]
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(db_url, **kwargs)
    if db_url.startswith("sqlite"):
        event.listen(engine, "connect", _enable_sqlite_fk)
    return engine


def init_db(engine: Engine) -> None:
    """Create all tables (idempotent)."""
    SQLModel.metadata.create_all(engine)


def session_scope(engine: Engine) -> Iterator[Session]:
    """FastAPI dependency body: one DB session per request."""
    with Session(engine) as db:
        yield db
