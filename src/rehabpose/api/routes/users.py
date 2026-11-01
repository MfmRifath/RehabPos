"""Per-user targets (therapist settings) and progress aggregation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlmodel import col, select

from rehabpose.api.db import SessionRow, Target, User
from rehabpose.api.deps import DbDep, StateDep
from rehabpose.api.schemas import TargetIn, TargetsOut
from rehabpose.core.exercise import apply_overrides

router = APIRouter(prefix="/api/users", tags=["users"])


def _user_or_404(db: DbDep, uid: str) -> User:
    user = db.get(User, uid)
    if user is None:
        raise HTTPException(404, f"unknown user {uid!r}")
    return user


@router.get("/{uid}/targets")
def get_targets(uid: str, db: DbDep) -> TargetsOut:
    """The user's overrides (exercises without one use the YAML defaults)."""
    _user_or_404(db, uid)
    rows = db.exec(select(Target).where(Target.user_id == uid)).all()
    return TargetsOut(user_id=uid, targets=[TargetIn.model_validate(r) for r in rows])


@router.put("/{uid}/targets")
def put_targets(uid: str, body: list[TargetIn], db: DbDep, state: StateDep) -> TargetsOut:
    """Upsert overrides; each is validated against the exercise config (422 if invalid)."""
    _user_or_404(db, uid)
    for t in body:
        try:
            apply_overrides(state.exercise(t.exercise_id), t.target_rom, t.min_rep_s, t.thresholds)
        except ValueError as exc:
            raise HTTPException(422, f"{t.exercise_id}: {exc}") from exc
        row = db.get(Target, (uid, t.exercise_id)) or Target(user_id=uid, exercise_id=t.exercise_id)
        row.target_rom, row.min_rep_s, row.thresholds = (
            t.target_rom,
            t.min_rep_s,
            dict(t.thresholds),
        )
        db.add(row)
    db.commit()
    return get_targets(uid, db)


@router.get("/{uid}/progress")
def progress(uid: str, db: DbDep) -> dict[str, Any]:
    """Per exercise and day: sessions, reps, mean ROM and quality of ended sessions."""
    _user_or_404(db, uid)
    rows = db.exec(
        select(SessionRow)
        .where(SessionRow.user_id == uid, col(SessionRow.ended_at).is_not(None))
        .order_by(col(SessionRow.started_at))
    ).all()
    days: dict[str, dict[str, list[SessionRow]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        days[r.exercise_id][r.started_at.date().isoformat()].append(r)
    exercises: dict[str, Any] = {}
    for eid, by_day in days.items():
        series: list[dict[str, Any]] = []
        day_roms: list[float] = []
        for day, ss in sorted(by_day.items()):
            roms = [s.mean_rom for s in ss if s.mean_rom is not None]
            if roms:
                day_roms.append(sum(roms) / len(roms))
            quals = [s.quality_score for s in ss if s.quality_score is not None]
            series.append({
                "date": day,
                "sessions": len(ss),
                "reps": sum(s.rep_count for s in ss),
                "mean_rom": round(sum(roms) / len(roms), 2) if roms else None,
                "quality": round(sum(quals) / len(quals), 4) if quals else None,
            })  # fmt: skip
        exercises[eid] = {
            "sessions": sum(d["sessions"] for d in series),
            "total_reps": sum(d["reps"] for d in series),
            "best_mean_rom": round(max(day_roms), 2) if day_roms else None,
            "series": series,
        }
    return {"user_id": uid, "exercises": exercises}
