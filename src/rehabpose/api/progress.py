"""Progress aggregation, streaks and badges (pure functions over ended sessions)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

from rehabpose.api.db import SessionRow

FIRST_REPS_GOAL = 50
STREAK_GOAL_DAYS = 7


def as_utc(dt: datetime) -> datetime:
    """Treat naive datetimes (SQLite drops the zone) as UTC."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def local_day(dt: datetime, tz_offset_minutes: int = 0) -> date:
    """Calendar day of a timestamp at a UTC offset in minutes (e.g. +330 for UTC+5:30)."""
    return (as_utc(dt) + timedelta(minutes=tz_offset_minutes)).date()


def current_streak(days: Iterable[date], today: date) -> int:
    """Consecutive active days ending today, or yesterday so a streak survives until tonight."""
    active = set(days)
    day = today if today in active else today - timedelta(days=1)
    count = 0
    while day in active:
        count += 1
        day -= timedelta(days=1)
    return count


def longest_streak(days: Iterable[date]) -> int:
    """Longest run of consecutive active days."""
    best = run = 0
    previous: date | None = None
    for day in sorted(set(days)):
        run = run + 1 if previous is not None and day - previous == timedelta(days=1) else 1
        best = max(best, run)
        previous = day
    return best


def _weighted(rows: Sequence[SessionRow], attr: str) -> float | None:
    """Mean of a session attribute, weighted by rep count (sessions without reps are ignored)."""
    pairs = [(getattr(r, attr), r.rep_count) for r in rows if getattr(r, attr) is not None]
    total = sum(n for _, n in pairs)
    return sum(v * n for v, n in pairs) / total if total else None


def _rounded(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def aggregate(
    uid: str, rows: Sequence[SessionRow], now: datetime, tz_offset_minutes: int = 0
) -> dict[str, Any]:
    """Overall totals, streaks, badges and a per-exercise daily series.

    Only ended sessions should be passed. A day counts towards streaks only if it has at
    least one rep, so opening and closing a session is not practice. Badges stay earned.
    """
    ordered = sorted(rows, key=lambda r: as_utc(r.started_at))
    by_exercise: dict[str, dict[date, list[SessionRow]]] = defaultdict(lambda: defaultdict(list))
    for r in ordered:
        by_exercise[r.exercise_id][local_day(r.started_at, tz_offset_minutes)].append(r)

    exercises: dict[str, Any] = {}
    pb_beaten = False
    for eid, by_day in by_exercise.items():
        series: list[dict[str, Any]] = []
        for day, sessions in sorted(by_day.items()):
            series.append(
                {
                    "date": day.isoformat(),
                    "sessions": len(sessions),
                    "reps": sum(s.rep_count for s in sessions),
                    "mean_rom": _rounded(_weighted(sessions, "mean_rom"), 2),
                    "quality": _rounded(_weighted(sessions, "quality_score"), 4),
                }
            )
        best: SessionRow | None = None
        for s in (x for x in ordered if x.exercise_id == eid and x.mean_rom is not None):
            if best is None or (s.mean_rom or 0.0) > (best.mean_rom or 0.0):
                pb_beaten = pb_beaten or best is not None
                best = s
        day_roms = [d["mean_rom"] for d in series if d["mean_rom"] is not None]
        exercises[eid] = {
            "sessions": sum(d["sessions"] for d in series),
            "total_reps": sum(d["reps"] for d in series),
            "best_mean_rom": max(day_roms) if day_roms else None,
            "rom_personal_best": (
                {
                    "value": best.mean_rom,
                    "date": local_day(best.started_at, tz_offset_minutes).isoformat(),
                }
                if best is not None
                else None
            ),
            "series": series,
        }

    active = [local_day(r.started_at, tz_offset_minutes) for r in ordered if r.rep_count > 0]
    today = local_day(now, tz_offset_minutes)
    total_reps = sum(r.rep_count for r in ordered)
    longest = longest_streak(active)
    current = current_streak(active, today)
    return {
        "user_id": uid,
        "today": today.isoformat(),
        "tz_offset_minutes": tz_offset_minutes,
        "overall": {
            "sessions": len(ordered),
            "total_reps": total_reps,
            "active_days": len(set(active)),
            "current_streak": current,
            "longest_streak": longest,
        },
        "badges": [
            {
                "id": "first_50",
                "label": f"First {FIRST_REPS_GOAL} reps",
                "earned": total_reps >= FIRST_REPS_GOAL,
                "detail": f"{total_reps} / {FIRST_REPS_GOAL} reps",
            },
            {
                "id": "streak_7",
                "label": f"{STREAK_GOAL_DAYS}-day streak",
                "earned": longest >= STREAK_GOAL_DAYS,
                "detail": f"longest streak {longest} days, current {current}",
            },
            {
                "id": "rom_pb",
                "label": "ROM personal best",
                "earned": pb_beaten,
                "detail": "a session beat your earlier best mean ROM"
                if pb_beaten
                else "beat your best session mean ROM",
            },
        ],
        "exercises": exercises,
    }
