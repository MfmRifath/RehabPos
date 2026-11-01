"""Application state shared by routes: settings, DB engine, exercises, job manager."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request
from sqlalchemy.engine import Engine
from sqlmodel import Session

from rehabpose import __version__
from rehabpose.api.db import SessionRow, Target
from rehabpose.core.exercise import ExerciseConfig, apply_overrides
from rehabpose.settings import Settings

if TYPE_CHECKING:
    from rehabpose.api.jobs import JobManager


@dataclass
class AppState:
    """Everything the API needs at runtime."""

    settings: Settings
    db: Engine
    exercises: dict[str, ExerciseConfig]
    jobs: JobManager | None = None

    @cached_property
    def torch_device(self) -> str:
        """Torch device (lazy: importing torch is slow)."""
        from rehabpose.settings import torch_device

        return torch_device()

    def exercise(self, exercise_id: str) -> ExerciseConfig:
        """Default exercise config or 404."""
        if exercise_id not in self.exercises:
            raise HTTPException(404, f"unknown exercise {exercise_id!r}")
        return self.exercises[exercise_id]

    def exercise_for(self, db: Session, user_id: str, exercise_id: str) -> ExerciseConfig:
        """Exercise config with the user's therapist overrides applied."""
        base = self.exercise(exercise_id)
        t = db.get(Target, (user_id, exercise_id))
        if t is None:
            return base
        return apply_overrides(base, t.target_rom, t.min_rep_s, dict(t.thresholds))


def model_version(pose: str) -> str:
    """Version string stored with each session, e.g. 'rehabpose-0.1.0|browser-mediapipe|rules'."""
    return f"rehabpose-{__version__}|{pose}|rules"


def get_state(request: Request) -> AppState:
    """FastAPI dependency."""
    state: AppState = request.app.state.rp
    return state


def get_session_row(db: Session, sid: str) -> SessionRow:
    """Session or 404."""
    row = db.get(SessionRow, sid)
    if row is None:
        raise HTTPException(404, f"unknown session {sid!r}")
    return row
