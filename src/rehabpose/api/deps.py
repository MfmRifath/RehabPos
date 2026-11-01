"""FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends
from sqlmodel import Session

from rehabpose.api.state import AppState, get_state

StateDep = Annotated[AppState, Depends(get_state)]


def get_db(state: StateDep) -> Iterator[Session]:
    """One DB session per request."""
    with Session(state.db) as db:
        yield db


DbDep = Annotated[Session, Depends(get_db)]
