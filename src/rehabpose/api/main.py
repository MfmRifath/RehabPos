"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from rehabpose import __version__
from rehabpose.api import ws
from rehabpose.api.db import init_db, make_engine
from rehabpose.api.jobs import EstimatorFactory, JobManager
from rehabpose.api.privacy import purge_expired_outputs
from rehabpose.api.routes import meta, sessions, users, videos
from rehabpose.api.state import AppState
from rehabpose.core.exercise import load_exercises
from rehabpose.settings import Settings, get_settings

log = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None, estimator_factory: EstimatorFactory | None = None
) -> FastAPI:
    """Build the app: DB (created if missing), exercises, job runner, routes, static frontend."""
    settings = settings or get_settings()
    db = make_engine(settings.db_url)
    init_db(db)
    state = AppState(settings=settings, db=db, exercises=load_exercises(settings.exercises_dir))
    purge_expired_outputs(db, settings.data_dir, settings.session_retention_days)

    def default_factory() -> object:
        from rehabpose.pose import create_estimator

        return create_estimator(settings.pose_model, settings.models_dir)

    state.jobs = JobManager(
        db,
        settings.data_dir / "sessions",
        state.exercise_for,
        estimator_factory or default_factory,  # type: ignore[arg-type]
        pose_label=f"mediapipe-{settings.pose_model}",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        if state.jobs is not None:
            state.jobs.shutdown(wait=False)
        db.dispose()

    app = FastAPI(title="RehabPose API", version=__version__, lifespan=lifespan)
    app.state.rp = state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    routers = (meta.router, meta.assets, sessions.router, users.router, videos.router, ws.router)
    for router in routers:
        app.include_router(router)
    if settings.frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")
    else:
        log.info("frontend build not found at %s; serving API only", settings.frontend_dist)
    return app
