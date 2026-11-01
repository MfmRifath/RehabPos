"""Video upload and job polling."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from rehabpose.api.deps import DbDep, StateDep
from rehabpose.api.routes.sessions import create_session_row
from rehabpose.api.schemas import JobOut, SessionCreate, SideT, VideoAccepted, ViewT

router = APIRouter(prefix="/api", tags=["videos"])

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
CHUNK = 1 << 20


@router.post("/videos", status_code=202)
async def upload_video(
    db: DbDep,
    state: StateDep,
    file: Annotated[UploadFile, File()],
    user_id: Annotated[str, Form()],
    exercise_id: Annotated[str, Form()],
    side: Annotated[SideT, Form()],
    view: Annotated[ViewT | None, Form()] = None,
) -> VideoAccepted:
    """Store the upload, create an upload session and queue processing."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(415, f"unsupported video type {suffix!r}; use {sorted(VIDEO_SUFFIXES)}")
    if state.jobs is None:
        raise HTTPException(503, "job runner unavailable")
    body = SessionCreate(
        user_id=user_id, exercise_id=exercise_id, side=side, view=view, source="upload"
    )
    row = create_session_row(body, db, state)
    uploads = state.settings.data_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    dest = uploads / f"{uuid.uuid4().hex}{suffix}"
    limit = state.settings.max_upload_mb * 1024 * 1024
    size = 0
    with dest.open("wb") as fh:
        while chunk := await file.read(CHUNK):
            size += len(chunk)
            if size > limit:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"upload exceeds {state.settings.max_upload_mb} MB")
            fh.write(chunk)
    job = state.jobs.submit(row.id, dest)
    return VideoAccepted(job_id=job.id, session_id=row.id)


@router.get("/jobs/{job_id}")
def get_job(job_id: str, state: StateDep) -> JobOut:
    """Job status, progress (0-1) and result links."""
    job = state.jobs.get(job_id) if state.jobs else None
    if job is None:
        raise HTTPException(404, f"unknown job {job_id!r}")
    return JobOut(
        id=job.id, status=job.status, progress=job.progress, session_id=job.session_id,
        error=job.error, result=job.result,
    )  # fmt: skip
