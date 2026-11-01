"""Background processing of uploaded videos (thread pool, MediaPipe on CPU)."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from sqlalchemy.engine import Engine
from sqlmodel import Session

from rehabpose.api.db import Event, SessionRow, rep_row, utcnow
from rehabpose.core.exercise import ExerciseConfig
from rehabpose.data.extract import extract_video
from rehabpose.offline import analyze
from rehabpose.pose.base import PoseEstimator

log = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "done", "error"]
EstimatorFactory = Callable[[], PoseEstimator]
ExerciseLookup = Callable[[Session, str, str], ExerciseConfig]


@dataclass
class Job:
    """In-memory job record (jobs do not survive a server restart)."""

    id: str
    session_id: str
    status: JobStatus = "queued"
    progress: float = 0.0
    error: str | None = None
    result: dict[str, str] = field(default_factory=dict)


class JobManager:
    """Runs upload jobs one at a time; raw uploads are deleted after processing."""

    def __init__(
        self,
        db: Engine,
        sessions_dir: Path,
        exercise_lookup: ExerciseLookup,
        estimator_factory: EstimatorFactory,
        pose_label: str,
        max_workers: int = 1,
    ) -> None:
        self.db = db
        self.sessions_dir = Path(sessions_dir)
        self.exercise_lookup = exercise_lookup
        self.estimator_factory = estimator_factory
        self.pose_label = pose_label
        self._executor = ThreadPoolExecutor(max_workers, thread_name_prefix="rehabpose-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, session_id: str, video: Path) -> Job:
        """Queue a job for an already-created upload session."""
        job = Job(id=uuid.uuid4().hex[:12], session_id=session_id)
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run, job, Path(video))
        return job

    def get(self, job_id: str) -> Job | None:
        """Job by id."""
        with self._lock:
            return self._jobs.get(job_id)

    def shutdown(self, wait: bool = True) -> None:
        """Stop the worker pool."""
        self._executor.shutdown(wait=wait)

    def _run(self, job: Job, video: Path) -> None:
        job.status = "running"
        sid = job.session_id
        out_dir = self.sessions_dir / sid
        est: PoseEstimator | None = None
        try:
            with Session(self.db) as db:
                row = db.get(SessionRow, sid)
                if row is None:
                    raise RuntimeError(f"session {sid} disappeared")
                exercise = self.exercise_lookup(db, row.user_id, row.exercise_id)
                side, view = row.side, row.view

            def progress(done: int, total: int) -> None:
                job.progress = round(0.9 * done / total, 3) if total else 0.0

            est = self.estimator_factory()
            landmarks = out_dir / "landmarks.parquet"
            extract_video(
                video, est, landmarks, meta={"pose_backend": self.pose_label}, on_progress=progress
            )
            res = analyze(landmarks, exercise, side, out_dir, view=view, video=video)  # type: ignore[arg-type]

            with Session(self.db) as db:
                row = db.get(SessionRow, sid)
                if row is None:
                    raise RuntimeError(f"session {sid} disappeared")
                for r in res.reps:
                    db.add(rep_row(sid, r))
                for e in res.events:
                    db.add(Event(session_id=sid, t=e.t, code=e.code, severity=e.severity,
                                 confidence=e.confidence, source=e.source))  # fmt: skip
                s = res.summary
                row.rep_count = int(s["rep_count"])
                row.mean_rom = s["mean_rom"]
                row.quality_score = s["quality_score"]
                row.ended_at = utcnow()
                db.add(row)
                db.commit()
            job.result = {
                "session": f"/api/sessions/{sid}",
                "report_json": f"/api/sessions/{sid}/report.json",
                "report_pdf": f"/api/sessions/{sid}/report.pdf",
                "annotated_video": f"/api/sessions/{sid}/annotated.mp4",
            }
            job.progress, job.status = 1.0, "done"
        except Exception as exc:  # noqa: BLE001 - surfaced to the client via job.error
            log.exception("job %s failed", job.id)
            job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            if est is not None:
                est.close()
            video.unlink(missing_ok=True)  # privacy: raw uploads are not kept (docs/09 §4)
