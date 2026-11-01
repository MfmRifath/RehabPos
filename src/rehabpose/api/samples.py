"""Sample session report from a SYNTHETIC session (reports/samples/)."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlmodel import Session

from rehabpose.api.db import Event, SessionRow, User, init_db, make_engine, rep_row
from rehabpose.api.reports import build_report, render_pdf
from rehabpose.core.engine import SessionEngine
from rehabpose.core.exercise import ExerciseConfig
from rehabpose.synth import RepSpec, rep_sequence

SAMPLE_NOTE = (
    "SYNTHETIC sample: generated skeletons, not a real person. These numbers are not a measurement."
)


def write_sample_report(exercise: ExerciseConfig, out_dir: Path) -> tuple[Path, Path]:
    """Run a synthetic shoulder-flexion set through the engine; write report JSON and PDF."""
    lean: dict[str, float | str] = {"trunk_lean_deg": 15.0, "trunk_lean_plane": "sagittal"}
    specs = [
        RepSpec(150),
        RepSpec(150, compensation=lean),
        RepSpec(148),
        RepSpec(110),
        RepSpec(150, duration_s=2.0),
        RepSpec(150, compensation={"elbow_flexion_deg": 40.0}),
    ]
    engine = SessionEngine(exercise, "R")
    for frame in rep_sequence(exercise.id, specs, seed=7):
        if frame is not None:
            engine.process(frame)

    started = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    summary = engine.summary()
    sid = "synthetic-sample"
    with tempfile.TemporaryDirectory() as tmp:
        db_engine = make_engine(f"sqlite:///{Path(tmp) / 'sample.db'}")
        init_db(db_engine)
        with Session(db_engine) as db:
            db.add(User(id="synthetic-demo", display_name="Synthetic demo"))
            db.flush()
            row = SessionRow(
                id=sid,
                user_id="synthetic-demo",
                exercise_id=exercise.id,
                side="R",
                view=engine.view,
                source="synthetic",
                started_at=started,
                ended_at=started + timedelta(seconds=engine.reps[-1].t_end + 1),
                rep_count=summary["rep_count"],
                mean_rom=summary["mean_rom"],
                quality_score=summary["quality_score"],
                model_version="synthetic-generator|rules",
            )
            db.add(row)
            db.flush()
            for rep in engine.reps:
                db.add(rep_row(sid, rep))
            for e in engine.events:
                db.add(
                    Event(
                        session_id=sid,
                        t=e.t,
                        code=e.code,
                        severity=e.severity,
                        confidence=e.confidence,
                        source=e.source,
                    )
                )
            db.commit()
            report = build_report(db, row, exercise, note=SAMPLE_NOTE)
        db_engine.dispose()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "sample_report.json"
    pdf_path = out_dir / "sample_report.pdf"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    pdf_path.write_bytes(render_pdf(report))
    return json_path, pdf_path
