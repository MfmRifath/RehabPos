"""Health and exercise catalogue."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import FileResponse

from rehabpose import __version__
from rehabpose.api.deps import StateDep
from rehabpose.api.schemas import ExerciseOut, HealthOut, RuleOut
from rehabpose.core.exercise import ExerciseConfig, FrameRule

router = APIRouter(prefix="/api", tags=["meta"])
assets = APIRouter(tags=["meta"])


@assets.get("/models/{name}", include_in_schema=False)
def pose_model_file(
    name: Annotated[str, Path(pattern=r"^pose_landmarker_(lite|full|heavy)\.task$")],
    state: StateDep,
) -> FileResponse:
    """Serve a pose model to the browser from data/models (a Docker image does not contain them)."""
    for folder in (state.settings.models_dir, state.settings.frontend_dist / "models"):
        if (folder / name).is_file():
            return FileResponse(folder / name, media_type="application/octet-stream")
    raise HTTPException(404, f"{name} not found; see rehabpose.pose.model_files for the URL")


@router.get("/health")
def health(state: StateDep) -> HealthOut:
    """Liveness, version, pose model and torch device."""
    return HealthOut(
        status="ok",
        version=__version__,
        pose_model=state.settings.pose_model,
        torch_device=state.torch_device,
        exercises=len(state.exercises),
    )


def exercise_out(ex: ExerciseConfig) -> ExerciseOut:
    """Client view of an exercise config."""
    rules = [
        RuleOut(code=r.code, kind="frame", measure=r.measure, op=r.op, value=r.value,
                severity=r.severity, experimental=r.experimental)
        if isinstance(r, FrameRule)
        else RuleOut(code=r.code, kind="rep", severity=r.severity)
        for r in ex.rules
    ]  # fmt: skip
    return ExerciseOut(
        id=ex.id,
        name=ex.name,
        view=ex.view,
        allowed_views=list(ex.views),
        experimental=ex.experimental,
        primary_angle=ex.primary_angle,
        unit=ex.unit,
        target=ex.target,
        min_rep_s=ex.min_rep_s,
        fsm=ex.fsm.model_dump(),
        rules=rules,
        cue_texts=ex.cue_texts,
    )


@router.get("/exercises")
def exercises(state: StateDep) -> list[ExerciseOut]:
    """All exercise definitions (engineering defaults, not clinical prescriptions)."""
    return [exercise_out(ex) for ex in state.exercises.values()]
