"""`rehabpose` command-line interface. Subcommands are stubs until their phase lands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.logging import RichHandler

from rehabpose import __version__
from rehabpose.settings import get_settings

log = logging.getLogger("rehabpose")

app = typer.Typer(
    name="rehabpose",
    help="RehabPose: camera-only rehabilitation exercise coach (research prototype).",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"rehabpose {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Configure logging for all subcommands."""
    level = "DEBUG" if verbose else get_settings().log_level
    logging.basicConfig(
        level=level, format="%(message)s", handlers=[RichHandler(rich_tracebacks=True)], force=True
    )


def _not_implemented(command: str, phase: int) -> NoReturn:
    log.error("`rehabpose %s` is not implemented yet (planned for phase %d).", command, phase)
    raise typer.Exit(code=1)


@app.command()
def extract(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Input video.")],
    model: Annotated[str, typer.Option(help="MediaPipe variant: lite | full | heavy.")] = "full",
    out: Annotated[
        Path | None, typer.Option(help="Output parquet (default data/interim/<stem>_<model>).")
    ] = None,
    download_model: Annotated[
        bool, typer.Option(help="Allow downloading the pose model if it is missing.")
    ] = False,
) -> None:
    """Extract pose landmarks from a video into a parquet file."""
    from rehabpose.data.extract import extract_video
    from rehabpose.pose import create_estimator
    from rehabpose.pose.model_files import VARIANTS, ModelFileMissingError

    if model not in VARIANTS:
        raise typer.BadParameter(f"must be one of {VARIANTS}", param_hint="--model")
    settings = get_settings()
    out = out or settings.data_dir / "interim" / f"{video.stem}_{model}.parquet"
    try:
        estimator = create_estimator(model, settings.models_dir, download=download_model)
    except ModelFileMissingError as exc:
        log.error("%s", exc)
        raise typer.Exit(code=2) from exc
    try:
        res = extract_video(
            video, estimator, out, meta={"pose_backend": "mediapipe", "model": model}, progress=True
        )
    finally:
        estimator.close()
    typer.echo(f"{res.out_path}: {res.n_frames} frames, {res.n_detected} with a person")


@app.command()
def analyze(
    source: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Video or landmark .parquet.")
    ],
    exercise: Annotated[str, typer.Option(help="Exercise id (configs/exercises/<id>.yaml).")],
    side: Annotated[str, typer.Option(help="Exercising side: L or R.")] = "R",
    view: Annotated[
        str | None, typer.Option(help="Camera view: sagittal | frontal (default: exercise's).")
    ] = None,
    out_dir: Annotated[
        Path | None, typer.Option(help="Output folder (default reports/run_<stem>).")
    ] = None,
    model: Annotated[str, typer.Option(help="Pose model for video input.")] = "full",
    download_model: Annotated[bool, typer.Option(help="Allow downloading the pose model.")] = False,
    video: Annotated[bool, typer.Option(help="Render annotated.mp4.")] = True,
) -> None:
    """Analyze a video or landmark parquet offline (reps, cues, annotated MP4)."""
    from rehabpose.core.exercise import load_exercises
    from rehabpose.offline import analyze as run_analysis
    from rehabpose.offline import extract_for_analysis
    from rehabpose.pose.model_files import ModelFileMissingError

    settings = get_settings()
    exercises = load_exercises(settings.exercises_dir)
    if exercise not in exercises:
        raise typer.BadParameter(f"choose from {sorted(exercises)}", param_hint="--exercise")
    if side not in ("L", "R"):
        raise typer.BadParameter("must be L or R", param_hint="--side")
    if view not in (None, "sagittal", "frontal"):
        raise typer.BadParameter("must be sagittal or frontal", param_hint="--view")
    out_dir = out_dir or Path("reports") / f"run_{source.stem}"

    if source.suffix == ".parquet":
        landmarks, source_video = source, None
    else:
        try:
            landmarks = extract_for_analysis(
                source, out_dir, model, settings.models_dir, download_model
            )
        except ModelFileMissingError as exc:
            log.error("%s", exc)
            raise typer.Exit(code=2) from exc
        source_video = source
    try:
        res = run_analysis(
            landmarks,
            exercises[exercise],
            side,  # type: ignore[arg-type]
            out_dir,
            view=view,  # type: ignore[arg-type]
            video=source_video,
            render_video=video,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    s = res.summary
    typer.echo(
        f"{exercise} {side}: {s['rep_count']} reps, labels {s['labels']}, "
        f"mean ROM {s['mean_rom']}, quality {s['quality_score']} -> {res.out_dir}"
    )


@app.command()
def serve() -> None:
    """Run the FastAPI server (REST + WebSocket)."""
    _not_implemented("serve", 5)


@app.command()
def data() -> None:
    """Dataset utilities (stats, extract-all)."""
    _not_implemented("data", 4)


@app.command()
def train() -> None:
    """Train a temporal form-quality model."""
    _not_implemented("train", 7)


@app.command()
def evaluate() -> None:
    """Evaluate models with subject-wise cross-validation."""
    _not_implemented("evaluate", 7)


@app.command()
def bench() -> None:
    """Benchmark pose backends, engine and server latency."""
    _not_implemented("bench", 9)


@app.command()
def label() -> None:
    """Open the OpenCV labeling helper on a video."""
    _not_implemented("label", 4)


if __name__ == "__main__":  # pragma: no cover
    app()
