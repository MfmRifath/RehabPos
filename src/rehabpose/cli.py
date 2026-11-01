"""`rehabpose` command-line interface. Subcommands are stubs until their phase lands."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any, NoReturn

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
def serve(
    host: Annotated[str | None, typer.Option(help="Bind address (default from settings).")] = None,
    port: Annotated[int | None, typer.Option(help="Port (default from settings).")] = None,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes (dev).")] = False,
) -> None:
    """Run the FastAPI server (REST + WebSocket)."""
    import uvicorn

    settings = get_settings()
    host, port = host or settings.host, port or settings.port
    log.info("RehabPose API on http://%s:%d (docs at /docs)", host, port)
    if reload:
        uvicorn.run(
            "rehabpose.api.main:create_app", host=host, port=port, reload=True, factory=True
        )
        return
    from rehabpose.api.main import create_app

    uvicorn.run(create_app(settings), host=host, port=port, log_level=settings.log_level.lower())


data_app = typer.Typer(help="Dataset utilities (list, info, stats, extract-all, splits, schema).")
app.add_typer(data_app, name="data")

AcceptOpt = Annotated[
    bool, typer.Option("--i-accept-license", help="Confirm you read and accept the license.")
]


def _loader(name: str, accept: bool) -> Any:
    from rehabpose.data.loaders import get_loader

    try:
        return get_loader(name, get_settings().data_dir, accept)
    except KeyError as exc:
        raise typer.BadParameter(str(exc.args[0])) from exc


@data_app.command("list")
def data_list() -> None:
    """List known datasets, licenses and whether files are present."""
    from rich.console import Console
    from rich.table import Table

    from rehabpose.data.loaders import REGISTRY

    root = get_settings().data_dir
    table = Table("name", "license", "access", "present", "layout verified")
    for name, cls in REGISTRY.items():
        folder = root / ("labels/own" if name == "own" else f"raw/{name}")
        present = folder.is_dir() and any(folder.iterdir())
        info = cls.info
        table.add_row(
            name, info.license, info.access, "yes" if present else "no", str(info.layout_verified)
        )
    Console(width=140).print(table)


@data_app.command("info")
def data_info(name: str) -> None:
    """Print license, citation and download instructions for a dataset."""
    from rehabpose.data.loaders import REGISTRY
    from rehabpose.data.loaders.base import license_text

    if name not in REGISTRY:
        raise typer.BadParameter(f"choose from {sorted(REGISTRY)}")
    info = REGISTRY[name].info
    typer.echo(license_text(info))
    typer.echo(f"Instructions: {info.instructions}")
    typer.echo(f"Layout verified on real files: {info.layout_verified}")


@data_app.command("stats")
def data_stats(name: str, i_accept_license: AcceptOpt = False) -> None:
    """Summarize a dataset (sequences, subjects, exercises, labels)."""
    from rehabpose.data.loaders import DatasetError
    from rehabpose.data.tasks import dataset_stats

    loader = _loader(name, i_accept_license)
    try:
        stats = dataset_stats(loader.sequences())
    except DatasetError as exc:
        log.error("%s", exc)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(stats, indent=2))


@data_app.command("extract-all")
def data_extract_all(
    name: str,
    i_accept_license: AcceptOpt = False,
    model: Annotated[str, typer.Option(help="MediaPipe variant for video datasets.")] = "full",
    limit: Annotated[int | None, typer.Option(help="Stop after N new files.")] = None,
    overwrite: Annotated[bool, typer.Option(help="Re-extract existing outputs.")] = False,
) -> None:
    """Extract/convert every sequence to landmark parquet under data/interim/<name>/."""
    from rehabpose.data.loaders import DatasetError
    from rehabpose.data.tasks import extract_all
    from rehabpose.pose import create_estimator
    from rehabpose.pose.model_files import VARIANTS, ModelFileMissingError, resolve_model

    if model not in VARIANTS:
        raise typer.BadParameter(f"must be one of {VARIANTS}", param_hint="--model")
    settings = get_settings()
    loader = _loader(name, i_accept_license)

    def factory() -> Any:
        return create_estimator(model, settings.models_dir)

    try:
        if loader.info.name in ("uco", "rehab24_6", "dib_upper", "own"):
            resolve_model(model, settings.models_dir)
        counts = extract_all(
            loader, settings.data_dir / "interim", factory, model, limit, overwrite
        )
    except (DatasetError, ModelFileMissingError) as exc:
        log.error("%s", exc)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(counts))


@data_app.command("splits")
def data_splits(
    name: str,
    i_accept_license: AcceptOpt = False,
    out: Annotated[Path, typer.Option(help="Output YAML.")] = Path("configs/train/splits.yaml"),
    test_fraction: float = 0.2,
    k: int = 5,
    seed: int = 0,
) -> None:
    """Write a subject-wise split plan (held-out test subjects + folds) for a dataset."""
    from rehabpose.data.loaders import DatasetError
    from rehabpose.data.splits import make_split_plan, save_plan

    loader = _loader(name, i_accept_license)
    try:
        plan = make_split_plan((s.subject for s in loader.sequences()), test_fraction, k, seed)
    except (DatasetError, ValueError) as exc:
        log.error("%s", exc)
        raise typer.Exit(code=2) from exc
    save_plan(plan, out)
    typer.echo(f"{plan.method}: {len(plan.test)} test subjects, {len(plan.folds)} folds -> {out}")


@data_app.command("schema")
def data_schema() -> None:
    """Print the annotation JSON Schema (v1)."""
    from rehabpose.data.schema import schema_json

    typer.echo(schema_json())


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
def label(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Video to label.")],
    exercise: Annotated[str, typer.Option(help="Exercise id.")],
    subject: Annotated[str, typer.Option(help="Pseudonymous subject id, e.g. P01.")],
    side: Annotated[str, typer.Option(help="L or R.")] = "R",
    view: Annotated[
        str | None, typer.Option(help="sagittal | frontal (default: exercise's).")
    ] = None,
    annotator: Annotated[str, typer.Option(help="Annotator id.")] = "A1",
    out: Annotated[
        Path | None, typer.Option(help="Output JSON (default data/labels/own/<stem>.json).")
    ] = None,
) -> None:
    """Open the OpenCV labeling helper on a video (resumes an existing label file)."""
    from rehabpose.core.exercise import load_exercises
    from rehabpose.data.labeling import HELP, LabelState
    from rehabpose.data.labeling.tool import run_labeler
    from rehabpose.data.schema import load_annotation
    from rehabpose.data.video_io import probe

    settings = get_settings()
    exercises = load_exercises(settings.exercises_dir)
    if exercise not in exercises:
        raise typer.BadParameter(f"choose from {sorted(exercises)}", param_hint="--exercise")
    if side not in ("L", "R"):
        raise typer.BadParameter("must be L or R", param_hint="--side")
    view = view or exercises[exercise].view
    if view not in ("sagittal", "frontal"):
        raise typer.BadParameter("must be sagittal or frontal", param_hint="--view")
    out = out or settings.data_dir / "labels" / "own" / f"{video.stem}.json"
    info = probe(video)
    if out.is_file():
        state = LabelState.from_annotation(load_annotation(out), info.frame_count)
    else:
        state = LabelState(
            video_id=video.stem, subject=subject, exercise_id=exercise, side=side,  # type: ignore[arg-type]
            view=view, fps=info.fps, n_frames=info.frame_count, annotator=annotator,  # type: ignore[arg-type]
        )  # fmt: skip
    typer.echo(HELP)
    ann = run_labeler(video, state, out)
    typer.echo(
        f"{out}: {len(ann.reps)} reps, {len(ann.compensation_segments)} segments, "
        f"{len(ann.angle_annotations)} angles"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
