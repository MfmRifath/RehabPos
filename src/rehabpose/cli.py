"""`rehabpose` command-line interface. Subcommands are stubs until their phase lands."""

from __future__ import annotations

import logging
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
def extract() -> None:
    """Extract pose landmarks from a video into a parquet file."""
    _not_implemented("extract", 2)


@app.command()
def analyze() -> None:
    """Analyze a video or landmark parquet offline (reps, cues, annotated MP4)."""
    _not_implemented("analyze", 3)


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
