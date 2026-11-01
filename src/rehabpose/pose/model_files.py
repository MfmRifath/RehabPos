"""Resolve MediaPipe pose model files under data/models/; download only with explicit consent."""

from __future__ import annotations

import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

ModelVariant = Literal["lite", "full", "heavy"]
VARIANTS: tuple[ModelVariant, ...] = ("lite", "full", "heavy")
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # project rule: larger files need the owner's approval

_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
)


class ModelFileMissingError(FileNotFoundError):
    """Raised when a model file is absent and download was not permitted."""


def model_url(variant: ModelVariant) -> str:
    """Official Google-hosted URL of the float16 pose_landmarker_<variant>.task model."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    return _URL.format(v=variant)


def model_path(variant: ModelVariant, models_dir: Path) -> Path:
    """Expected local path of pose_landmarker_<variant>.task."""
    model_url(variant)  # validates variant
    return Path(models_dir) / f"pose_landmarker_{variant}.task"


def download_model(variant: ModelVariant, models_dir: Path, timeout_s: float = 60.0) -> Path:
    """Download the model to models_dir (atomic write); refuses files over 50 MB."""
    dest = model_path(variant, models_dir)
    url = model_url(variant)
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s", url, dest)
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # fixed https URL
        size = int(resp.headers.get("Content-Length") or 0)
        if size > MAX_DOWNLOAD_BYTES:
            raise RuntimeError(f"{url} is {size} bytes (> 50 MB); ask the owner before downloading")
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, suffix=".part") as tmp:
            copied = 0
            while chunk := resp.read(1 << 20):
                copied += len(chunk)
                if copied > MAX_DOWNLOAD_BYTES:
                    Path(tmp.name).unlink(missing_ok=True)
                    raise RuntimeError(f"{url} exceeded 50 MB while downloading; aborted")
                tmp.write(chunk)
    shutil.move(tmp.name, dest)
    log.info("Saved %s (%d bytes)", dest, copied)
    return dest


def resolve_model(variant: ModelVariant, models_dir: Path, *, download: bool = False) -> Path:
    """Return the local model path; download only if `download` is True (explicit consent)."""
    path = model_path(variant, models_dir)
    if path.is_file():
        return path
    if download:
        return download_model(variant, models_dir)
    raise ModelFileMissingError(
        f"Pose model not found: {path}\n"
        f"Official download URL: {model_url(variant)}\n"
        f"Download it manually to that path, or re-run with --download-model to fetch it."
    )
