"""OpenCV video reading and writing with frame timestamps in milliseconds."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Basic video properties: fps (frames/s), frame_count, width/height (pixels)."""

    path: Path
    fps: float
    frame_count: int
    width: int
    height: int


def _open(path: Path) -> cv2.VideoCapture:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV cannot open video: {path}")
    return cap


def probe(path: Path) -> VideoInfo:
    """Read fps, frame count and size from the container header."""
    cap = _open(path)
    try:
        return VideoInfo(
            path=Path(path),
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def iter_frames(path: Path) -> Iterator[tuple[int, int, NDArray[np.uint8]]]:
    """Yield (frame_idx, ts_ms, bgr_image); ts_ms = round(idx * 1000 / fps)."""
    fps = probe(path).fps
    cap = _open(path)
    try:
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield idx, int(round(idx * 1000.0 / fps)), np.asarray(img, dtype=np.uint8)
            idx += 1
    finally:
        cap.release()


class VideoWriter:
    """MP4 writer (mp4v codec) for annotated output; use as a context manager."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if not self._writer.isOpened():
            raise ValueError(f"OpenCV cannot write video: {path}")
        self.size = size

    def write(self, frame_bgr: NDArray[np.uint8]) -> None:
        """Append one BGR frame of the configured (width, height)."""
        self._writer.write(frame_bgr)

    def close(self) -> None:
        """Finalize the file."""
        self._writer.release()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
