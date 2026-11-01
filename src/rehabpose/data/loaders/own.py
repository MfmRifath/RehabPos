"""Own recordings: labels/<video_id>.json + extracted landmarks (+ optional raw video).

Layout (defined by this project, docs/05 §8):
  data/labels/own/<video_id>.json        annotation schema v1
  data/interim/own/<video_id>.parquet    `rehabpose extract` output (optional)
  data/raw/own/videos/<video_id>.<ext>   raw video (optional, consent-dependent)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from rehabpose.data.extract import load_frames, read_meta
from rehabpose.data.loaders.base import DatasetInfo, DatasetLoader, DatasetMissingError, Sequence
from rehabpose.data.schema import load_annotation

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


class OwnLoader(DatasetLoader):
    """RehabPose's own consented recordings (keypoints by default)."""

    info = DatasetInfo(
        name="own",
        title="RehabPose own recordings",
        license="Own data under participant consent; not redistributable without opt-in (d)",
        citation="RehabPose own dataset (see docs/11_datasheet.md)",
        url="local",
        access="own",
        instructions="Record per docs/05 §3, then label with `rehabpose label <video>`.",
        layout_verified=True,
        requires_acceptance=False,
    )

    def __init__(self, data_dir: Path = Path("data"), accept_license: bool = False) -> None:
        super().__init__(Path(data_dir) / "raw", accept_license)
        self.labels_dir = Path(data_dir) / "labels" / "own"
        self.interim_dir = Path(data_dir) / "interim" / "own"

    def check(self) -> None:
        """Own data needs no license acceptance; labels must exist."""
        if not self.labels_dir.is_dir() or not any(self.labels_dir.glob("*.json")):
            raise DatasetMissingError(f"No labels in {self.labels_dir}/\n{self.info.instructions}")

    def _iter(self) -> Iterator[Sequence]:
        for path in sorted(self.labels_dir.glob("*.json")):
            ann = load_annotation(path)
            parquet = self.interim_dir / f"{ann.video_id}.parquet"
            video = next(
                (
                    p
                    for ext in VIDEO_EXTS
                    if (p := self.root / "videos" / f"{ann.video_id}{ext}").is_file()
                ),
                None,
            )
            frames = load_frames(parquet) if parquet.is_file() else None
            fps = float(read_meta(parquet).get("fps") or ann.fps) if parquet.is_file() else ann.fps
            yield Sequence(
                dataset="own",
                subject=ann.subject,
                exercise=ann.exercise_id,
                exercise_id=ann.exercise_id,
                side=ann.side,
                view=ann.view,
                fps=fps,
                source=path,
                joint_format="mediapipe33" if frames is not None else "video",
                frames=frames,
                video=video,
                labels=ann.model_dump(),
                license=self.info.license,
                citation=self.info.citation,
            )
