"""Common dataset-loader interface: license gate, missing-data errors, layout patterns, readers.

Nothing here downloads data. Loaders read from data/raw/<name>/ and raise helpful errors.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import yaml
from numpy.typing import NDArray

from rehabpose.core.landmarks import Frame, Side

log = logging.getLogger(__name__)

View = Literal["sagittal", "frontal"]
Access = Literal["open", "on_request", "eula", "kaggle", "own"]
FloatArray = NDArray[np.float64]


@dataclass
class Sequence:
    """One recording/sequence from a dataset.

    Exactly one of `frames` (MediaPipe 33-landmark Frames), `skeleton` ((T, J, 3) in the
    dataset's native joint format) or `video` (not yet extracted) is the primary payload.
    """

    dataset: str
    subject: str
    exercise: str  # dataset-native exercise code/name
    exercise_id: str | None  # RehabPose exercise id when the mapping is verified
    side: Side | None
    view: View | None
    fps: float
    source: Path
    joint_format: str  # mediapipe33 | kinect_v2_25 | plug_in_gait | video | unknown_<J>
    frames: list[Frame | None] | None = None
    skeleton: FloatArray | None = None
    video: Path | None = None
    labels: dict[str, Any] = field(default_factory=dict)
    license: str = ""
    citation: str = ""

    @property
    def n_frames(self) -> int | None:
        """Number of frames, if the payload is loaded."""
        if self.frames is not None:
            return len(self.frames)
        if self.skeleton is not None:
            return int(self.skeleton.shape[0])
        return None


@dataclass(frozen=True)
class DatasetInfo:
    """Static facts about a dataset (as stated in the project spec; verify on the source page)."""

    name: str
    title: str
    license: str
    citation: str
    url: str
    access: Access
    instructions: str
    layout_verified: bool = False
    requires_acceptance: bool = True


class DatasetError(RuntimeError):
    """Base class for dataset problems."""


class LicenseNotAcceptedError(DatasetError):
    """The license was not explicitly accepted (--i-accept-license)."""


class DatasetMissingError(DatasetError):
    """Dataset files are not present under data/raw/<name>/."""


class LayoutError(DatasetError):
    """Files are present but do not match the expected layout/format."""


def license_text(info: DatasetInfo) -> str:
    """Human-readable license/citation block printed before using a dataset."""
    return (
        f"{info.title} [{info.name}]\n"
        f"  License: {info.license}\n"
        f"  Citation: {info.citation}\n"
        f"  Source: {info.url} (access: {info.access})"
    )


class DatasetLoader(ABC):
    """Base loader: call `sequences()` to iterate after the license and presence checks."""

    info: ClassVar[DatasetInfo]

    def __init__(self, raw_root: Path = Path("data/raw"), accept_license: bool = False) -> None:
        self.root = Path(raw_root) / self.info.name
        self.accept_license = accept_license

    def check(self) -> None:
        """Raise if the license is not accepted or the data is missing."""
        info = self.info
        if info.requires_acceptance and not self.accept_license:
            raise LicenseNotAcceptedError(
                f"{license_text(info)}\nRead the license, then re-run with --i-accept-license."
            )
        if not self.root.is_dir() or not any(self.root.iterdir()):
            raise DatasetMissingError(
                f"{info.title} not found at {self.root}/\n{info.instructions}\n"
                f"RehabPose never downloads datasets automatically."
            )
        if not info.layout_verified:
            log.warning(
                "%s: expected file layout is ASSUMED from the dataset description and has not "
                "been verified on real files; check results and adjust configs/datasets/%s.yaml.",
                info.name,
                info.name,
            )

    def sequences(self) -> Iterator[Sequence]:
        """Iterate all sequences (license + presence checked first)."""
        self.check()
        log.info("%s", license_text(self.info))
        n = 0
        for seq in self._iter():
            n += 1
            yield seq
        if n == 0:
            raise LayoutError(self._layout_hint())

    def _layout_hint(self) -> str:
        return f"No sequences found under {self.root}."

    @abstractmethod
    def _iter(self) -> Iterator[Sequence]:
        """Yield sequences (implemented per dataset)."""


class PatternLoader(DatasetLoader):
    """Loader driven by glob patterns + a regex on the root-relative POSIX path.

    Class defaults describe the ASSUMED layout; configs/datasets/<name>.yaml may override
    `patterns`, `regex`, `fps`, `view` and `exercise_map` once the real layout is known.
    """

    patterns: ClassVar[tuple[str, ...]] = ()
    regex: ClassVar[str] = ""
    fps: ClassVar[float] = 30.0
    view: ClassVar[View | None] = None
    exercise_map: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        raw_root: Path = Path("data/raw"),
        accept_license: bool = False,
        config_dir: Path = Path("configs/datasets"),
    ) -> None:
        super().__init__(raw_root, accept_license)
        self.layout: dict[str, Any] = {
            "patterns": list(self.patterns),
            "regex": self.regex,
            "fps": self.fps,
            "view": self.view,
            "exercise_map": dict(self.exercise_map),
        }
        override = Path(config_dir) / f"{self.info.name}.yaml"
        if override.is_file():
            data = yaml.safe_load(override.read_text()) or {}
            unknown = set(data) - set(self.layout)
            if unknown:
                raise LayoutError(f"{override}: unknown keys {sorted(unknown)}")
            self.layout.update(data)
        self._re = re.compile(self.layout["regex"], re.IGNORECASE)

    def files(self) -> list[Path]:
        """All files matching the layout patterns (sorted, unique)."""
        found = {p for pat in self.layout["patterns"] for p in self.root.glob(pat) if p.is_file()}
        return sorted(found)

    def parse_path(self, path: Path) -> dict[str, str] | None:
        """Regex groups for a file, or None if it does not match the layout."""
        m = self._re.search(path.relative_to(self.root).as_posix())
        return {k: v for k, v in m.groupdict().items() if v is not None} if m else None

    def _iter(self) -> Iterator[Sequence]:
        for path in self.files():
            groups = self.parse_path(path)
            if groups is None:
                log.debug("%s: skipping %s (does not match layout)", self.info.name, path)
                continue
            yield self.read(path, groups)

    def _layout_hint(self) -> str:
        return (
            f"No files under {self.root} matched the expected layout\n"
            f"  patterns: {self.layout['patterns']}\n  regex: {self.layout['regex']}\n"
            f"Adjust configs/datasets/{self.info.name}.yaml to the real layout."
        )

    def base_sequence(self, path: Path, groups: dict[str, str], **kw: Any) -> Sequence:
        """Sequence pre-filled from regex groups and dataset info."""
        exercise = groups.get("exercise", "unknown")
        side = groups.get("side", "").upper()[:1] or None
        fields: dict[str, Any] = {
            "dataset": self.info.name,
            "subject": groups.get("subject") or f"unknown:{path.stem}",
            "exercise": exercise,
            "exercise_id": self.layout["exercise_map"].get(exercise),
            "side": side if side in ("L", "R") else None,
            "view": self.layout["view"],
            "fps": float(self.layout["fps"]),
            "source": path,
            "joint_format": "unknown",
            "labels": {k: v for k, v in groups.items() if k not in ("subject", "exercise", "side")},
            "license": self.info.license,
            "citation": self.info.citation,
        }
        fields.update(kw)
        return Sequence(**fields)

    @abstractmethod
    def read(self, path: Path, groups: dict[str, str]) -> Sequence:
        """Read one file into a Sequence."""


class VideoLoader(PatternLoader):
    """Video files; frames are produced later by `rehabpose data extract-all`."""

    patterns = ("**/*.mp4", "**/*.avi", "**/*.mkv", "**/*.mov")

    def read(self, path: Path, groups: dict[str, str]) -> Sequence:
        """Reference the video (fps from the container if readable)."""
        from rehabpose.data.video_io import probe

        try:
            fps = probe(path).fps
        except (ValueError, FileNotFoundError):
            fps = float(self.layout["fps"])
        return self.base_sequence(path, groups, joint_format="video", video=path, fps=fps)


FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def read_skeleton_rows(path: Path, n_joints: int, min_match: float = 0.5) -> FloatArray:
    """Parse a text/CSV skeleton file into (T, n_joints, 3).

    Each data row must contain n_joints x 3 or n_joints x 4 numbers (x, y, z[, state]),
    optionally preceded by one index/timestamp column. Header/comment rows are skipped.
    Raises LayoutError if fewer than `min_match` of the numeric rows fit.
    """
    rows: list[list[float]] = []
    numeric = 0
    for line in Path(path).read_text(errors="replace").splitlines():
        vals = [float(v) for v in FLOAT_RE.findall(line)]
        if not vals:
            continue
        numeric += 1
        for k in (3, 4):
            if len(vals) == n_joints * k:
                rows.append(vals)
                break
            if len(vals) == n_joints * k + 1:
                rows.append(vals[1:])
                break
    if not rows or len(rows) < min_match * numeric:
        raise LayoutError(
            f"{path}: expected rows of {n_joints}x3 or {n_joints}x4 numbers; "
            f"{len(rows)} of {numeric} numeric rows matched"
        )
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise LayoutError(f"{path}: inconsistent row widths {sorted(widths)}")
    k = widths.pop() // n_joints
    return np.asarray(rows, dtype=np.float64).reshape(-1, n_joints, k)[:, :, :3]
