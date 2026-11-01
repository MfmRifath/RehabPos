"""MediaPipe 33-landmark frames, side selection and JSON (de)serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

NUM_LANDMARKS = 33
Side = Literal["L", "R"]
FloatArray = NDArray[np.float64]


class LM(IntEnum):
    """MediaPipe pose landmark indices used by RehabPose."""

    NOSE = 0
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24


@dataclass(frozen=True, slots=True)
class SideJoints:
    """Landmark indices of one body side."""

    ear: int
    shoulder: int
    elbow: int
    wrist: int
    hip: int


SIDE_JOINTS: dict[Side, SideJoints] = {
    "L": SideJoints(LM.LEFT_EAR, LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST, LM.LEFT_HIP),
    "R": SideJoints(LM.RIGHT_EAR, LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST, LM.RIGHT_HIP),
}


def side_joints(side: Side) -> SideJoints:
    """Return landmark indices for the subject's anatomical side "L" or "R"."""
    try:
        return SIDE_JOINTS[side]
    except KeyError:
        raise ValueError(f"side must be 'L' or 'R', got {side!r}") from None


def _as_landmark_array(value: Any, name: str) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (NUM_LANDMARKS, 4):
        raise ValueError(f"{name} must have shape (33, 4), got {arr.shape}")
    return arr


@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    """One pose observation.

    lm: (33, 4) normalized image coords (x, y in [0, 1], z relative depth, visibility).
    world: (33, 4) hip-centred metres (x, y, z, visibility).
    ts_ms: source timestamp in milliseconds; img_w/img_h: image size in pixels.
    """

    lm: FloatArray
    world: FloatArray
    ts_ms: int
    img_w: int
    img_h: int
    frame_idx: int = 0
    _px: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lm", _as_landmark_array(self.lm, "lm"))
        object.__setattr__(self, "world", _as_landmark_array(self.world, "world"))
        if self.img_w <= 0 or self.img_h <= 0:
            raise ValueError("img_w and img_h must be positive")
        scale = np.array([self.img_w, self.img_h], dtype=np.float64)
        object.__setattr__(self, "_px", self.lm[:, :2] * scale)

    @property
    def visibility(self) -> FloatArray:
        """Per-landmark visibility in [0, 1], shape (33,)."""
        return self.lm[:, 3]

    @property
    def pixel_xy(self) -> FloatArray:
        """Pixel-scaled 2D coordinates (x*w, y*h), shape (33, 2)."""
        return self._px

    @property
    def world_xyz(self) -> FloatArray:
        """World coordinates in metres, shape (33, 3)."""
        return self.world[:, :3]

    def to_dict(self) -> dict[str, Any]:
        """Serialize using the WS protocol v1 field names."""
        return {
            "frame": self.frame_idx,
            "ts_ms": self.ts_ms,
            "img": {"w": self.img_w, "h": self.img_h},
            "lm": self.lm.tolist(),
            "world": self.world.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Frame:
        """Parse a dict with WS protocol v1 field names."""
        return cls(
            lm=data["lm"],
            world=data["world"],
            ts_ms=int(data["ts_ms"]),
            img_w=int(data["img"]["w"]),
            img_h=int(data["img"]["h"]),
            frame_idx=int(data.get("frame", 0)),
        )

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> Frame:
        """Parse a JSON string produced by to_json (or a WS landmarks message)."""
        data: dict[str, Any] = json.loads(text)
        return cls.from_dict(data)
