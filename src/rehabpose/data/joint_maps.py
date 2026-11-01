"""Map Kinect v2 / mocap skeletons to a 13-joint upper-body subset in MediaPipe indices.

Target subset UPPER13 (MediaPipe indices): nose 0, ears 7/8, shoulders 11/12, elbows 13/14,
wrists 15/16, index fingers 19/20, hips 23/24. Unmapped MediaPipe landmarks get visibility 0.

Approximations (documented, see docs/05 §2):
- Kinect has no face points: nose and both ears <- Head. Vertical ear-shoulder distance
  (hike_ratio) is approximate; horizontal ear positions are meaningless.
- Kinect HandTip -> index finger; Kinect hips are joint centres (MediaPipe hips are similar).
- Plug-in Gait: ears <- midpoint of front/back head markers per side, nose <- midpoint of
  LFHD/RFHD, wrists <- midpoint of WRA/WRB, hips <- midpoint of ASI/PSI per side (a crude
  hip-joint-centre proxy, offset from the true centre).

Coordinate conventions converted to MediaPipe world (metres, y down, +x = subject's left):
- Kinect camera space (metres; +x sensor's left, +y up, +z away from sensor):
  (x, y, z) -> (-x, -y, z), a 180 deg rotation about z (chirality preserved).
- Mocap lab frame (millimetres, z up, assumed): (x, y, z) -> (x, -z, y) / 1000.
Results are re-centred on the mid-hip, as MediaPipe world landmarks are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rehabpose.core.landmarks import NUM_LANDMARKS, Frame

FloatArray = NDArray[np.float64]
View = Literal["frontal", "sagittal"]

UPPER13: tuple[int, ...] = (0, 7, 8, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24)

KINECT_V2_JOINTS: tuple[str, ...] = (
    "SpineBase", "SpineMid", "Neck", "Head",
    "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft",
    "ShoulderRight", "ElbowRight", "WristRight", "HandRight",
    "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",
    "HipRight", "KneeRight", "AnkleRight", "FootRight",
    "SpineShoulder", "HandTipLeft", "ThumbLeft", "HandTipRight", "ThumbRight",
)  # fmt: skip


@dataclass(frozen=True)
class JointMap:
    """Source-joint names (averaged) for each MediaPipe index, plus the source frame."""

    name: str
    joints: tuple[str, ...]
    to_mp: dict[int, tuple[str, ...]]
    frame: Literal["kinect_camera", "mocap_zup_mm"]
    approximate: frozenset[int] = field(default_factory=frozenset)

    def indices(self, mp_index: int) -> list[int]:
        """Source column indices averaged for one MediaPipe landmark."""
        return [self.joints.index(n) for n in self.to_mp[mp_index]]


KINECT_V2 = JointMap(
    name="kinect_v2_25",
    joints=KINECT_V2_JOINTS,
    to_mp={
        0: ("Head",),
        7: ("Head",),
        8: ("Head",),
        11: ("ShoulderLeft",),
        12: ("ShoulderRight",),
        13: ("ElbowLeft",),
        14: ("ElbowRight",),
        15: ("WristLeft",),
        16: ("WristRight",),
        19: ("HandTipLeft",),
        20: ("HandTipRight",),
        23: ("HipLeft",),
        24: ("HipRight",),
    },  # fmt: skip
    frame="kinect_camera",
    approximate=frozenset({0, 7, 8, 19, 20}),
)

PIG_MARKERS: tuple[str, ...] = (
    "LFHD", "RFHD", "LBHD", "RBHD", "LSHO", "RSHO", "LELB", "RELB",
    "LWRA", "LWRB", "RWRA", "RWRB", "LFIN", "RFIN", "LASI", "RASI", "LPSI", "RPSI",
)  # fmt: skip

PLUG_IN_GAIT = JointMap(
    name="plug_in_gait",
    joints=PIG_MARKERS,
    to_mp={
        0: ("LFHD", "RFHD"),
        7: ("LFHD", "LBHD"),
        8: ("RFHD", "RBHD"),
        11: ("LSHO",),
        12: ("RSHO",),
        13: ("LELB",),
        14: ("RELB",),
        15: ("LWRA", "LWRB"),
        16: ("RWRA", "RWRB"),
        19: ("LFIN",),
        20: ("RFIN",),
        23: ("LASI", "LPSI"),
        24: ("RASI", "RPSI"),
    },  # fmt: skip
    frame="mocap_zup_mm",
    approximate=frozenset({0, 7, 8, 23, 24}),
)

JOINT_MAPS: dict[str, JointMap] = {m.name: m for m in (KINECT_V2, PLUG_IN_GAIT)}


def _to_mp_axes(xyz: FloatArray, frame: str) -> FloatArray:
    if frame == "kinect_camera":
        return xyz * np.array([-1.0, -1.0, 1.0])
    if frame == "mocap_zup_mm":
        return np.stack([xyz[..., 0], -xyz[..., 2], xyz[..., 1]], axis=-1) / 1000.0
    raise ValueError(f"unknown source frame {frame!r}")


def to_mediapipe_world(skeleton: FloatArray, joint_map: JointMap) -> FloatArray:
    """(T, J, 3) source skeleton -> (T, 33, 4) MediaPipe-style world landmarks (+visibility).

    Mapped landmarks get visibility 1 unless any source joint is non-finite or all-zero
    (untracked), in which case visibility is 0. Coordinates are centred on the mid-hip.
    """
    skel = np.asarray(skeleton, dtype=np.float64)
    if skel.ndim != 3 or skel.shape[1:] != (len(joint_map.joints), 3):
        raise ValueError(
            f"expected (T, {len(joint_map.joints)}, 3) for {joint_map.name}, got {skel.shape}"
        )
    t = skel.shape[0]
    out = np.zeros((t, NUM_LANDMARKS, 4))
    for mp_idx in joint_map.to_mp:
        src = skel[:, joint_map.indices(mp_idx), :]
        bad = ~np.isfinite(src).all(axis=(1, 2)) | (np.abs(src).sum(axis=2) == 0).any(axis=1)
        # rows with any bad source joint are zeroed below, so NaN -> 0 before averaging is safe
        pts = _to_mp_axes(np.nan_to_num(src).mean(axis=1), joint_map.frame)
        out[:, mp_idx, :3] = np.where(bad[:, None], 0.0, pts)
        out[:, mp_idx, 3] = np.where(bad, 0.0, 1.0)
    hips_ok = (out[:, 23, 3] > 0) & (out[:, 24, 3] > 0)
    mid = np.where(hips_ok[:, None], (out[:, 23, :3] + out[:, 24, :3]) / 2, 0.0)
    mapped = out[:, :, 3] > 0
    out[:, :, :3] = np.where(mapped[..., None], out[:, :, :3] - mid[:, None, :], 0.0)
    return out


def project_world(
    world: FloatArray,
    view: View,
    img_w: int = 1280,
    img_h: int = 720,
    px_per_m: float = 400.0,
) -> FloatArray:
    """Orthographic projection of (T, 33, 4) world landmarks to normalized image landmarks.

    frontal camera sees the x-y plane, sagittal the y-z plane (forward = -z to the right).
    """
    w = np.asarray(world, dtype=np.float64)
    if view == "frontal":
        u, v, depth = w[..., 0], w[..., 1], w[..., 2]
    else:
        u, v, depth = -w[..., 2], w[..., 1], w[..., 0]
    lm = np.empty_like(w)
    lm[..., 0] = (img_w / 2 + px_per_m * u) / img_w
    lm[..., 1] = (img_h * 0.55 + px_per_m * v) / img_h
    lm[..., 2] = px_per_m * depth / img_w
    lm[..., 3] = w[..., 3]
    return lm


def skeleton_to_frames(
    skeleton: FloatArray,
    joint_map: JointMap,
    fps: float,
    view: View = "frontal",
    img_w: int = 1280,
    img_h: int = 720,
    px_per_m: float = 400.0,
) -> list[Frame]:
    """Convert a source skeleton sequence to engine Frames (virtual orthographic camera)."""
    world = to_mediapipe_world(skeleton, joint_map)
    lm = project_world(world, view, img_w, img_h, px_per_m)
    return [
        Frame(lm[i], world[i], int(round(i * 1000.0 / fps)), img_w, img_h, i)
        for i in range(world.shape[0])
    ]
