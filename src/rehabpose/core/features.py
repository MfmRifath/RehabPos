"""Model features shared by training and live inference (pure numpy, no torch).

Per frame (2D space, F = 45):
  26  canonical xy of the 13-joint upper-body subset (mid-hip centred, / body_scale)
   3  angles / 180: upper_arm_trunk_angle, elbow_flexion, trunk_tilt (exercising side)
  13  joint visibilities
   3  causal angular velocities (deg/s / 180) of the 3 angles
3D space uses world xyz (39 coordinate features, F = 58).

"Canonical" = exercising side mapped to the subject's right: left-side data is mirrored in x
and L/R joints are swapped, so one model serves both arms. Joints below the visibility gate
and frames without a person are zeros (with visibility 0).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rehabpose.core import angles
from rehabpose.core.landmarks import NUM_LANDMARKS, Frame, Side

FloatArray = NDArray[np.float64]
Space = Literal["2d", "3d"]

UPPER13: tuple[int, ...] = (0, 7, 8, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24)
# positions inside UPPER13 of the (left, right) pairs, and of the canonical (right) arm
_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12))
_SWAP = list(range(13))
for _a, _b in _PAIRS:
    _SWAP[_a], _SWAP[_b] = _b, _a
L_SH, R_SH, R_EL, R_WR, L_HIP, R_HIP = 3, 4, 6, 8, 11, 12
VIS_GATE = angles.VISIBILITY_THRESHOLD
N_ANGLES = 3


def n_features(space: Space = "2d") -> int:
    """Feature vector length for a space."""
    return 13 * (2 if space == "2d" else 3) + N_ANGLES + 13 + N_ANGLES


def frames_to_arrays(
    frames: Sequence[Frame | None], space: Space = "2d"
) -> tuple[FloatArray, FloatArray]:
    """(T, 33, D) points (pixels for 2d, metres for 3d) and (T, 33) visibility (0 if missing)."""
    d = 2 if space == "2d" else 3
    pts = np.zeros((len(frames), NUM_LANDMARKS, d))
    vis = np.zeros((len(frames), NUM_LANDMARKS))
    for i, f in enumerate(frames):
        if f is None:
            continue
        pts[i] = f.pixel_xy if space == "2d" else f.world_xyz
        vis[i] = f.visibility
    return pts, vis


def canonical_points(pts: FloatArray, vis: FloatArray, side: Side) -> tuple[FloatArray, FloatArray]:
    """(T, 33, D), (T, 33) -> canonical (T, 13, D) coordinates and (T, 13) visibility."""
    p = np.asarray(pts, dtype=np.float64)[:, UPPER13, :].copy()
    v = np.asarray(vis, dtype=np.float64)[:, UPPER13].copy()
    hip = (p[:, L_HIP] + p[:, R_HIP]) / 2.0
    scale = angles.body_scale(np.asarray(pts, dtype=np.float64))
    scale = np.where(scale > 0, scale, 1.0)
    p = (p - hip[:, None, :]) / scale[:, None, None]
    if side == "L":
        p[..., 0] *= -1.0
        p, v = p[:, _SWAP], v[:, _SWAP]
    ok = v >= VIS_GATE
    p = np.where(ok[..., None], p, 0.0)
    v = np.where(ok, v, 0.0)
    return p, v


def canonical_angles(p13: FloatArray, v13: FloatArray) -> FloatArray:
    """(T, 3) angles in degrees from canonical points (0 where landmarks are not visible)."""
    return _angles_and_valid(p13, v13)[0]


def _angles_and_valid(p13: FloatArray, v13: FloatArray) -> tuple[FloatArray, NDArray[np.bool_]]:
    """(..., T, 13, D) points -> (..., T, 3) angles in degrees and validity mask."""
    p = np.asarray(p13, dtype=np.float64)
    up = angles.UP_2D if p.shape[-1] == 2 else angles.UP_3D
    s, e, w, h = p[..., R_SH, :], p[..., R_EL, :], p[..., R_WR, :], p[..., R_HIP, :]
    uat = angles.joint_angle(e, s, h)
    flex = 180.0 - angles.joint_angle(s, e, w)
    trunk = (p[..., L_SH, :] + p[..., R_SH, :]) / 2.0 - (p[..., L_HIP, :] + p[..., R_HIP, :]) / 2.0
    tilt = angles.angle_between(trunk, np.broadcast_to(up, trunk.shape))
    vis = np.asarray(v13) > 0
    masks = (
        vis[..., [R_SH, R_EL, L_HIP, R_HIP]].all(axis=-1),
        vis[..., [R_SH, R_EL, R_WR]].all(axis=-1),
        vis[..., [L_SH, R_SH, L_HIP, R_HIP]].all(axis=-1),
    )
    out = np.stack([uat, flex, tilt], axis=-1)
    valid = np.stack(masks, axis=-1) & np.isfinite(out)
    return np.asarray(np.where(valid, out, 0.0), dtype=np.float64), valid


def features_from_canonical(p13: FloatArray, v13: FloatArray, fps: float) -> FloatArray:
    """(..., T, F) features from canonical points; velocities are causal differences.

    Works on one window (T, 13, D) or a batch (N, T, 13, D).
    """
    p = np.asarray(p13, dtype=np.float64)
    ang, valid = _angles_and_valid(p, v13)
    vel = np.zeros_like(ang)
    if ang.shape[-2] > 1:  # only where both this and the previous angle are known
        both = valid[..., 1:, :] & valid[..., :-1, :]
        vel[..., 1:, :] = np.where(both, np.diff(ang, axis=-2) * fps, 0.0)
    coords = p.reshape(*p.shape[:-2], -1)
    return np.concatenate([coords, ang / 180.0, np.asarray(v13), vel / 180.0], axis=-1)


def sequence_features(
    frames: Sequence[Frame | None], side: Side, fps: float, space: Space = "2d"
) -> FloatArray:
    """(T, F) features for a whole sequence."""
    pts, vis = frames_to_arrays(frames, space)
    p13, v13 = canonical_points(pts, vis, side)
    return features_from_canonical(p13, v13, fps)


class StreamingFeatures:
    """Frame-by-frame features identical to `sequence_features` (causal velocity state)."""

    def __init__(self, side: Side, fps: float, space: Space = "2d") -> None:
        self.side, self.fps, self.space = side, fps, space
        self._prev: tuple[FloatArray, NDArray[np.bool_]] | None = None

    def reset(self) -> None:
        """Forget velocity state."""
        self._prev = None

    def update(self, frame: Frame | None) -> FloatArray:
        """Features for one frame, shape (F,)."""
        pts, vis = frames_to_arrays([frame], self.space)
        p13, v13 = canonical_points(pts, vis, self.side)
        ang_all, valid_all = _angles_and_valid(p13, v13)
        ang, valid = ang_all[0], valid_all[0]
        vel = np.zeros(N_ANGLES)
        if self._prev is not None:
            prev_ang, prev_valid = self._prev
            vel = np.where(valid & prev_valid, (ang - prev_ang) * self.fps, 0.0)
        self._prev = (ang, valid)
        return np.concatenate([p13.reshape(-1), ang / 180.0, v13[0], vel / 180.0])
