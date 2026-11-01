"""Joint-angle geometry and RehabPose measures (degrees), 2D pixel or 3D world space.

All array functions broadcast over leading axes: points have shape (..., 33, D) with D = 2 or 3,
so a single frame (33, D) yields a 0-d array and a sequence (T, 33, D) yields shape (T,).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rehabpose.core.landmarks import LM, Frame, Side, side_joints

FloatArray = NDArray[np.float64]
Space = Literal["2d", "3d"]

VISIBILITY_THRESHOLD = 0.6
# Image and MediaPipe world coordinates both have y pointing down.
UP_2D = np.array([0.0, -1.0])
UP_3D = np.array([0.0, -1.0, 0.0])

MEASURES = (
    "upper_arm_trunk_angle",
    "elbow_interior",
    "elbow_flexion",
    "trunk_tilt",
    "shoulder_width",
    "hike_ratio",
    "forearm_rotation",
    "plane_drift_m",
    "elbow_away",
)


def _f(x: object) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def _angle_single(u: list[float], v: list[float]) -> float:
    """angle_between for one pair of vectors without numpy overhead (hot path)."""
    if len(u) == 2:
        cross = abs(u[0] * v[1] - u[1] * v[0])
    elif len(u) == 3:
        cx = u[1] * v[2] - u[2] * v[1]
        cy = u[2] * v[0] - u[0] * v[2]
        cz = u[0] * v[1] - u[1] * v[0]
        cross = math.sqrt(cx * cx + cy * cy + cz * cz)
    else:
        raise ValueError(f"vectors must be 2D or 3D, got dimension {len(u)}")
    if not any(u) or not any(v):
        return math.nan
    return math.degrees(math.atan2(cross, sum(a * b for a, b in zip(u, v, strict=True))))


def angle_between(u: FloatArray, v: FloatArray) -> FloatArray:
    """Unsigned angle between vectors along the last axis (2D or 3D), degrees in [0, 180].

    Uses atan2(|u x v|, u . v) for numerical stability; NaN if either vector has zero length.
    """
    u, v = _f(u), _f(v)
    if u.ndim == 1 and v.ndim == 1 and u.shape == v.shape:
        return _f(_angle_single(u.tolist(), v.tolist()))
    u, v = np.broadcast_arrays(u, v)
    if u.shape[-1] == 2:
        cross = np.abs(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])
    elif u.shape[-1] == 3:
        cross = np.linalg.norm(np.cross(u, v), axis=-1)
    else:
        raise ValueError(f"vectors must be 2D or 3D, got dimension {u.shape[-1]}")
    dot = np.sum(u * v, axis=-1)
    degenerate = (np.linalg.norm(u, axis=-1) == 0) | (np.linalg.norm(v, axis=-1) == 0)
    return np.where(degenerate, np.nan, np.degrees(np.arctan2(cross, dot)))


def joint_angle(a: FloatArray, b: FloatArray, c: FloatArray) -> FloatArray:
    """Angle at vertex b between rays b->a and b->c, degrees in [0, 180]."""
    return angle_between(_f(a) - _f(b), _f(c) - _f(b))


def _up(points: FloatArray) -> FloatArray:
    return UP_2D if points.shape[-1] == 2 else UP_3D


def upper_arm_trunk_angle(points: FloatArray, side: Side) -> FloatArray:
    """angle(S->E, S->H) in degrees: 0 = arm at side, 180 = overhead."""
    j = side_joints(side)
    p = _f(points)
    return joint_angle(p[..., j.elbow, :], p[..., j.shoulder, :], p[..., j.hip, :])


def elbow_interior(points: FloatArray, side: Side) -> FloatArray:
    """angle(E->S, E->W) in degrees: 180 = straight arm."""
    j = side_joints(side)
    p = _f(points)
    return joint_angle(p[..., j.shoulder, :], p[..., j.elbow, :], p[..., j.wrist, :])


def elbow_flexion(points: FloatArray, side: Side) -> FloatArray:
    """180 - elbow_interior, degrees: 0 = straight, larger = more flexed."""
    return 180.0 - elbow_interior(points, side)


def mid_point(points: FloatArray, i: int, k: int) -> FloatArray:
    """Midpoint of landmarks i and k."""
    p = _f(points)
    return (p[..., i, :] + p[..., k, :]) / 2.0


def trunk_tilt(points: FloatArray) -> FloatArray:
    """angle(mid_hip->mid_shoulder, vertical up), degrees: 0 = upright."""
    p = _f(points)
    trunk = mid_point(p, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER) - mid_point(
        p, LM.LEFT_HIP, LM.RIGHT_HIP
    )
    return angle_between(trunk, np.broadcast_to(_up(p), trunk.shape))


def trunk_tilt_side(points: FloatArray, side: Side) -> FloatArray:
    """angle(hip->shoulder on one side, vertical up), degrees: trunk tilt for side-on views.

    A side-on camera usually cannot see the far shoulder and hip, so the midpoint form is unknown.
    """
    j = side_joints(side)
    p = _f(points)
    trunk = p[..., j.shoulder, :] - p[..., j.hip, :]
    return angle_between(trunk, np.broadcast_to(_up(p), trunk.shape))


def shoulder_width(points: FloatArray) -> FloatArray:
    """Distance between landmarks 11 and 12 (pixels in 2D, metres in 3D)."""
    p = _f(points)
    return _f(np.linalg.norm(p[..., LM.LEFT_SHOULDER, :] - p[..., LM.RIGHT_SHOULDER, :], axis=-1))


SHOULDER_TO_TORSO = 0.7  # approx. adult shoulder width / torso length


def body_scale(points: FloatArray) -> FloatArray:
    """Normalization unit: shoulder width, floored at 0.7 x torso length (same units as input).

    In side-on (sagittal) 2D views the shoulders overlap and the projected width collapses
    towards 0; the torso-length floor keeps normalized speeds/distances well defined.
    """
    p = _f(points)
    torso = np.linalg.norm(
        mid_point(p, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER) - mid_point(p, LM.LEFT_HIP, LM.RIGHT_HIP),
        axis=-1,
    )
    return _f(np.maximum(shoulder_width(p), SHOULDER_TO_TORSO * torso))


def hike_ratio(points: FloatArray, side: Side) -> FloatArray:
    """(shoulder_y - ear_y) / shoulder_width on the exercising side (2D, y down).

    Positive when the ear is above the shoulder; decreases when the shoulder hikes.
    NaN when shoulder width is 0 (shoulders overlap, e.g. a pure sagittal view).
    """
    j = side_joints(side)
    p = _f(points)
    width = shoulder_width(p[..., :2])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (p[..., j.shoulder, 1] - p[..., j.ear, 1]) / width
    return _f(np.where(width > 0, ratio, np.nan))


def forearm_rotation(world: FloatArray, side: Side) -> FloatArray:
    """Experimental: forearm rotation about the upper-arm axis (3D world), degrees.

    0 = forearm pointing forward (trunk forward direction), +90 = pointing laterally
    (external rotation), negative = medial. Range (-180, 180].
    """
    j = side_joints(side)
    p = _f(world)
    s, e, w = p[..., j.shoulder, :], p[..., j.elbow, :], p[..., j.wrist, :]
    axis = e - s
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    l_minus_r = p[..., LM.LEFT_SHOULDER, :] - p[..., LM.RIGHT_SHOULDER, :]
    forward = np.cross(l_minus_r, np.broadcast_to(UP_3D, l_minus_r.shape))
    lateral = l_minus_r if side == "L" else -l_minus_r

    def project(v: FloatArray) -> FloatArray:
        return v - np.sum(v * axis, axis=-1, keepdims=True) * axis

    f, fw, lat = project(w - e), project(forward), project(lateral)
    fw = fw / np.linalg.norm(fw, axis=-1, keepdims=True)
    lat = lat / np.linalg.norm(lat, axis=-1, keepdims=True)
    return _f(np.degrees(np.arctan2(np.sum(f * lat, axis=-1), np.sum(f * fw, axis=-1))))


def plane_drift(world: FloatArray, side: Side) -> FloatArray:
    """Experimental: |wrist_z - shoulder_z| in world metres (out-of-plane arm motion)."""
    j = side_joints(side)
    p = _f(world)
    return _f(np.abs(p[..., j.wrist, 2] - p[..., j.shoulder, 2]))


def elbow_away(points: FloatArray, side: Side) -> FloatArray:
    """|elbow_x - hip_x| / body_scale (2D): how far the elbow leaves the side of the trunk."""
    j = side_joints(side)
    p = _f(points)
    return _f(np.abs(p[..., j.elbow, 0] - p[..., j.hip, 0]) / body_scale(p))


def reach_ratio(points: FloatArray, side: Side, rest_wrist: FloatArray) -> FloatArray:
    """Optional: horizontal wrist displacement from its rest position / arm length (2D)."""
    j = side_joints(side)
    p = _f(points)
    s, e, w = p[..., j.shoulder, :], p[..., j.elbow, :], p[..., j.wrist, :]
    arm = np.linalg.norm(e - s, axis=-1) + np.linalg.norm(w - e, axis=-1)
    return _f(np.abs(w[..., 0] - _f(rest_wrist)[..., 0]) / arm)


def required_landmarks(measure: str, side: Side) -> tuple[int, ...]:
    """Landmark indices a measure depends on (for the visibility gate)."""
    j = side_joints(side)
    torso = (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP)
    table: dict[str, tuple[int, ...]] = {
        "upper_arm_trunk_angle": (j.shoulder, j.elbow, j.hip),
        "elbow_interior": (j.shoulder, j.elbow, j.wrist),
        "elbow_flexion": (j.shoulder, j.elbow, j.wrist),
        "trunk_tilt": torso,
        "shoulder_width": (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
        "hike_ratio": (j.ear, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
        "forearm_rotation": (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, j.elbow, j.wrist),
        "plane_drift_m": (j.shoulder, j.wrist),
        "elbow_away": (*torso, j.elbow),
        "reach_ratio": (j.shoulder, j.elbow, j.wrist),
    }
    if measure not in table:
        raise ValueError(f"unknown measure {measure!r}")
    return table[measure]


def is_visible(
    visibility: FloatArray, indices: tuple[int, ...], threshold: float = VISIBILITY_THRESHOLD
) -> bool:
    """True if every landmark in `indices` has visibility >= threshold."""
    return all(visibility[i] >= threshold for i in indices)


def compute_measures(
    frame: Frame,
    side: Side,
    space: Space = "2d",
    threshold: float = VISIBILITY_THRESHOLD,
    names: tuple[str, ...] = MEASURES,
) -> dict[str, float | None]:
    """Measures `names` for one frame; a measure is None ("unknown") if visibility gating fails.

    Angles in degrees; shoulder_width in pixels (2d) or metres (3d); forearm_rotation and
    plane_drift_m always use 3D world landmarks; hike_ratio and elbow_away always 2D.
    """
    pts = frame.pixel_xy if space == "2d" else frame.world_xyz
    funcs: dict[str, Callable[[], FloatArray]] = {
        "upper_arm_trunk_angle": lambda: upper_arm_trunk_angle(pts, side),
        "elbow_interior": lambda: elbow_interior(pts, side),
        "elbow_flexion": lambda: elbow_flexion(pts, side),
        "trunk_tilt": lambda: trunk_tilt(pts),
        "shoulder_width": lambda: shoulder_width(pts),
        "hike_ratio": lambda: hike_ratio(frame.pixel_xy, side),
        "forearm_rotation": lambda: forearm_rotation(frame.world_xyz, side),
        "plane_drift_m": lambda: plane_drift(frame.world_xyz, side),
        "elbow_away": lambda: elbow_away(frame.pixel_xy, side),
    }
    out: dict[str, float | None] = {}
    for name in names:
        fn = funcs[name]
        if not is_visible(frame.visibility, required_landmarks(name, side), threshold):
            out[name] = None
            continue
        value = float(fn())
        out[name] = value if np.isfinite(value) else None
    return out
