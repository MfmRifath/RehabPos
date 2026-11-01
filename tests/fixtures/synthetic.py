"""Synthetic MediaPipe-format skeletons with exactly known joint angles.

World frame (metres, MediaPipe convention): origin at mid-hip, y down, +x = subject's left,
forward = -z (towards a camera the subject faces). Hip width equals shoulder width so that the
shoulder->hip line is parallel to the trunk axis; then, by construction:

- upper_arm_trunk_angle == elevation_deg (3D always; 2D in the view whose plane holds the arm)
- elbow_interior == 180 - elbow_flexion_deg
- trunk_tilt == trunk_lean_deg (in the view whose plane holds the lean)
- forearm_rotation == rotation_deg (3D, no trunk lean/rotation)
- frontal shoulder_width ratio == cos(trunk_rotation_deg)

Views use orthographic projection: frontal sees the x-y plane, sagittal sees the y-z plane.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from rehabpose.core.landmarks import Frame, Side

View = Literal["frontal", "sagittal"]
F = NDArray[np.float64]

SHOULDER_HALF = 0.18
TRUNK_LEN = 0.50
UPPER_ARM = 0.30
FOREARM = 0.27

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28), (7, 0), (8, 0),
]  # fmt: skip


@dataclass
class SynthPose:
    """Parameters of one synthetic pose (degrees, metres)."""

    side: Side = "R"
    elevation_deg: float = 0.0  # upper arm vs trunk-down
    plane_deg: float = 0.0  # 0 = sagittal (flexion), 90 = frontal (abduction)
    elbow_flexion_deg: float = 0.0
    rotation_deg: float = 0.0  # forearm about upper-arm axis: 0 forward, +90 lateral
    trunk_lean_deg: float = 0.0
    trunk_lean_plane: View = "frontal"
    trunk_rotation_deg: float = 0.0  # axial, about vertical
    hike_m: float = 0.0  # exercising shoulder raised along the trunk axis
    other_arm_elevation_deg: float = 0.0


def _unit(v: F) -> F:
    return v / np.linalg.norm(v)


def _rot(axis: F, deg: float) -> F:
    """Rotation matrix about `axis` by `deg` (right-hand rule)."""
    a = _unit(axis)
    t = np.radians(deg)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * (k @ k)


def _arm(
    shoulder: F, lateral: F, elevation: float, plane: float, flex: float, rotation: float
) -> tuple[F, F]:
    down, fwd = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])
    el, pl = np.radians(elevation), np.radians(plane)
    d = np.cos(el) * down + np.sin(el) * (np.cos(pl) * fwd + np.sin(pl) * lateral)
    elbow = shoulder + UPPER_ARM * d
    b0 = fwd - np.dot(fwd, d) * d
    b0 = _unit(b0) if np.linalg.norm(b0) > 1e-6 else _unit(-down - np.dot(-down, d) * d)
    c0 = lateral - np.dot(lateral, d) * d - np.dot(lateral, b0) * b0
    c0 = _unit(c0) if np.linalg.norm(c0) > 1e-6 else _unit(np.cross(d, b0))
    r, a = np.radians(rotation), np.radians(flex)
    b = np.cos(r) * b0 + np.sin(r) * c0
    wrist = elbow + FOREARM * (np.cos(a) * d + np.sin(a) * b)
    return elbow, wrist


def world_skeleton(p: SynthPose) -> F:
    """(33, 3) world landmarks in metres for pose `p`."""
    w = np.zeros((33, 3))
    lat = {"L": np.array([1.0, 0.0, 0.0]), "R": np.array([-1.0, 0.0, 0.0])}
    sh = {
        "L": np.array([SHOULDER_HALF, -TRUNK_LEN, 0.0]),
        "R": np.array([-SHOULDER_HALF, -TRUNK_LEN, 0.0]),
    }
    up = np.array([0.0, -1.0, 0.0])
    sh[p.side] = sh[p.side] + p.hike_m * up

    w[11], w[12] = sh["L"], sh["R"]
    w[23], w[24] = [SHOULDER_HALF, 0, 0], [-SHOULDER_HALF, 0, 0]
    for s, (ie, iw) in {"L": (13, 15), "R": (14, 16)}.items():
        if s == p.side:
            e, wr = _arm(
                sh[s], lat[s], p.elevation_deg, p.plane_deg, p.elbow_flexion_deg, p.rotation_deg
            )
        else:
            e, wr = _arm(sh[s], lat[s], p.other_arm_elevation_deg, 90.0, 0.0, 0.0)
        w[ie], w[iw] = e, wr
        d = _unit(wr - e)
        base = 17 if s == "L" else 18  # pinky 17/18, index 19/20, thumb 21/22
        w[base], w[base + 2], w[base + 4] = wr + 0.08 * d, wr + 0.09 * d, wr + 0.05 * d

    head_y = -TRUNK_LEN - 0.18
    w[0] = [0, head_y + 0.02, -0.09]
    w[1:4] = [[0.03, head_y - 0.01, -0.08]] * 3
    w[4:7] = [[-0.03, head_y - 0.01, -0.08]] * 3
    w[7], w[8] = [0.07, head_y, 0.0], [-0.07, head_y, 0.0]
    w[9], w[10] = [0.025, head_y + 0.05, -0.08], [-0.025, head_y + 0.05, -0.08]
    for i, sgn in ((25, 1), (26, -1)):
        w[i] = [sgn * SHOULDER_HALF, 0.45, 0]  # knees
        w[i + 2] = [sgn * SHOULDER_HALF, 0.88, 0]  # ankles
        w[i + 4] = [sgn * SHOULDER_HALF, 0.92, 0.03]  # heels
        w[i + 6] = [sgn * SHOULDER_HALF, 0.92, -0.12]  # foot index

    lean_axis = (
        np.array([0.0, 0.0, 1.0]) if p.trunk_lean_plane == "frontal" else np.array([1.0, 0.0, 0.0])
    )
    rmat = _rot(lean_axis, p.trunk_lean_deg) @ _rot(np.array([0.0, 1.0, 0.0]), p.trunk_rotation_deg)
    return w @ rmat.T


def make_frame(
    p: SynthPose,
    view: View = "sagittal",
    *,
    img_w: int = 1280,
    img_h: int = 720,
    px_per_m: float = 400.0,
    noise_px: float = 0.0,
    noise_m: float = 0.0,
    occluded: Sequence[int] = (),
    visibility: float = 0.95,
    ts_ms: int = 0,
    frame_idx: int = 0,
    rng: np.random.Generator | None = None,
) -> Frame:
    """Project pose `p` into a MediaPipe-format Frame for the given camera view."""
    rng = rng or np.random.default_rng(0)
    world = world_skeleton(p)
    if view == "frontal":
        u, v, depth = world[:, 0], world[:, 1], world[:, 2]
    else:
        u, v, depth = -world[:, 2], world[:, 1], world[:, 0]
    px = np.stack([img_w / 2 + px_per_m * u, img_h * 0.55 + px_per_m * v], axis=1)
    if noise_px:
        px = px + rng.normal(0.0, noise_px, px.shape)
    vis = np.full(33, visibility)
    vis[list(occluded)] = 0.1
    lm = np.column_stack([px[:, 0] / img_w, px[:, 1] / img_h, px_per_m * depth / img_w, vis])
    wv = world + (rng.normal(0.0, noise_m, world.shape) if noise_m else 0.0)
    return Frame(lm, np.column_stack([wv, vis]), ts_ms, img_w, img_h, frame_idx)


def arm_raise_sequence(
    n_frames: int = 90, fps: float = 30.0, peak_deg: float = 150.0, view: View = "sagittal",
    side: Side = "R", plane_deg: float = 0.0, **frame_kw: object,
) -> list[Frame]:  # fmt: skip
    """One smooth (cosine) raise-and-lower of the arm over n_frames."""
    out = []
    for i in range(n_frames):
        el = peak_deg * 0.5 * (1 - np.cos(2 * np.pi * i / max(n_frames - 1, 1)))
        pose = SynthPose(side=side, elevation_deg=float(el), plane_deg=plane_deg)
        ts = int(round(i * 1000 / fps))
        out.append(make_frame(pose, view, ts_ms=ts, frame_idx=i, **frame_kw))  # type: ignore[arg-type]
    return out


def render_stick_figure(frame: Frame, thickness: int = 6) -> NDArray[np.uint8]:
    """Draw a frame as a black stick figure on a light background (BGR)."""
    img = np.full((frame.img_h, frame.img_w, 3), 235, dtype=np.uint8)
    pts = frame.pixel_xy.round().astype(int)
    for a, b in CONNECTIONS:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (30, 30, 30), thickness, cv2.LINE_AA)
    cv2.circle(img, tuple(pts[0]), 40, (30, 30, 30), thickness, cv2.LINE_AA)
    return img


@dataclass
class SyntheticVideo:
    """A written synthetic video and its ground-truth frames."""

    path: Path
    fps: float
    frames: list[Frame] = field(default_factory=list)


def write_synthetic_video(path: Path, n_frames: int = 30, fps: float = 30.0) -> SyntheticVideo:
    """Write a small stick-figure MP4 (640x360) of an arm raise; returns ground truth."""
    from rehabpose.data.video_io import VideoWriter

    frames = arm_raise_sequence(n_frames, fps, img_w=640, img_h=360, px_per_m=220.0)
    with VideoWriter(path, fps, (640, 360)) as wr:
        for fr in frames:
            wr.write(render_stick_figure(fr, thickness=4))
    return SyntheticVideo(Path(path), fps, frames)
