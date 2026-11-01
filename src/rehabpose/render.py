"""OpenCV overlay drawing for annotated videos (skeleton, angle, rep count, cue)."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from rehabpose.core.engine import FeedbackFrame
from rehabpose.core.exercise import ExerciseConfig
from rehabpose.core.landmarks import Frame, Side, side_joints

BGR = tuple[int, int, int]
SKELETON = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]  # fmt: skip
SEVERITY_COLOR: dict[str, BGR] = {
    "info": (200, 120, 30),
    "warn": (0, 140, 255),
    "error": (40, 40, 220),
}
FOOTER = "Research prototype - not a medical device"


def _text(
    img: NDArray[np.uint8],
    text: str,
    org: tuple[int, int],
    scale: float,
    color: BGR,
    box: bool = True,
) -> None:
    """Draw text at `org` (baseline-left), optionally over a translucent dark box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = max(1, round(scale * 1.6))
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    if box:
        pad = max(2, round(4 * scale))
        x0, y0 = max(org[0] - pad, 0), max(org[1] - th - pad, 0)
        x1, y1 = min(org[0] + tw + pad, img.shape[1]), min(org[1] + base + pad, img.shape[0])
        if x1 > x0 and y1 > y0:
            roi = img[y0:y1, x0:x1]
            img[y0:y1, x0:x1] = (roi * 0.45).astype(np.uint8)
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)


def draw_overlay(
    img: NDArray[np.uint8],
    frame: Frame | None,
    fb: FeedbackFrame,
    exercise: ExerciseConfig,
    side: Side,
    visibility_threshold: float = 0.6,
) -> NDArray[np.uint8]:
    """Draw the analysis overlay onto a copy of `img` (BGR)."""
    out = img.copy()
    h, w = out.shape[:2]
    s = max(h / 720.0, 0.5)
    j = side_joints(side)
    active = {j.shoulder, j.elbow, j.wrist, j.hip}

    if frame is not None:
        pts = frame.pixel_xy.round().astype(int)
        vis = frame.visibility
        for a, b in SKELETON:
            if vis[a] >= visibility_threshold and vis[b] >= visibility_threshold:
                col: BGR = (0, 220, 255) if a in active and b in active else (230, 230, 230)
                cv2.line(out, tuple(pts[a]), tuple(pts[b]), col, max(2, int(3 * s)), cv2.LINE_AA)
        for i in active:
            if vis[i] >= visibility_threshold:
                cv2.circle(out, tuple(pts[i]), max(4, int(7 * s)), (0, 220, 255), -1, cv2.LINE_AA)
        value = fb.angles.get(exercise.primary_angle)
        vertex = j.elbow if exercise.primary_angle == "elbow_flexion" else j.shoulder
        if value is not None and vis[vertex] >= visibility_threshold:
            unit = "" if exercise.unit == "ratio" else " deg"
            label = f"{value:.2f}{unit}" if exercise.unit == "ratio" else f"{value:.0f}{unit}"
            x, y = pts[vertex]
            _text(out, label, (int(x + 12 * s), int(y - 12 * s)), 0.8 * s, (0, 220, 255))

    hud = [
        exercise.name + (" (experimental)" if exercise.experimental else ""),
        f"Reps: {fb.rep_count}   Phase: {fb.phase}",
        f"Target: {exercise.target:g}",
    ]
    if fb.last_rep is not None:
        hud.append(f"Last: {fb.last_rep.peak_angle:g} / {fb.last_rep.label}")
    for k, line in enumerate(hud):
        _text(out, line, (int(14 * s), int((34 + 30 * k) * s)), 0.75 * s, (255, 255, 255))

    if fb.cue is not None:
        color = SEVERITY_COLOR.get(fb.cue.severity, (200, 120, 30))
        bh = int(56 * s)
        cv2.rectangle(out, (0, h - bh - int(26 * s)), (w, h - int(26 * s)), color, -1)
        _text(
            out,
            fb.cue.text,
            (int(16 * s), h - int(26 * s) - bh // 3),
            0.9 * s,
            (255, 255, 255),
            False,
        )
    _text(out, FOOTER, (int(10 * s), h - int(8 * s)), 0.5 * s, (220, 220, 220))
    return out
