"""OpenCV window around LabelState. Display/key/mouse functions are injectable for tests."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from rehabpose.data.labeling.state import HELP, LabelState
from rehabpose.data.schema import Annotation, save_annotation

log = logging.getLogger(__name__)

WINDOW = "RehabPose labeler"
Image = NDArray[np.uint8]
MouseCallback = Callable[[int, int, int, int, Any], None]


class FrameSource:
    """Random-access frames from a video (sequential reads are cheap, seeks re-position)."""

    def __init__(self, path: Path) -> None:
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise ValueError(f"cannot open {path}")
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self._pos = -1
        self._last: Image | None = None

    def get(self, idx: int) -> Image:
        """Frame `idx` (BGR)."""
        if idx == self._pos and self._last is not None:
            return self._last
        if idx != self._pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = self.cap.read()
        if not ok:
            raise IndexError(f"frame {idx} unreadable")
        self._pos, self._last = idx, np.asarray(img, dtype=np.uint8)
        return self._last

    def close(self) -> None:
        """Release the capture."""
        self.cap.release()


def render(img: Image, st: LabelState) -> Image:
    """Draw the labelling HUD over a frame."""
    out = img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    # timeline with reps, segments and the cursor
    bar_y = h - 18
    cv2.rectangle(out, (0, bar_y), (w, h), (30, 30, 30), -1)
    scale = w / max(st.n_frames - 1, 1)
    colors = {
        "correct": (80, 200, 80),
        "incomplete_rom": (200, 160, 40),
        "too_fast": (40, 160, 230),
        "compensated": (60, 60, 230),
    }
    for r in st.reps:
        cv2.rectangle(
            out,
            (int(r.start_frame * scale), bar_y + 2),
            (int(r.end_frame * scale), h - 8),
            colors[r.label],
            -1,
        )
    for s in st.segments:
        cv2.line(
            out,
            (int(s.start_frame * scale), h - 4),
            (int(s.end_frame * scale), h - 4),
            (230, 60, 230),
            3,
        )
    cv2.line(
        out, (int(st.frame * scale), bar_y - 4), (int(st.frame * scale), h), (255, 255, 255), 2
    )
    mode = "PLAY" if st.playing else "pause"
    is_open = " (open)" if st.open_segment else ""
    lines = [
        f"frame {st.frame}/{st.n_frames - 1}  t={st.frame / st.fps:.2f}s  {mode}",
        f"reps {len(st.reps)}  marks {st.marks}  code {st.code}{is_open}",
        f"measure {st.measure}  clicks {len(st.clicks)}/3  angles {len(st.angles)}",
        st.message,
    ]
    for i, text in enumerate(lines):
        cv2.putText(out, text, (8, 20 + 18 * i), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (8, 20 + 18 * i), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, HELP, (8, bar_y - 8), font, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    for x, y in st.clicks:
        cv2.circle(out, (int(x), int(y)), 4, (0, 220, 255), -1)
    return out


def run_labeler(
    video: Path,
    state: LabelState,
    out_path: Path,
    *,
    imshow: Callable[[str, Image], None] = cv2.imshow,
    wait_key: Callable[[int], int] = cv2.waitKey,
    set_mouse: Callable[[str, MouseCallback], None] = cv2.setMouseCallback,
    named_window: Callable[[str], None] = cv2.namedWindow,
    destroy: Callable[[], None] = cv2.destroyAllWindows,
) -> Annotation:
    """Interactive loop; saves on `w` and on quit. Returns the final annotation."""
    src = FrameSource(video)
    state.n_frames = min(state.n_frames, src.n_frames) if state.n_frames else src.n_frames

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state.click(x, y)

    named_window(WINDOW)
    set_mouse(WINDOW, on_mouse)
    try:
        while True:
            imshow(WINDOW, render(src.get(state.frame), state))
            key = wait_key(max(1, int(1000 / state.fps)) if state.playing else 30)
            if key == -1 and state.playing:
                if state.frame >= state.n_frames - 1:
                    state.playing = False
                state.seek(1)
                continue
            if chr(key & 0xFF) == "w" and key >= 0:
                save_annotation(state.to_annotation(), out_path)
                state.dirty = False
                state.message = f"saved {out_path}"
                continue
            if not state.handle_key(key):
                break
    finally:
        src.close()
        destroy()
    ann = state.to_annotation()
    save_annotation(ann, out_path)
    log.info(
        "Saved %s (%d reps, %d segments, %d angles)",
        out_path,
        len(ann.reps),
        len(ann.compensation_segments),
        len(ann.angle_annotations),
    )
    return ann
