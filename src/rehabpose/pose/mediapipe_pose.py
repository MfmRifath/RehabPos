"""MediaPipe Tasks PoseLandmarker backend (VIDEO running mode, CPU delegate by default)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from rehabpose.core.landmarks import NUM_LANDMARKS, Frame

log = logging.getLogger(__name__)

Delegate = Literal["cpu", "gpu"]


def _to_array(landmarks: Any) -> NDArray[np.float64]:
    """Convert a list of 33 MediaPipe landmarks to (33, 4) [x, y, z, visibility]."""
    arr = np.array(
        [
            [p.x or 0.0, p.y or 0.0, p.z or 0.0, p.visibility if p.visibility is not None else 0.0]
            for p in landmarks
        ],
        dtype=np.float64,
    )
    if arr.shape != (NUM_LANDMARKS, 4):
        raise ValueError(f"expected 33 landmarks, got {arr.shape[0]}")
    return arr


def result_to_frame(result: Any, ts_ms: int, img_w: int, img_h: int) -> Frame | None:
    """Convert a PoseLandmarkerResult to a Frame (first person only); None if no detection."""
    if not result.pose_landmarks:
        return None
    lm = _to_array(result.pose_landmarks[0])
    if result.pose_world_landmarks:
        world = _to_array(result.pose_world_landmarks[0])
    else:
        world = np.zeros((NUM_LANDMARKS, 4))
    # World landmarks carry the same visibility as image landmarks.
    world[:, 3] = lm[:, 3]
    return Frame(lm=lm, world=world, ts_ms=ts_ms, img_w=img_w, img_h=img_h)


class MediaPipePose:
    """PoseEstimator using mp.tasks.vision.PoseLandmarker in VIDEO mode.

    Timestamps passed to detect_for_video are forced to be strictly increasing, as the
    Tasks API requires; the Frame keeps the caller's ts_ms.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        delegate: Delegate = "cpu",
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path),
                delegate=BaseOptions.Delegate.GPU
                if delegate == "gpu"
                else BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1
        log.debug("MediaPipe PoseLandmarker ready (%s, delegate=%s)", model_path, delegate)

    def process(self, frame_bgr: NDArray[np.uint8], ts_ms: int) -> Frame | None:
        """Run pose on one BGR frame; returns None if no person is detected."""
        mp_ts = max(int(ts_ms), self._last_ts + 1)
        self._last_ts = mp_ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, mp_ts)
        h, w = frame_bgr.shape[:2]
        return result_to_frame(result, int(ts_ms), w, h)

    def close(self) -> None:
        """Release the landmarker."""
        self._landmarker.close()

    def __enter__(self) -> MediaPipePose:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
