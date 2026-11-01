"""PoseEstimator protocol shared by all pose backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from rehabpose.core.landmarks import Frame


@runtime_checkable
class PoseEstimator(Protocol):
    """Single-person pose estimator for video frames."""

    def process(self, frame_bgr: NDArray[np.uint8], ts_ms: int) -> Frame | None:
        """Estimate pose on a BGR uint8 image at `ts_ms` (ms, increasing); None if no person."""
        ...

    def close(self) -> None:
        """Release backend resources."""
        ...
