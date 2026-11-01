"""Pose estimation backends behind the PoseEstimator protocol."""

from __future__ import annotations

from pathlib import Path

from rehabpose.pose.base import PoseEstimator
from rehabpose.pose.model_files import ModelVariant, resolve_model


def create_estimator(
    variant: ModelVariant, models_dir: Path, *, download: bool = False
) -> PoseEstimator:
    """Build the default MediaPipe estimator (CPU delegate) for a model variant."""
    from rehabpose.pose.mediapipe_pose import MediaPipePose

    return MediaPipePose(resolve_model(variant, models_dir, download=download))


__all__ = ["PoseEstimator", "create_estimator"]
