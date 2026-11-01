"""Dataset-level jobs: statistics and bulk landmark extraction/conversion to parquet."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from rehabpose.core.landmarks import Frame
from rehabpose.data.extract import extract_video, frames_to_table
from rehabpose.data.joint_maps import JOINT_MAPS, skeleton_to_frames
from rehabpose.data.loaders.base import DatasetLoader, Sequence
from rehabpose.pose.base import PoseEstimator

log = logging.getLogger(__name__)


def dataset_stats(sequences: Iterable[Sequence]) -> dict[str, Any]:
    """Counts over sequences: subjects, exercises, joint formats, labels, frames."""
    seqs = list(sequences)
    labels: Counter[str] = Counter()
    for s in seqs:
        for key in ("correct", "complete", "correctness", "label"):
            if key in s.labels:
                labels[f"{key}={s.labels[key]}"] += 1
        if isinstance(s.labels.get("reps"), list):
            labels.update(f"rep:{r['label']}" for r in s.labels["reps"])
    frames = [n for s in seqs if (n := s.n_frames) is not None]
    return {
        "dataset": seqs[0].dataset if seqs else None,
        "sequences": len(seqs),
        "subjects": len({s.subject for s in seqs}),
        "unknown_subject_ids": sum(s.subject.startswith("unknown:") for s in seqs),
        "exercises": dict(Counter(s.exercise for s in seqs)),
        "mapped_to_rehabpose": sum(s.exercise_id is not None for s in seqs),
        "joint_formats": dict(Counter(s.joint_format for s in seqs)),
        "labels": dict(labels),
        "frames_known": sum(frames),
        "sequences_with_frames": len(frames),
    }


def _out_path(loader: DatasetLoader, seq: Sequence, out_root: Path) -> Path:
    try:
        rel = seq.source.relative_to(loader.root)
    except ValueError:
        rel = Path(seq.source.name)
    return Path(out_root) / loader.info.name / rel.with_suffix(".parquet")


def _meta(seq: Sequence, **extra: Any) -> dict[str, Any]:
    return {
        "dataset": seq.dataset,
        "subject": seq.subject,
        "exercise": seq.exercise,
        "exercise_id": seq.exercise_id,
        "source": str(seq.source),
        "fps": seq.fps,
        "license": seq.license,
        **extra,
    }


def extract_all(
    loader: DatasetLoader,
    out_root: Path,
    estimator_factory: Callable[[], PoseEstimator] | None = None,
    model: str = "full",
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    """Write one landmark parquet per sequence under <out_root>/<dataset>/.

    Videos are pose-extracted with a fresh estimator each (VIDEO mode needs increasing
    timestamps); skeletons with a known joint map are converted via a virtual camera.
    """
    counts: Counter[str] = Counter()
    for seq in loader.sequences():
        if limit is not None and counts["written"] >= limit:
            break
        out = _out_path(loader, seq, out_root)
        if out.exists() and not overwrite:
            counts["skipped_existing"] += 1
            continue
        if seq.video is not None:
            if estimator_factory is None:
                raise ValueError("video dataset: an estimator factory is required")
            est = estimator_factory()
            try:
                extract_video(
                    seq.video, est, out, meta=_meta(seq, pose_backend="mediapipe", model=model)
                )
            finally:
                est.close()
        elif seq.skeleton is not None and seq.joint_format in JOINT_MAPS:
            frames = skeleton_to_frames(
                seq.skeleton, JOINT_MAPS[seq.joint_format], seq.fps, seq.view or "frontal"
            )
            rows: list[tuple[int, int, int, int, Frame | None]] = [
                (f.frame_idx, f.ts_ms, f.img_w, f.img_h, f) for f in frames
            ]
            out.parent.mkdir(parents=True, exist_ok=True)
            meta = _meta(
                seq,
                pose_backend=seq.joint_format,
                model="joint_map",
                virtual_view=seq.view or "frontal",
            )
            pq.write_table(frames_to_table(rows, meta), out)
        else:
            log.warning(
                "%s: no extractor for joint format %s; skipped", seq.source, seq.joint_format
            )
            counts["unsupported"] += 1
            continue
        counts["written"] += 1
    return dict(counts)
