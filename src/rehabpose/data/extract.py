"""Extract pose landmarks from a video into a parquet file, and read them back as Frames."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from rehabpose.core.landmarks import NUM_LANDMARKS, Frame
from rehabpose.data.video_io import iter_frames, probe
from rehabpose.pose.base import PoseEstimator

log = logging.getLogger(__name__)

SCHEMA = pa.schema(
    [
        ("frame", pa.int32()),
        ("ts_ms", pa.int64()),
        ("img_w", pa.int32()),
        ("img_h", pa.int32()),
        ("detected", pa.bool_()),
        # Variable-size lists: parquet cannot round-trip nulls in fixed-size lists.
        # Rows hold 33 x 4 values (row-major), validated on load.
        ("lm", pa.list_(pa.float32())),
        ("world", pa.list_(pa.float32())),
    ]
)
META_KEY = b"rehabpose"


@dataclass
class ExtractResult:
    """Summary of an extraction run (counts, wall-clock seconds)."""

    out_path: Path
    n_frames: int
    n_detected: int
    elapsed_s: float
    meta: dict[str, Any] = field(default_factory=dict)


def frames_to_table(
    rows: list[tuple[int, int, int, int, Frame | None]], meta: dict[str, Any]
) -> pa.Table:
    """Build a parquet table from (frame_idx, ts_ms, w, h, Frame|None) rows."""
    cols: dict[str, list[Any]] = {name: [] for name in SCHEMA.names}
    for idx, ts, w, h, fr in rows:
        cols["frame"].append(idx)
        cols["ts_ms"].append(ts)
        cols["img_w"].append(w)
        cols["img_h"].append(h)
        cols["detected"].append(fr is not None)
        cols["lm"].append(None if fr is None else fr.lm.ravel().tolist())
        cols["world"].append(None if fr is None else fr.world.ravel().tolist())
    table = pa.table(cols, schema=SCHEMA)
    return table.replace_schema_metadata({META_KEY: json.dumps(meta).encode()})


def extract_video(
    video: Path,
    estimator: PoseEstimator,
    out_path: Path,
    meta: dict[str, Any] | None = None,
    progress: bool = False,
) -> ExtractResult:
    """Run `estimator` over every frame of `video` and write landmarks to `out_path`."""
    info = probe(video)
    meta = {
        "source": str(video),
        "fps": info.fps,
        "width": info.width,
        "height": info.height,
        **(meta or {}),
    }
    rows: list[tuple[int, int, int, int, Frame | None]] = []
    t0 = time.perf_counter()
    frames = iter_frames(video)
    for idx, ts, img in tqdm(frames, total=info.frame_count or None, disable=not progress):
        fr = estimator.process(img, ts)
        if fr is not None:
            fr = Frame(fr.lm, fr.world, ts, fr.img_w, fr.img_h, idx)
        rows.append((idx, ts, img.shape[1], img.shape[0], fr))
    elapsed = time.perf_counter() - t0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(frames_to_table(rows, meta), out_path)
    n_det = sum(r[4] is not None for r in rows)
    log.info(
        "Extracted %d frames (%d with a person) in %.2f s -> %s",
        len(rows),
        n_det,
        elapsed,
        out_path,
    )
    return ExtractResult(out_path, len(rows), n_det, elapsed, meta)


def read_meta(path: Path) -> dict[str, Any]:
    """Read the RehabPose metadata dict stored in a landmark parquet file."""
    raw = (pq.read_schema(path).metadata or {}).get(META_KEY, b"{}")
    data: dict[str, Any] = json.loads(raw)
    return data


def load_frames(path: Path) -> list[Frame | None]:
    """Load a landmark parquet file as a list of Frames (None where no person was detected)."""
    table = pq.read_table(path)
    if not SCHEMA.equals(table.schema, check_metadata=False):
        raise ValueError(f"{path} is not a RehabPose landmark file (schema mismatch)")
    d = table.to_pydict()
    out: list[Frame | None] = []
    for i in range(table.num_rows):
        if not d["detected"][i]:
            out.append(None)
            continue
        out.append(
            Frame(
                lm=np.asarray(d["lm"][i], dtype=np.float64).reshape(NUM_LANDMARKS, 4),
                world=np.asarray(d["world"][i], dtype=np.float64).reshape(NUM_LANDMARKS, 4),
                ts_ms=int(d["ts_ms"][i]),
                img_w=int(d["img_w"][i]),
                img_h=int(d["img_h"][i]),
                frame_idx=int(d["frame"][i]),
            )
        )
    return out
