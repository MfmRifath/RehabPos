"""Synthetic landmark JSONL in WebSocket protocol v1 (as the browser would send it)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rehabpose.core.landmarks import Frame


def landmark_messages(frames: Sequence[Frame | None], sid: str) -> list[dict[str, Any]]:
    """One `landmarks` message per detected frame (frames without a person are not sent)."""
    return [
        {
            "v": 1,
            "type": "landmarks",
            "sid": sid,
            "frame": f.frame_idx,
            "ts_ms": f.ts_ms,
            "img": {"w": f.img_w, "h": f.img_h},
            "lm": [[round(float(v), 6) for v in row] for row in f.lm],
            "world": [[round(float(v), 6) for v in row] for row in f.world],
            "client_infer_ms": 12.0,
        }
        for f in frames
        if f is not None
    ]


def write_jsonl(path: Path, messages: Sequence[dict[str, Any]]) -> Path:
    """Write one JSON message per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(m) + "\n" for m in messages))
    return path


def read_jsonl(path: Path, sid: str | None = None) -> list[dict[str, Any]]:
    """Read messages, optionally re-targeting them to session `sid`."""
    msgs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if sid is not None:
        for m in msgs:
            m["sid"] = sid
    return msgs
