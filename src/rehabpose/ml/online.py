"""Streaming inference: an exported stream-task bundle as a SessionEngine FlagModel."""

from __future__ import annotations

import json
from collections import deque
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from rehabpose.core.features import StreamingFeatures
from rehabpose.core.landmarks import Frame, Side
from rehabpose.core.rules import Flag

Backend = Literal["auto", "onnx", "torchscript"]


class TemporalFlagModel:
    """Runs an exported stream model every `stride` frames over the last `window` frames.

    Backend "auto" prefers ONNX Runtime (TorchScript is deprecated in torch >= 2.15, ADR 0004)
    and falls back to TorchScript. Flags (source "model", confidence = probability) stay
    raised between evaluations. Feature velocities use the bundle's fps (training fps), so
    live input should arrive at about that rate.
    """

    def __init__(
        self,
        bundle_dir: Path,
        side: Side,
        threshold: float | None = None,
        backend: Backend = "auto",
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.meta: dict[str, Any] = json.loads((self.bundle_dir / "meta.json").read_text())
        if self.meta["task"] != "stream":
            raise ValueError(
                f"{bundle_dir} is a {self.meta['task']!r} model; online use needs 'stream'"
            )
        onnx_ok = (self.bundle_dir / "model.onnx").is_file() and find_spec(
            "onnxruntime"
        ) is not None
        if backend == "onnx" and not onnx_ok:
            raise RuntimeError("the ONNX backend needs model.onnx and onnxruntime")
        self.backend: Literal["onnx", "torchscript"] = (
            "onnx" if backend == "onnx" or (backend == "auto" and onnx_ok) else "torchscript"
        )
        if self.backend == "onnx":
            import onnxruntime as ort

            self._sess = ort.InferenceSession(
                str(self.bundle_dir / "model.onnx"), providers=["CPUExecutionProvider"]
            )
            self._input = self._sess.get_inputs()[0].name
        else:
            self._ts = torch.jit.load(str(self.bundle_dir / "model.ts"), map_location="cpu").eval()
        self.codes: list[str] = list(self.meta["classes"])
        self.window = int(self.meta["window"])
        self.stride = int(self.meta["stride"])
        self.threshold = float(threshold if threshold is not None else self.meta["threshold"])
        self.features = StreamingFeatures(side, float(self.meta["fps"]), self.meta["space"])
        self._buf: deque[np.ndarray] = deque(maxlen=self.window)
        self._since = 0
        self._flags: list[Flag] = []
        self.last_proba: dict[str, float] = {}

    def reset(self) -> None:
        """Clear the window and raised flags."""
        self.features.reset()
        self._buf.clear()
        self._since = 0
        self._flags = []
        self.last_proba = {}

    def _logits(self, x: np.ndarray) -> np.ndarray:
        if self.backend == "onnx":
            return np.asarray(self._sess.run(None, {self._input: x})[0], dtype=np.float64)
        with torch.no_grad():
            return self._ts(torch.from_numpy(x)).double().numpy()

    def update(self, t_s: float, frame: Frame | None) -> list[Flag]:
        """Add one frame; returns currently raised model flags."""
        self._buf.append(self.features.update(frame))
        self._since += 1
        if len(self._buf) == self.window and self._since >= self.stride:
            self._since = 0
            logits = self._logits(np.asarray(self._buf, dtype=np.float32)[None])[0]
            proba = 1.0 / (1.0 + np.exp(-logits))
            self.last_proba = dict(zip(self.codes, map(float, proba), strict=True))
            self._flags = [
                Flag(code, "warn", round(p, 3), "model")
                for code, p in self.last_proba.items()
                if p >= self.threshold
            ]
        return list(self._flags)
