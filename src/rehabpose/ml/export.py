"""Export torch models to TorchScript and ONNX, with numerical parity checks."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


def export_torchscript(model: torch.nn.Module, example: torch.Tensor, path: Path) -> Path:
    """Trace on CPU and save (fixed input length; batch size may vary)."""
    model = model.cpu().eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, example.cpu())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(path))
    return Path(path)


def export_onnx(model: torch.nn.Module, example: torch.Tensor, path: Path) -> Path:
    """Export with the torch.onnx exporter (dynamo) at a fixed input shape."""
    model = model.cpu().eval()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # The transformer fast-path op has no ONNX translation; use the plain code path for export.
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            prog = torch.onnx.export(model, (example.cpu(),), dynamo=True, verbose=False)
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    if prog is None:
        raise RuntimeError("torch.onnx.export returned no program")
    prog.save(str(path))
    return Path(path)


def parity(
    model: torch.nn.Module, ts_path: Path | None, onnx_path: Path | None, x: torch.Tensor
) -> dict[str, float]:
    """Max |difference| of TorchScript / ONNX Runtime outputs vs eager PyTorch on x."""
    model = model.cpu().eval()
    out: dict[str, float] = {}
    with torch.no_grad():
        ref = model(x.cpu()).numpy()
        if ts_path is not None:
            ts = torch.jit.load(str(ts_path)).eval()
            out["torchscript_max_abs_diff"] = float(np.abs(ts(x.cpu()).numpy() - ref).max())
    if onnx_path is not None:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        got = sess.run(None, {name: x.cpu().numpy().astype(np.float32)})[0]
        out["onnx_max_abs_diff"] = float(np.abs(got - ref).max())
    return out
