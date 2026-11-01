"""Temporal models: shapes, causality, overfit-one-batch; LightGBM + OpenMP regression."""

import subprocess
import sys

import numpy as np
import pytest
import torch

from rehabpose.ml.models import CAUSAL_MODELS, TORCH_MODELS, build_model, n_parameters
from rehabpose.ml.models.lgbm import make_lgbm, summary_features

F2D, F3D = 45, 58


@pytest.mark.parametrize("name", TORCH_MODELS)
@pytest.mark.parametrize("n_features", [F2D, F3D])
def test_forward_shapes(name: str, n_features: int) -> None:
    torch.manual_seed(0)
    m = build_model(name, n_features, 4).eval()
    out = m(torch.randn(3, 45, n_features))
    assert out.shape == (3, 4)
    assert 1_000 < n_parameters(m) < 1_000_000  # small models (target < 1 M params)


@pytest.mark.parametrize("name", CAUSAL_MODELS)
def test_causal_models_ignore_the_future(name: str) -> None:
    """Per-step output at t depends only on frames <= t: step 29 of a 40-frame input equals the
    prediction from the 30-frame prefix and is unchanged when frames 30..39 change; changing
    frame 25 does change it (so the check is not vacuous)."""
    torch.manual_seed(0)
    m = build_model(name, F2D, 3).eval()
    x = torch.randn(2, 40, F2D)
    future = x.clone()
    future[:, 30:] = torch.randn(2, 10, F2D)
    past = x.clone()
    past[:, 25] += 3.0
    with torch.no_grad():
        steps = m.forward_steps(x)  # type: ignore[operator]
        assert steps.shape == (2, 40, 3)
        assert torch.allclose(steps[:, 29], m(x[:, :30]), atol=1e-5)
        assert torch.allclose(steps[:, 29], m.forward_steps(future)[:, 29], atol=1e-5)  # type: ignore[operator]
        assert not torch.allclose(steps[:, 29], m.forward_steps(past)[:, 29], atol=1e-5)  # type: ignore[operator]
        assert torch.allclose(steps[:, -1], m(x), atol=1e-5)


def test_bilstm_is_not_causal() -> None:
    torch.manual_seed(0)
    m = build_model("bilstm", F2D, 3).eval()
    x = torch.randn(1, 20, F2D)
    y = x.clone()
    y[:, 0] += 5.0
    with torch.no_grad():
        assert not torch.allclose(m(x), m(y))
        with pytest.raises(RuntimeError, match="no causal"):
            m.forward_steps(x)  # type: ignore[operator]


def test_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model("resnet", F2D, 2)


@pytest.mark.parametrize("name", TORCH_MODELS)
def test_overfit_one_batch(name: str) -> None:
    """Acceptance: each model can memorise one small batch (loss -> ~0, 100 % accuracy)."""
    torch.manual_seed(0)
    x = torch.randn(16, 32, F2D)
    y = torch.arange(16) % 4
    m = build_model(name, F2D, 4)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    m.train()
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        acc = (m(x).argmax(1) == y).float().mean().item()
        final = torch.nn.functional.cross_entropy(m(x), y).item()
    print(f"{name}: final loss {final:.4f}, acc {acc:.2f}")
    assert acc == 1.0 and final < 0.05


def test_lgbm_summary_and_fit() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 64, F2D))
    y = np.arange(60) % 3
    x[y == 1, :, 0] += 2.0  # separable by the mean of feature 0
    s = summary_features(x)
    assert s.shape == (60, 5 * F2D)
    clf = make_lgbm(seed=0, n_classes=3).fit(s, y)
    assert (clf.predict(s) == y).mean() > 0.9
    assert clf.get_params()["n_jobs"] == 1


def test_torch_then_lightgbm_does_not_segfault() -> None:
    """Regression: torch's and Homebrew's libomp in one process crash multi-threaded LightGBM."""
    code = (
        "import torch, numpy as n\n"
        "from rehabpose.ml.models.lgbm import make_lgbm\n"
        "make_lgbm(0, 2).fit(n.random.rand(80, 5), n.arange(80) % 2)\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "ok" in r.stdout, f"exit {r.returncode}: {r.stderr[-500:]}"


@pytest.mark.parametrize("name", ["tcn", "transformer", "stgcn"])
def test_last_step_sees_most_of_a_rep_window(name: str) -> None:
    """Regression: ST-GCN once saw only 9 of 64 frames (never the rep's peak)."""
    torch.manual_seed(0)
    m = build_model(name, F2D, 4).eval()
    x = torch.randn(1, 64, F2D)
    changed = x.clone()
    changed[:, 5] += 3.0  # early in the window
    with torch.no_grad():
        assert not torch.allclose(m(x), m(changed), atol=1e-6)


@pytest.mark.parametrize("name", TORCH_MODELS)
def test_every_model_exports_with_parity(name: str, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """TorchScript and ONNX match eager PyTorch; the transformer fast-path flag is restored."""
    from rehabpose.ml.export import export_onnx, export_torchscript, parity

    torch.manual_seed(0)
    m = build_model(name, F2D, 4).eval()
    x = torch.randn(1, 45, F2D)
    ts = export_torchscript(m, x, tmp_path / "m.ts")
    onnx = export_onnx(m, x, tmp_path / "m.onnx")
    assert torch.backends.mha.get_fastpath_enabled()
    diffs = parity(m, ts, onnx, x)
    assert diffs["torchscript_max_abs_diff"] < 1e-4 and diffs["onnx_max_abs_diff"] < 1e-4
