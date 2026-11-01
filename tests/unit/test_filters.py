import numpy as np
import pytest

from rehabpose.core.filters import FrameSmoother, OneEuroFilter, OneEuroParams, savgol_smooth
from tests.fixtures.synthetic import SynthPose, make_frame

FPS = 30.0


def _run(f: OneEuroFilter, xs: np.ndarray, fps: float = FPS) -> np.ndarray:
    return np.array([f(np.atleast_1d(x), i / fps) for i, x in enumerate(xs)]).squeeze()


@pytest.mark.parametrize("sigma", [0.005, 0.02, 0.05])
def test_static_noise_std_reduced_by_half(sigma: float) -> None:
    """Acceptance: >= 50 % std reduction on a noisy static signal (shoulder-width units)."""
    x = np.random.default_rng(0).normal(0.0, sigma, 600)
    y = _run(OneEuroFilter(), x)
    ratio = y[60:].std() / x[60:].std()
    print(f"sigma={sigma}: output/input std = {ratio:.3f}")
    assert ratio <= 0.5


@pytest.mark.parametrize("step", [0.2, 0.5, 1.0])
def test_step_lag_under_150ms(step: float) -> None:
    """Acceptance: step lag < 150 ms at 30 fps; lag = time to reach 63 % (one time constant)."""
    x = np.r_[np.zeros(30), np.full(60, step)]
    y = _run(OneEuroFilter(), x)
    lag_ms = np.argmax(y[30:] >= 0.63 * step) / FPS * 1000
    print(f"step={step}: lag to 63% = {lag_ms:.0f} ms")
    assert lag_ms < 150
    assert y[-1] == pytest.approx(step, rel=1e-3)


def test_vectorized_shape_and_independence() -> None:
    f = OneEuroFilter()
    out = f(np.zeros((33, 3)), 0.0)
    assert out.shape == (33, 3)
    x = np.zeros((33, 3))
    x[5, 1] = 1.0
    out = f(x, 1 / FPS)
    assert out[5, 1] > 0 and np.count_nonzero(out) == 1


def test_reset_on_gap() -> None:
    f = OneEuroFilter(OneEuroParams(reset_gap_s=0.5))
    f(np.array([0.0]), 0.0)
    f(np.array([0.0]), 1 / FPS)
    assert f(np.array([1.0]), 0.1)[0] < 1.0  # smoothed
    assert f(np.array([5.0]), 0.7)[0] == 5.0  # gap 0.6 s > 0.5 s -> passes through


def test_nan_passthrough_keeps_state() -> None:
    f = OneEuroFilter()
    f(np.array([1.0, 1.0]), 0.0)
    out = f(np.array([np.nan, 1.0]), 1 / FPS)
    assert np.isnan(out[0]) and out[1] == pytest.approx(1.0)
    out = f(np.array([1.0, 1.0]), 2 / FPS)
    assert out[0] == pytest.approx(1.0)


def test_nonincreasing_time_returns_previous() -> None:
    f = OneEuroFilter()
    f(np.array([1.0]), 0.0)
    assert f(np.array([9.0]), 0.0)[0] == 1.0


def test_scale_normalizes_speed() -> None:
    """Same motion in pixels vs normalized units filters identically when scale matches."""
    x = np.r_[np.zeros(10), np.linspace(0, 1, 20), np.ones(10)]
    y_unit = _run(OneEuroFilter(), x)
    f = OneEuroFilter()
    y_px = np.array([f(np.array([v * 120.0]), i / FPS, scale=120.0)[0] for i, v in enumerate(x)])
    np.testing.assert_allclose(y_px / 120.0, y_unit, atol=1e-9)


def test_frame_smoother_reduces_landmark_jitter() -> None:
    rng = np.random.default_rng(3)
    sm = FrameSmoother()
    raw, smooth = [], []
    for i in range(120):
        fr = make_frame(
            SynthPose(elevation_deg=45),
            "sagittal",
            noise_px=2.0,
            ts_ms=round(i * 1000 / FPS),
            rng=rng,
        )
        out = sm(fr)
        raw.append(fr.pixel_xy[14])
        smooth.append(out.pixel_xy[14])
        assert out.visibility[14] == fr.visibility[14]
    assert np.std(smooth[30:], axis=0).mean() < 0.5 * np.std(raw[30:], axis=0).mean()


def test_savgol_smooth() -> None:
    t = np.arange(90) / FPS
    trend = 30 * t
    noisy = trend + np.random.default_rng(0).normal(0, 1.0, t.size)
    out = savgol_smooth(noisy, FPS, window_s=0.3)
    assert out.shape == noisy.shape
    assert np.std(out - trend) < 0.6 * np.std(noisy - trend)
    np.testing.assert_allclose(savgol_smooth(trend, FPS), trend, atol=1e-9)  # preserves lines
    assert savgol_smooth(np.array([1.0, 2.0]), FPS).shape == (2,)  # too short: returned as-is
