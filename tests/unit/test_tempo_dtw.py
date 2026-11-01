import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rehabpose.core.dtw import dtw_distance, dtw_similarity, resample, znorm
from rehabpose.core.exercise import min_jerk_rep
from rehabpose.core.tempo import VelocityEstimator, angular_velocity, rep_tempo

# --- tempo -------------------------------------------------------------------------------------


def test_angular_velocity_of_ramp() -> None:
    t = np.arange(30) / 30.0
    v = angular_velocity(45.0 * t, t)
    assert v[0] == 0.0
    np.testing.assert_allclose(v[1:], 45.0)
    np.testing.assert_allclose(angular_velocity(45.0 * t, t, lag=3)[3:], 45.0)
    with pytest.raises(ValueError):
        angular_velocity(t, t, lag=0)


def test_angular_velocity_is_causal() -> None:
    t = np.arange(20) / 30.0
    x = np.zeros(20)
    x[10] = 5.0
    v = angular_velocity(x, t)
    assert np.all(v[:10] == 0.0)  # no response before the change


def test_velocity_estimator_window() -> None:
    est = VelocityEstimator(window_s=0.1)
    out = [est.update(i / 30.0, 30.0 * i / 30.0) for i in range(10)]
    assert out[0] == 0.0
    np.testing.assert_allclose(out[1:], 30.0)
    est.reset()
    assert est.update(5.0, 1.0) == 0.0


def test_rep_tempo() -> None:
    rt = rep_tempo(np.array([1.0, 1.5, 3.0]), np.array([10.0, -60.0, 20.0]))
    assert (rt.duration_s, rt.peak_vel, rt.mean_abs_vel) == (2.0, 60.0, 30.0)
    assert rep_tempo(np.array([]), np.array([])).duration_s == 0.0


# --- dtw ---------------------------------------------------------------------------------------


def test_znorm_and_resample() -> None:
    z = znorm(np.array([1.0, 2.0, 3.0]))
    assert z.mean() == pytest.approx(0) and z.std() == pytest.approx(1)
    assert np.all(znorm(np.ones(5)) == 0)
    np.testing.assert_allclose(resample(np.array([0.0, 10.0]), 3), [0, 5, 10])
    assert resample(np.array([7.0]), 4).tolist() == [7.0] * 4
    with pytest.raises(ValueError):
        resample(np.array([]), 3)


def test_dtw_known_values() -> None:
    assert dtw_distance(np.array([1.0, 2, 3]), np.array([1.0, 2, 3])) == (0.0, 3)
    # b repeats a's middle sample: zero-cost warp, path length 4
    assert dtw_distance(np.array([1.0, 2, 3]), np.array([1.0, 2, 2, 3]), band=0.0) == (0.0, 4)
    assert dtw_distance(np.array([0.0]), np.array([2.0])) == (2.0, 1)
    with pytest.raises(ValueError):
        dtw_distance(np.array([]), np.array([1.0]))


def test_sakoe_chiba_band_constrains_warping() -> None:
    a = np.r_[np.zeros(10), 1.0, np.zeros(9)]
    b = np.r_[np.zeros(15), 1.0, np.zeros(4)]
    free, _ = dtw_distance(a, b, band=1.0)
    tight, _ = dtw_distance(a, b, band=0.1)
    assert free == 0.0 and tight > 0.0


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.floats(-100, 100), min_size=2, max_size=30),
    st.lists(st.floats(-100, 100), min_size=2, max_size=30),
)
def test_dtw_symmetry(a: list[float], b: list[float]) -> None:
    d1, _ = dtw_distance(np.array(a), np.array(b), band=1.0)
    d2, _ = dtw_distance(np.array(b), np.array(a), band=1.0)
    assert d1 == pytest.approx(d2)


def test_similarity_properties() -> None:
    ref = np.array(min_jerk_rep(150.0, 64))
    assert dtw_similarity(ref, ref) == pytest.approx(1.0)
    # amplitude/offset and tempo invariance (z-norm + resampling)
    assert dtw_similarity(0.5 * ref + 20, ref) == pytest.approx(1.0)
    slow = np.array(min_jerk_rep(150.0, 200))
    assert dtw_similarity(slow, ref) > 0.97
    # a time-warped rep (fast up, slow down) beats an unrelated shape
    warped = np.r_[np.array(min_jerk_rep(150.0, 40))[:20], np.array(min_jerk_rep(150.0, 120))[60:]]
    noise = np.random.default_rng(0).normal(0, 1, 64)
    assert dtw_similarity(warped, ref) > dtw_similarity(noise, ref)
    assert 0.0 < dtw_similarity(noise, ref) < 1.0
