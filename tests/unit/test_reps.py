"""Rep FSM on 1D primary curves (exact counts with noise, pauses, partial reps, gaps)."""

import numpy as np
import pytest

from rehabpose.core.exercise import FsmConfig
from rehabpose.core.reps import RepCounter
from tests.fixtures.synthetic import RepSpec, rep_profile

FSM = FsmConfig(start_below=30, peak_above_offset=15, return_below=40)
PEAK_TH = 135.0  # target 150 - 15
FPS = 30.0


def _run(values: np.ndarray, fps: float = FPS, gaps: tuple[tuple[int, int], ...] = ()):  # type: ignore[no-untyped-def]
    rc = RepCounter(FSM, PEAK_TH)
    reps, phases = [], []
    for i, v in enumerate(values):
        unknown = any(a <= i < b for a, b in gaps)
        cand = rc.update(i / fps, None if unknown else float(v))
        phases.append(rc.phase)
        if cand is not None:
            reps.append(cand)
    return reps, phases


def test_single_rep_timing_and_phases() -> None:
    values, _, _ = rep_profile([RepSpec(150, duration_s=6.0)], FPS, rest_s=1.0)
    reps, phases = _run(values)
    assert len(reps) == 1
    r = reps[0]
    assert r.reached_peak and r.peak_value == pytest.approx(150, abs=0.5)
    assert r.t_start < r.t_peak < r.t_end and r.t_peak == pytest.approx(1.0 + 3.0, abs=0.05)
    assert {"rest", "ascending", "peak", "descending"} <= set(phases)
    assert r.values.size == r.times.size == r.velocities.size


def _expected(reps: list[RepSpec]) -> int:
    return sum(r.peak >= FSM.start_below + FSM.min_excursion for r in reps)


@pytest.mark.parametrize("seed", range(25))
def test_exact_count_randomized(seed: int) -> None:
    """5-15 reps; random tempo, holds, pauses; partial reps (counted and ignored); noise."""
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(int(rng.integers(5, 16))):
        kind = rng.choice(["full", "partial", "tiny"], p=[0.7, 0.2, 0.1])
        peak = {
            "full": rng.uniform(140, 160),
            "partial": rng.uniform(60, 130),
            "tiny": rng.uniform(35, 45),
        }[kind]
        reps.append(
            RepSpec(
                peak=float(peak),
                duration_s=float(rng.uniform(2.0, 8.0)),
                hold_s=float(rng.choice([0.0, rng.uniform(0.2, 2.0)])),
                pause_s=float(rng.uniform(0.3, 3.0)),
            )
        )
    values, _, _ = rep_profile(reps, FPS, rest_s=1.0)
    noisy = values + rng.normal(0, 0.7, values.size)  # ~smoothed-angle noise level
    found, _ = _run(noisy)
    assert len(found) == _expected(reps)
    partial = [r for r in reps if FSM.start_below + FSM.min_excursion <= r.peak < PEAK_TH]
    assert sum(not c.reached_peak for c in found) == len(partial)


def test_tiny_and_short_movements_ignored() -> None:
    values, _, _ = rep_profile([RepSpec(45), RepSpec(150, duration_s=0.6)], FPS, rest_s=1.0)
    reps, _ = _run(values)
    assert reps == []  # peak < 50 deg, and a 0.6 s movement < 0.8 s minimum


def test_dip_near_peak_does_not_split_rep() -> None:
    up = np.linspace(0, 150, 60)
    dip = np.r_[np.linspace(150, 120, 15), np.linspace(120, 150, 15)]
    down = np.linspace(150, 0, 60)
    reps, _ = _run(np.r_[np.zeros(30), up, dip, down, np.zeros(30)])
    assert len(reps) == 1


def test_rest_above_start_threshold_reanchors_start() -> None:
    """Resting at 35 deg (between start 30 and return 40) between reps keeps durations honest."""
    rep = np.r_[np.linspace(35, 150, 90), np.linspace(150, 35, 90)]
    values = np.r_[np.zeros(30), rep, np.full(90, 35.0), rep, np.zeros(30)]
    reps, _ = _run(values)
    assert len(reps) == 2
    assert reps[1].duration_s < 6.5  # ~6 s movement, not 9 s including the 3 s rest


def test_unknown_gap_holds_then_resets() -> None:
    values, _, _ = rep_profile([RepSpec(150, duration_s=6.0)] * 2, FPS, rest_s=1.0)
    # short gap (0.3 s) mid-rep 1: rep still counted
    reps, _ = _run(values, gaps=((80, 89),))
    assert len(reps) == 2
    # long gap (1 s) mid-rep 1: that rep is abandoned
    reps, _ = _run(values, gaps=((70, 100),))
    assert len(reps) == 1


def test_reset() -> None:
    rc = RepCounter(FSM, PEAK_TH)
    rc.update(0.0, 0.0)
    rc.update(0.1, 60.0)
    assert rc.phase == "ascending"
    rc.reset()
    assert rc.phase == "rest"


def test_must_start_from_rest() -> None:
    """A set that begins with the arm already raised does not count that first movement."""
    values = np.r_[
        np.linspace(140, 0, 60), np.zeros(15), np.linspace(0, 150, 90), np.linspace(150, 0, 90)
    ]
    reps, _ = _run(values)
    assert len(reps) == 1
