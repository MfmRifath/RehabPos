"""Rule engine unit tests: ops, persistence, unknown handling, calibration, rep rules, labels."""

from pathlib import Path

import pytest

from rehabpose.core.exercise import ExerciseConfig, load_exercises
from rehabpose.core.rules import Calibrator, RuleEngine, label_rep

EX = load_exercises(Path(__file__).resolve().parents[2] / "configs" / "exercises")


def _eval(
    ex: ExerciseConfig, view: str, measures: dict, baseline: dict | None = None, t: float = 0.0
):  # type: ignore[no-untyped-def]
    eng = RuleEngine(ex, view, include_experimental=True)  # type: ignore[arg-type]
    flags, status = eng.evaluate(t, measures, {k: 0.9 for k in measures}, baseline or {})
    return {f.code for f in flags}, status


FLEX = EX["shoulder_flexion"]
ABD = EX["shoulder_abduction"]
ELB = EX["elbow_flexion"]
ER = EX["shoulder_external_rotation"]
REACH = EX["forward_reach"]


# (exercise, view, measures, baseline, code, expected)
CASES = [
    # lt: ELBOW_BEND elbow_interior < 160 (min_duration 0.3 s tested separately)
    (ABD, "frontal", {"elbow_interior": 175.0}, {}, "ELBOW_BEND", False),
    # gt: SHOULDER_DRIFT upper_arm_trunk_angle > 20
    (ELB, "sagittal", {"upper_arm_trunk_angle": 25.0}, {}, "SHOULDER_DRIFT", True),
    (ELB, "sagittal", {"upper_arm_trunk_angle": 15.0}, {}, "SHOULDER_DRIFT", False),
    # abs_delta_gt: TRUNK_LEAN |tilt - baseline| > 10 (flexion), > 8 (abduction), > 15 (reach)
    (FLEX, "sagittal", {"trunk_tilt": 14.0}, {"trunk_tilt": 2.0}, "TRUNK_LEAN", True),
    (FLEX, "sagittal", {"trunk_tilt": 11.0}, {"trunk_tilt": 2.0}, "TRUNK_LEAN", False),
    (FLEX, "sagittal", {"trunk_tilt": -9.0}, {"trunk_tilt": 2.0}, "TRUNK_LEAN", True),
    (ABD, "frontal", {"trunk_tilt": 9.5}, {"trunk_tilt": 1.0}, "TRUNK_LEAN", True),
    (ABD, "frontal", {"trunk_tilt": 8.5}, {"trunk_tilt": 1.0}, "TRUNK_LEAN", False),
    (REACH, "sagittal", {"trunk_tilt": 16.5}, {"trunk_tilt": 1.0}, "TRUNK_LEAN", True),
    (REACH, "sagittal", {"trunk_tilt": 14.0}, {"trunk_tilt": 1.0}, "TRUNK_LEAN", False),
    (ELB, "sagittal", {"trunk_tilt": 12.0}, {"trunk_tilt": 0.0}, "TRUNK_LEAN", True),
    # drop_frac_gt: SHOULDER_HIKE hike_ratio drop > 15 % vs baseline
    (ABD, "frontal", {"hike_ratio": 0.40}, {"hike_ratio": 0.50}, "SHOULDER_HIKE", True),
    (ABD, "frontal", {"hike_ratio": 0.45}, {"hike_ratio": 0.50}, "SHOULDER_HIKE", False),
    (ABD, "frontal", {"hike_ratio": 0.60}, {"hike_ratio": 0.50}, "SHOULDER_HIKE", False),
    # TRUNK_ROTATION shoulder width shrink > 15 %
    (ER, "frontal", {"shoulder_width": 80.0}, {"shoulder_width": 100.0}, "TRUNK_ROTATION", True),
    (ER, "frontal", {"shoulder_width": 90.0}, {"shoulder_width": 100.0}, "TRUNK_ROTATION", False),
    # ELBOW_AWAY |E_x - H_x| / shoulder width > 0.25
    (ER, "frontal", {"elbow_away": 0.3}, {}, "ELBOW_AWAY", True),
    (ER, "frontal", {"elbow_away": 0.2}, {}, "ELBOW_AWAY", False),
    # PLANE_DRIFT (experimental) world |W_z - S_z| > 0.15 m
    (ABD, "frontal", {"plane_drift_m": 0.2}, {}, "PLANE_DRIFT", True),
    (ABD, "frontal", {"plane_drift_m": 0.1}, {}, "PLANE_DRIFT", False),
]


@pytest.mark.parametrize(("ex", "view", "measures", "baseline", "code", "expected"), CASES)
def test_frame_rule(ex, view, measures, baseline, code, expected) -> None:  # type: ignore[no-untyped-def]
    raised, status = _eval(ex, view, measures, baseline)
    assert (code in raised) is expected
    assert status[code] is expected


def test_min_duration_persistence() -> None:
    eng = RuleEngine(FLEX, "sagittal")
    bent = {"elbow_interior": 150.0}
    seen = []
    for i in range(12):  # 30 fps
        flags, _ = eng.evaluate(i / 30, bent, {}, {})
        seen.append("ELBOW_BEND" in {f.code for f in flags})
    assert seen.index(True) == 9  # 0.3 s after onset
    flags, _ = eng.evaluate(0.5, {"elbow_interior": 175.0}, {}, {})
    assert not flags
    flags, _ = eng.evaluate(0.55, bent, {}, {})
    assert not flags  # timer restarted


@pytest.mark.parametrize(
    ("measures", "baseline", "view"),
    [
        ({"trunk_tilt": None}, {"trunk_tilt": 0.0}, "sagittal"),  # invisible landmarks
        ({"trunk_tilt": 30.0}, {}, "sagittal"),  # not calibrated yet
        ({}, {"trunk_tilt": 0.0}, "sagittal"),  # measure missing
    ],
)
def test_unknown_inputs_never_raise(measures: dict, baseline: dict, view: str) -> None:  # type: ignore[type-arg]
    raised, status = _eval(FLEX, view, measures, baseline)
    assert "TRUNK_LEAN" not in raised and status["TRUNK_LEAN"] is None


def test_shoulder_hike_unknown_in_sagittal_view() -> None:
    raised, status = _eval(FLEX, "sagittal", {"hike_ratio": 0.1}, {"hike_ratio": 0.5})
    assert "SHOULDER_HIKE" not in raised and status["SHOULDER_HIKE"] is None


def test_confidence_and_severity_from_config() -> None:
    eng = RuleEngine(ELB, "sagittal")
    flags, _ = eng.evaluate(
        0.0, {"upper_arm_trunk_angle": 40.0}, {"upper_arm_trunk_angle": 0.7}, {}
    )
    assert flags[0].confidence == 0.7 and flags[0].severity == "warn" and flags[0].source == "rule"


@pytest.mark.parametrize(
    ("peak", "duration", "vel", "expected"),
    [
        (150, 5.0, 60, set()),
        (130, 5.0, 60, {"INCOMPLETE_ROM"}),  # < 0.9 * 150 = 135
        (136, 5.0, 60, set()),
        (150, 1.9, 60, {"TOO_FAST"}),  # < 2.0 s
        (150, 5.0, 95, {"TOO_FAST"}),  # > 90 deg/s
        (100, 1.0, 200, {"INCOMPLETE_ROM", "TOO_FAST"}),
    ],
)
def test_rep_rules(peak: float, duration: float, vel: float, expected: set[str]) -> None:
    flags = RuleEngine(FLEX, "sagittal").evaluate_rep(peak, duration, vel)
    assert {f.code for f in flags} == expected
    sev = {f.code: f.severity for f in flags}
    assert sev.get("INCOMPLETE_ROM", "info") == "info"


def test_elbow_flexion_too_fast_threshold() -> None:
    eng = RuleEngine(ELB, "sagittal")
    assert not eng.evaluate_rep(130, 1.6, 50)
    assert eng.evaluate_rep(130, 1.4, 50)[0].code == "TOO_FAST"


def test_calibrator_median_and_timeout() -> None:
    cal = Calibrator(duration_s=1.0, max_duration_s=3.0, min_samples=5)
    for i in range(31):
        cal.update(i / 30, {"trunk_tilt": 2.0 if i != 3 else 50.0, "hike_ratio": None})
    assert cal.done and cal.baseline == {"trunk_tilt": 2.0}  # median ignores the outlier
    cal.update(2.0, {"trunk_tilt": 99.0})
    assert cal.baseline["trunk_tilt"] == 2.0  # frozen after done

    cal.reset()
    for i in range(100):
        cal.update(i / 30, {"trunk_tilt": None})
    assert cal.done and cal.timed_out


@pytest.mark.parametrize(
    ("codes", "label"),
    [
        (set(), "correct"),
        ({"INCOMPLETE_ROM"}, "incomplete_rom"),
        ({"INCOMPLETE_ROM", "TOO_FAST"}, "too_fast"),
        ({"TOO_FAST", "ELBOW_BEND"}, "compensated"),
        ({"TRUNK_ROTATION"}, "compensated"),
        ({"SHOULDER_DRIFT"}, "compensated"),
        ({"PLANE_DRIFT"}, "compensated"),
    ],
)
def test_label_precedence(codes: set[str], label: str) -> None:
    assert label_rep(codes) == label
