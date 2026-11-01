from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rehabpose.core.exercise import (
    ExerciseConfig,
    FrameRule,
    RepRule,
    load_exercise,
    load_exercises,
    min_jerk_rep,
)

EX_DIR = Path(__file__).resolve().parents[2] / "configs" / "exercises"
IDS = {
    "shoulder_flexion",
    "shoulder_abduction",
    "elbow_flexion",
    "shoulder_external_rotation",
    "forward_reach",
}


def _base() -> dict[str, Any]:
    return {
        "id": "demo",
        "name": "Demo",
        "view": "sagittal",
        "primary_angle": "upper_arm_trunk_angle",
        "target": 150,
        "fsm": {"start_below": 30, "peak_above_offset": 15, "return_below": 40},
        "min_rep_s": 2.0,
        "max_peak_vel_dps": 90,
        "rules": [
            {
                "kind": "frame",
                "code": "TRUNK_LEAN",
                "measure": "trunk_tilt",
                "op": "abs_delta_gt",
                "value": 10,
            },
            {"kind": "rep", "code": "TOO_FAST"},
        ],
        "cue_texts": {"TRUNK_LEAN": "Upright", "TOO_FAST": "Slow"},
        "reference": {"source": "synthetic"},
    }


def test_all_shipped_configs_load() -> None:
    ex = load_exercises(EX_DIR)
    assert set(ex) == IDS
    flex = ex["shoulder_flexion"]
    assert (flex.target, flex.view, flex.peak_threshold) == (150, "sagittal", 135)
    assert {r.code for r in flex.rules} == {
        "ELBOW_BEND", "TRUNK_LEAN", "SHOULDER_HIKE", "INCOMPLETE_ROM", "TOO_FAST",
    }  # fmt: skip
    hike = next(r for r in flex.frame_rules if r.code == "SHOULDER_HIKE")
    assert hike.unknown_in_views == ("sagittal",) and hike.needs_baseline
    assert ex["elbow_flexion"].views == ("sagittal", "frontal")
    assert ex["shoulder_external_rotation"].experimental
    for e in ex.values():
        assert e.reference.source == "synthetic"  # until recorded references exist
        assert len(e.reference_curve()) == 64


def test_spec_thresholds_match_docs() -> None:
    ex = load_exercises(EX_DIR)
    abd = {r.code: r for r in ex["shoulder_abduction"].frame_rules}
    assert abd["TRUNK_LEAN"].value == 8 and abd["SHOULDER_HIKE"].value == 0.15
    assert abd["PLANE_DRIFT"].experimental and abd["PLANE_DRIFT"].value == 0.15
    elbow = ex["elbow_flexion"]
    assert elbow.min_rep_s == 1.5 and elbow.target == 130
    er = {r.code: r for r in ex["shoulder_external_rotation"].frame_rules}
    assert er["ELBOW_AWAY"].value == 0.25 and er["TRUNK_ROTATION"].value == 0.15


def test_min_jerk_rep_shape() -> None:
    c = min_jerk_rep(100.0, 65)
    assert c[0] == 0 and c[-1] == 0 and max(c) == pytest.approx(100.0)
    assert c.index(max(c)) == 32


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"primary_angle": "nose_angle"}, "primary_angle"),
        ({"cue_texts": {"TRUNK_LEAN": "x"}}, "cue_texts missing"),
        ({"fsm": {"start_below": 50, "peak_above_offset": 15, "return_below": 40}}, "return_below"),
        ({"target": 40}, "target must be"),
        ({"view": "frontal", "allowed_views": ["sagittal"]}, "allowed_views"),
        ({"reference": {"source": "recorded"}}, "recorded reference"),
        ({"unknown_key": 1}, "Extra inputs"),
    ],
)
def test_validation_errors(patch: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ExerciseConfig.model_validate({**_base(), **patch})


def test_rule_validation() -> None:
    with pytest.raises(ValidationError, match="unknown measure"):
        FrameRule(code="X_Y", measure="knee", op="lt", value=1)
    with pytest.raises(ValidationError):
        FrameRule(code="lower", measure="trunk_tilt", op="lt", value=1)
    with pytest.raises(ValidationError):
        RepRule(kind="rep", code="ELBOW_BEND")  # type: ignore[arg-type]


def test_recorded_reference_overrides_synthetic() -> None:
    ex = ExerciseConfig.model_validate(
        {**_base(), "reference": {"source": "recorded", "curve": [0, 5, 0]}}
    )
    assert ex.reference_curve() == [0, 5, 0]


def test_id_must_match_filename(tmp_path: Path) -> None:
    import yaml

    (tmp_path / "other.yaml").write_text(yaml.safe_dump(_base()))
    assert load_exercise(tmp_path / "other.yaml").id == "demo"
    with pytest.raises(ValueError, match="must match"):
        load_exercises(tmp_path)


def test_apply_overrides() -> None:
    from rehabpose.core.exercise import apply_overrides

    flex = load_exercises(EX_DIR)["shoulder_flexion"]
    ex = apply_overrides(flex, target=120, min_rep_s=3.0, thresholds={"TRUNK_LEAN": 6})
    assert (ex.target, ex.min_rep_s, ex.peak_threshold) == (120, 3.0, 105)
    assert next(r for r in ex.frame_rules if r.code == "TRUNK_LEAN").value == 6
    assert flex.target == 150  # original untouched
    with pytest.raises(ValueError, match="no frame rule"):
        apply_overrides(flex, thresholds={"TOO_FAST": 1})
    with pytest.raises(ValidationError):
        apply_overrides(flex, target=40)  # below start_below + min_excursion
