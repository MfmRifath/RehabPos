"""SessionEngine end-to-end on synthetic skeleton sequences."""

import time
from pathlib import Path

import numpy as np
import pytest

from rehabpose.core.engine import EngineConfig, SessionEngine, detect_view
from rehabpose.core.exercise import load_exercises
from rehabpose.core.landmarks import Frame
from tests.fixtures.synthetic import RepSpec, SynthPose, make_frame, rep_sequence

EX = load_exercises(Path(__file__).resolve().parents[2] / "configs" / "exercises")


def run(eid: str, frames: list[Frame | None], **kw) -> tuple[SessionEngine, list]:  # type: ignore[no-untyped-def,type-arg]
    eng = SessionEngine(EX[eid], "R", **kw)
    out = []
    for i, f in enumerate(frames):
        out.append(
            eng.process(f) if f is not None else eng.process_missing(round(i * 1000 / 30), i)
        )
    return eng, out


# --- rep counting -------------------------------------------------------------------------------

# (tiny: ignored, partial: counted as incomplete, full) peak ranges per exercise
PEAKS = {
    "shoulder_flexion": ((36, 44), (60, 120), (147, 153)),
    "shoulder_abduction": ((36, 44), (60, 105), (127, 133)),
    "elbow_flexion": ((36, 44), (60, 105), (127, 133)),
    "shoulder_external_rotation": ((20, 30), (38, 42), (58, 62)),
    "forward_reach": ((0.2, 0.3), (0.45, 0.7), (0.9, 0.95)),
}


def _random_set(eid: str, seed: int) -> tuple[list[RepSpec], int]:
    rng = np.random.default_rng(seed)
    tiny, partial, full = PEAKS[eid]
    reps, expected = [], 0
    for _ in range(int(rng.integers(5, 16))):
        kind = rng.choice(3, p=[0.1, 0.2, 0.7])
        lo, hi = (tiny, partial, full)[kind]
        reps.append(
            RepSpec(
                peak=float(rng.uniform(lo, hi)),
                duration_s=float(rng.uniform(3.0, 9.0)),
                hold_s=float(rng.choice([0.0, 1.0])),
                pause_s=float(rng.uniform(0.5, 3.0)),
            )
        )
        expected += int(kind > 0)
    return reps, expected


@pytest.mark.parametrize("eid", sorted(PEAKS))
def test_rep_count_exact_on_synthetic_sets(eid: str) -> None:
    """Acceptance: rep-count MAE 0 on synthetic sets (5-15 reps, noise, pauses, partial reps)."""
    errors = []
    for seed in range(4):
        reps, expected = _random_set(eid, seed)
        eng, _ = run(eid, rep_sequence(eid, reps, noise_px=1.0, seed=seed))
        errors.append(abs(len(eng.reps) - expected))
    print(f"{eid}: rep-count abs errors {errors} -> MAE {np.mean(errors):.2f}")
    assert np.mean(errors) == 0


def test_rep_record_fields_and_labels() -> None:
    reps = [RepSpec(150), RepSpec(100), RepSpec(150, duration_s=2.0)]
    eng, out = run("shoulder_flexion", rep_sequence("shoulder_flexion", reps))
    assert [r.label for r in eng.reps] == ["correct", "incomplete_rom", "too_fast"]
    r0 = eng.reps[0]
    assert r0.idx == 0 and r0.t_start < r0.t_peak < r0.t_end
    assert r0.peak_angle == pytest.approx(150, abs=2)
    assert r0.duration_s == pytest.approx(r0.t_end - r0.t_start, abs=1e-3)
    assert 0.0 < r0.peak_vel_dps < 90  # 8 s min-jerk rep: ~77 deg/s
    assert 0.8 < r0.dtw_similarity <= 1.0
    completed = [fb.completed_rep for fb in out if fb.completed_rep]
    assert completed == eng.reps
    assert out[-1].rep_count == 3 and out[-1].last_rep == eng.reps[-1]
    assert set(r0.to_dict()) >= {
        "idx",
        "t_start",
        "t_peak",
        "t_end",
        "label",
        "flags",
        "dtw_similarity",
    }


# --- rules end to end: positive (compensated middle rep) and negative (clean reps) ------------

RULE_CASES = [
    ("shoulder_flexion", "ELBOW_BEND", {"elbow_flexion_deg": 40.0}),
    ("shoulder_flexion", "TRUNK_LEAN", {"trunk_lean_deg": 15.0, "trunk_lean_plane": "sagittal"}),
    ("shoulder_abduction", "TRUNK_LEAN", {"trunk_lean_deg": -12.0}),
    ("shoulder_abduction", "SHOULDER_HIKE", {"hike_m": 0.05}),
    ("shoulder_abduction", "ELBOW_BEND", {"elbow_flexion_deg": 40.0, "rotation_deg": 90.0}),
    ("shoulder_abduction", "PLANE_DRIFT", {"plane_deg": 40.0}),
    ("elbow_flexion", "SHOULDER_DRIFT", {"elevation_deg": 35.0}),
    ("elbow_flexion", "TRUNK_LEAN", {"trunk_lean_deg": 15.0, "trunk_lean_plane": "sagittal"}),
    ("shoulder_external_rotation", "ELBOW_AWAY", {"elevation_deg": 30.0, "plane_deg": 90.0}),
    ("shoulder_external_rotation", "TRUNK_ROTATION", {"trunk_rotation_deg": 40.0}),
    ("forward_reach", "TRUNK_LEAN", {"trunk_lean_deg": 20.0, "trunk_lean_plane": "sagittal"}),
]


@pytest.mark.parametrize(
    ("eid", "code", "comp"), RULE_CASES, ids=[f"{e}-{c}" for e, c, _ in RULE_CASES]
)
def test_rule_end_to_end(eid: str, code: str, comp: dict[str, float | str]) -> None:
    target = EX[eid].target
    reps = [RepSpec(target), RepSpec(target, compensation=comp), RepSpec(target)]
    cfg = EngineConfig(experimental_rules=code == "PLANE_DRIFT")
    eng, out = run(eid, rep_sequence(eid, reps, seed=1), config=cfg)
    assert len(eng.reps) == 3
    clean0, bad, clean2 = eng.reps
    assert clean0.flags == () and clean2.flags == ()  # negative
    assert clean0.label == clean2.label == "correct"
    assert bad.flags == (code,) and bad.label == "compensated"  # positive
    shown = {fb.cue.code for fb in out if fb.cue}
    assert code in shown


@pytest.mark.parametrize(
    ("spec", "code"),
    [(RepSpec(100), "INCOMPLETE_ROM"), (RepSpec(150, duration_s=2.0), "TOO_FAST")],
)
def test_rep_rules_end_to_end(spec: RepSpec, code: str) -> None:
    eng, out = run("shoulder_flexion", rep_sequence("shoulder_flexion", [RepSpec(150), spec]))
    assert eng.reps[0].flags == () and eng.reps[1].flags == (code,)
    assert code in {fb.cue.code for fb in out if fb.cue}  # held long enough to pass debounce


def test_events_are_rising_edges() -> None:
    comp = {"trunk_lean_deg": 15.0, "trunk_lean_plane": "sagittal"}
    eng, _ = run(
        "shoulder_flexion", rep_sequence("shoulder_flexion", [RepSpec(150, compensation=comp)])
    )
    codes = [e.code for e in eng.events]
    assert codes.count("TRUNK_LEAN") == 1
    ev = eng.events[codes.index("TRUNK_LEAN")]
    assert ev.source == "rule" and ev.severity == "warn" and 0 < ev.confidence <= 1


# --- setup, view, control -----------------------------------------------------------------------


def test_calibrating_then_calibrated() -> None:
    _, out = run("shoulder_flexion", rep_sequence("shoulder_flexion", [RepSpec(150)], rest_s=2.0))
    assert not out[0].calibrated and out[40].calibrated
    assert out[20].cue is not None and out[20].cue.code == "CALIBRATING"


def test_missing_person_raises_setup_visibility() -> None:
    frames = rep_sequence("shoulder_flexion", [RepSpec(150)], dropouts=[(3.0, 4.5)])
    _, out = run("shoulder_flexion", frames)
    i = int(4.2 * 30)
    assert not out[i].visibility_ok and out[i].angles["upper_arm_trunk_angle"] is None
    assert out[i].cue is not None and out[i].cue.code == "SETUP_VISIBILITY"
    assert out[i].cue.severity == "error"


def test_occluded_wrist_gates_elbow_rules() -> None:
    frames = [
        make_frame(
            SynthPose(elbow_flexion_deg=60), "sagittal", occluded=[16], ts_ms=i * 33, frame_idx=i
        )
        for i in range(60)
    ]
    _, out = run("shoulder_flexion", frames)  # type: ignore[arg-type]
    assert all("ELBOW_BEND" not in {f.code for f in fb.flags} for fb in out)
    assert out[-1].angles["elbow_flexion"] is None


NEAR_R_HIDDEN = [8, 12, 14, 16, 24]
FAR_HIDDEN_FOR_R = [7, 11, 13, 15, 23]


def _side_on_set(hidden: list[int], lean: bool = True) -> list[Frame]:
    from tests.fixtures.synthetic import rep_profile

    reps = [RepSpec(150), RepSpec(150)]
    values, ramps, owner = rep_profile(reps, 30.0, 1.5)
    out = []
    for i, (v, r, k) in enumerate(zip(values, ramps, owner, strict=True)):
        extra = (
            {"trunk_lean_deg": 18.0 * r, "trunk_lean_plane": "sagittal"} if lean and k == 1 else {}
        )
        pose = SynthPose(side="R", elevation_deg=float(v), **extra)  # type: ignore[arg-type]
        ts = round(i * 1000 / 30)
        out.append(make_frame(pose, "sagittal", occluded=hidden, ts_ms=ts, frame_idx=i))
    return out


def test_side_on_trunk_lean_works_with_far_side_hidden() -> None:
    """Regression: hiding the far shoulder/hip made trunk tilt unknown on every frame."""
    eng, out = run("shoulder_flexion", _side_on_set(FAR_HIDDEN_FOR_R))  # type: ignore[arg-type]
    assert [r.flags for r in eng.reps] == [(), ("TRUNK_LEAN",)]
    assert all(fb.angles["trunk_tilt"] is not None for fb in out)
    assert {fb.cue.code for fb in out if fb.cue} >= {"TRUNK_LEAN"}
    lean = next(f for fb in out for f in fb.flags if f.code == "TRUNK_LEAN")
    assert lean.confidence > 0.6  # near-side landmarks only, not the hidden far side


def test_side_on_exercising_arm_hidden_says_which_way_to_turn() -> None:
    eng, out = run("shoulder_flexion", _side_on_set(NEAR_R_HIDDEN, lean=False))  # type: ignore[arg-type]
    assert eng.reps == []
    cues = {fb.cue.text for fb in out if fb.cue and fb.cue.code == "SETUP_VISIBILITY"}
    assert cues == {"Turn so your right arm faces the camera"}


def test_no_person_keeps_the_step_back_cue() -> None:
    frames = rep_sequence("shoulder_flexion", [RepSpec(150)], dropouts=[(3.0, 4.5)])
    _, out = run("shoulder_flexion", frames)
    cue = out[int(4.2 * 30)].cue
    assert cue is not None and cue.text.startswith("Step back")


def test_view_mismatch_raises_setup_view() -> None:
    frames = [make_frame(SynthPose(), "frontal", ts_ms=i * 33, frame_idx=i) for i in range(60)]
    _, out = run("shoulder_flexion", frames)  # type: ignore[arg-type]
    assert not out[-1].view_ok
    assert out[-1].cue is not None and out[-1].cue.code == "SETUP_VIEW_SAGITTAL"


def test_detect_view() -> None:
    cfg = EngineConfig()
    assert detect_view(make_frame(SynthPose(), "frontal").pixel_xy, cfg) == "frontal"
    assert detect_view(make_frame(SynthPose(), "sagittal").pixel_xy, cfg) == "sagittal"
    half = make_frame(SynthPose(trunk_rotation_deg=60), "frontal").pixel_xy  # width ratio 0.36
    assert detect_view(half, cfg) is None


def test_allowed_views() -> None:
    SessionEngine(EX["elbow_flexion"], "L", view="frontal")
    with pytest.raises(ValueError, match="does not support"):
        SessionEngine(EX["shoulder_flexion"], "R", view="frontal")
    with pytest.raises(ValueError):
        SessionEngine(EX["shoulder_flexion"], "X")  # type: ignore[arg-type]


def test_pause_and_new_set() -> None:
    frames = rep_sequence("shoulder_flexion", [RepSpec(150)] * 2)
    eng = SessionEngine(EX["shoulder_flexion"], "R")
    for f in frames[:100]:
        eng.process(f)  # type: ignore[arg-type]
    eng.pause()
    paused = eng.process(frames[100])  # type: ignore[arg-type]
    assert paused.phase == "rest" and paused.cue is None and not paused.flags
    eng.start_set()
    for f in frames[101:]:
        eng.process(f)  # type: ignore[arg-type]
    assert len(eng.reps) >= 1 and eng.active
    eng.end_set()
    assert not eng.active


def test_left_side_and_summary() -> None:
    frames = rep_sequence("shoulder_abduction", [RepSpec(130)] * 3, side="L")
    eng = SessionEngine(EX["shoulder_abduction"], "L")
    for f in frames:
        eng.process(f)  # type: ignore[arg-type]
    s = eng.summary()
    assert s["rep_count"] == 3 and s["labels"] == {"correct": 3} and s["quality_score"] == 1.0
    assert s["mean_rom"] == pytest.approx(130, abs=2) and s["side"] == "L"


def test_smoothing_off_still_counts() -> None:
    frames = rep_sequence("shoulder_flexion", [RepSpec(150)] * 3, noise_px=0.5)
    eng, _ = run("shoulder_flexion", frames, config=EngineConfig(smoothing=False))
    assert len(eng.reps) == 3


# --- performance --------------------------------------------------------------------------------


@pytest.mark.no_cover  # coverage tracing slows Python ~2.4x; timing would be meaningless
def test_engine_10k_frames_under_2s() -> None:
    """Acceptance: 10k frames processed in < 2 s on CPU (frame generation excluded)."""
    base = [
        f for f in rep_sequence("shoulder_flexion", [RepSpec(150)] * 4, seed=3) if f is not None
    ]
    frames = []
    period = base[-1].ts_ms + 33
    while len(frames) < 10_000:
        k = len(frames) // len(base)
        for f in base:
            frames.append(Frame(f.lm, f.world, f.ts_ms + k * period, f.img_w, f.img_h, len(frames)))
    frames = frames[:10_000]
    eng = SessionEngine(EX["shoulder_flexion"], "R")
    t0 = time.perf_counter()
    for f in frames:
        eng.process(f)
    elapsed = time.perf_counter() - t0
    print(
        f"engine: 10000 frames in {elapsed:.3f} s ({elapsed / 10:.3f} ms/frame), "
        f"reps={len(eng.reps)}"
    )
    assert elapsed < 2.0
    full_copies = 10_000 // len(base)  # each copy of the base set holds 4 reps
    assert 4 * full_copies <= len(eng.reps) <= 4 * (full_copies + 1)


def test_experimental_rules_off_by_default() -> None:
    reps = [RepSpec(130, compensation={"plane_deg": 40.0})]
    eng, _ = run("shoulder_abduction", rep_sequence("shoulder_abduction", reps))
    assert eng.reps[0].flags == ()
