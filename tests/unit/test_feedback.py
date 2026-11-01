import pytest

from rehabpose.core.feedback import SETUP_TEXTS, CueSelector, priority
from rehabpose.core.rules import Flag

TEXTS = {
    "TRUNK_LEAN": "Upright",
    "ELBOW_BEND": "Straight",
    "TOO_FAST": "Slow",
    "INCOMPLETE_ROM": "Higher",
}


def _f(code: str, conf: float = 1.0) -> Flag:
    return Flag(code, "warn", conf)


def test_priority_order() -> None:
    order = [
        "SETUP_VISIBILITY", "TRUNK_LEAN", "SHOULDER_HIKE",
        "ELBOW_BEND", "TOO_FAST", "INCOMPLETE_ROM",
    ]  # fmt: skip
    assert [priority(c) for c in order] == sorted(priority(c) for c in order)
    assert priority("CALIBRATING") == 0 and priority("TRUNK_ROTATION") == 1
    assert priority("SHOULDER_DRIFT") == priority("ELBOW_AWAY") == 3
    assert priority("SOMETHING_NEW") > priority("INCOMPLETE_ROM")


def test_single_highest_priority_cue_after_debounce() -> None:
    sel = CueSelector()
    flags = [_f("ELBOW_BEND"), _f("TRUNK_LEAN"), _f("TOO_FAST")]
    assert sel.update(0.0, flags, TEXTS) is None
    assert sel.update(0.49, flags, TEXTS) is None
    cue = sel.update(0.5, flags, TEXTS)
    assert cue is not None and cue.code == "TRUNK_LEAN" and cue.text == "Upright" and cue.speak


def test_flicker_shorter_than_debounce_never_shows() -> None:
    sel = CueSelector()
    for i in range(10):
        flags = [_f("TRUNK_LEAN")] if i % 2 == 0 else []
        assert sel.update(i * 0.1, flags, TEXTS) is None


def test_speech_rate_limited_to_one_per_3s() -> None:
    sel = CueSelector()
    spoken = []
    t = 0.0
    # alternate between two persistent cues every 1 s for 10 s
    while t < 10.0:
        code = "TRUNK_LEAN" if int(t) % 2 == 0 else "ELBOW_BEND"
        cue = sel.update(round(t, 2), [_f(code)], TEXTS)
        if cue and cue.speak:
            spoken.append(round(t, 2))
        t += 1 / 30
    gaps = [b - a for a, b in zip(spoken, spoken[1:], strict=False)]
    assert spoken and all(g >= 3.0 - 1e-9 for g in gaps)


def test_pending_speech_fires_when_allowed() -> None:
    sel = CueSelector()
    sel.update(0.0, [_f("TRUNK_LEAN")], TEXTS)
    assert sel.update(0.5, [_f("TRUNK_LEAN")], TEXTS).speak  # type: ignore[union-attr]
    sel.update(1.0, [_f("ELBOW_BEND")], TEXTS)
    cue = sel.update(1.5, [_f("ELBOW_BEND")], TEXTS)
    assert cue is not None and cue.code == "ELBOW_BEND" and not cue.speak  # 1 s after last speech
    later = sel.update(3.5, [_f("ELBOW_BEND")], TEXTS)
    assert later is not None and later.speak


def test_cleared_when_flags_disappear_and_setup_texts() -> None:
    sel = CueSelector()
    sel.update(0.0, [_f("SETUP_VISIBILITY")], {})
    cue = sel.update(0.6, [_f("SETUP_VISIBILITY")], {})
    assert cue is not None and cue.text == SETUP_TEXTS["SETUP_VISIBILITY"]
    assert sel.update(0.7, [], {}) is None


def test_ties_broken_by_confidence() -> None:
    sel = CueSelector(debounce_s=0.0)
    cue = sel.update(0.0, [_f("TRUNK_LEAN", 0.6), _f("TRUNK_ROTATION", 0.9)], {})
    assert cue is not None and cue.code == "TRUNK_ROTATION"


@pytest.mark.parametrize("debounce", [0.0, 0.5, 1.0])
def test_debounce_configurable(debounce: float) -> None:
    sel = CueSelector(debounce_s=debounce)
    sel.update(0.0, [_f("TRUNK_LEAN")], TEXTS)
    assert sel.update(debounce, [_f("TRUNK_LEAN")], TEXTS) is not None
