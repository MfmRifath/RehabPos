"""Labelling state logic and the OpenCV loop driven by a scripted fake UI (no window)."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from rehabpose.data.labeling import LabelState
from rehabpose.data.labeling.tool import FrameSource, render, run_labeler
from rehabpose.data.schema import load_annotation
from tests.fixtures.synthetic import RepSpec, rep_sequence, write_frames_video


def _state(n: int = 300) -> LabelState:
    return LabelState(
        video_id="v", subject="P01", exercise_id="shoulder_flexion", side="R",
        view="sagittal", fps=30.0, n_frames=n, annotator="A1",
    )  # fmt: skip


def _keys(st: LabelState, keys: str) -> None:
    for ch in keys:
        assert st.handle_key(ord(ch))


def test_mark_rep_label_and_order() -> None:
    st = _state()
    st.goto(100)
    _keys(st, "s")
    st.goto(130)
    _keys(st, "p")
    st.goto(160)
    _keys(st, "e")
    st.goto(10)
    _keys(st, "s")
    st.goto(40)
    _keys(st, "p")
    st.goto(70)
    _keys(st, "e")  # earlier rep added second: re-sorted and re-indexed
    assert [(r.idx, r.start_frame) for r in st.reps] == [(0, 10), (1, 100)]
    st.goto(120)
    _keys(st, "4")  # label the rep under the cursor
    _keys(st, "2")
    assert [r.label for r in st.reps] == ["correct", "incomplete_rom"]


@pytest.mark.parametrize(
    ("marks", "msg"),
    [([("e", 10)], "start and peak"), ([("s", 50), ("p", 40), ("e", 60)], "start <= peak")],
)
def test_invalid_marks_rejected(marks: list[tuple[str, int]], msg: str) -> None:
    st = _state()
    for key, frame in marks:
        st.goto(frame)
        _keys(st, key)
    assert st.reps == [] and msg in st.message


def test_overlapping_rep_rejected() -> None:
    st = _state()
    for k, f in (("s", 10), ("p", 20), ("e", 30), ("s", 25), ("p", 35), ("e", 45)):
        st.goto(f)
        _keys(st, k)
    assert len(st.reps) == 1 and "overlaps" in st.message


def test_segments_flags_and_angles_and_undo() -> None:
    st = _state()
    for k, f in (("s", 10), ("p", 40), ("e", 80)):
        st.goto(f)
        _keys(st, k)
    _keys(st, "kk")  # TRUNK_LEAN -> TRUNK_ROTATION -> SHOULDER_HIKE
    assert st.code == "SHOULDER_HIKE"
    st.goto(30)
    _keys(st, "c")
    st.goto(50)
    _keys(st, "c")
    assert st.segments[0].code == "SHOULDER_HIKE" and (
        st.segments[0].start_frame,
        st.segments[0].end_frame,
    ) == (30, 50)
    st.goto(40)
    for x, y in ((10, 0), (0, 0), (0, 10)):
        st.click(x, y)
    assert (
        st.angles[0].angle_deg == pytest.approx(90)
        and st.angles[0].measure == "upper_arm_trunk_angle"
    )
    ann = st.to_annotation()
    assert ann.reps[0].flags == ["SHOULDER_HIKE"]  # derived from overlapping segments
    _keys(st, "uuu")  # undo the angle (3 clicks)
    assert st.angles == [] and st.clicks == []
    _keys(st, "u")  # undo closing the segment
    assert st.segments == [] and st.open_segment is not None


def test_degenerate_angle_discarded_and_measure_cycle() -> None:
    st = _state()
    _keys(st, "n")
    assert st.measure == "elbow_interior"
    for _ in range(3):
        st.click(5, 5)
    assert st.angles == [] and "degenerate" in st.message


def test_navigation_clamps_and_quit() -> None:
    st = _state(50)
    _keys(st, "A")
    assert st.frame == 0
    _keys(st, "DDDDDDd")
    assert st.frame == 49
    assert st.handle_key(-1) and not st.handle_key(ord("q")) and not st.handle_key(27)


@pytest.fixture
def video(tmp_path: Path) -> Path:
    frames = rep_sequence(
        "shoulder_abduction", [RepSpec(130, duration_s=3.0)], img_w=320, img_h=240, px_per_m=160
    )
    return write_frames_video(tmp_path / "clip.mp4", frames[:120])


class FakeUI:
    """Scripted keys; ("click", x, y) items trigger the registered mouse callback."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.shown = 0
        self.callback: Callable[..., None] | None = None

    def imshow(self, name: str, img: np.ndarray) -> None:
        assert img.ndim == 3
        self.shown += 1

    def set_mouse(self, name: str, cb: Callable[..., None]) -> None:
        self.callback = cb

    def wait_key(self, delay: int) -> int:
        while self.script and isinstance(self.script[0], tuple):
            _, x, y = self.script.pop(0)
            assert self.callback is not None
            self.callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
        if not self.script:
            return ord("q")
        item = self.script.pop(0)
        return -1 if item is None else ord(item)


def _run(video: Path, out: Path, script: list[Any]) -> tuple[FakeUI, Any]:
    ui = FakeUI(script)
    ann = run_labeler(
        video, _state(0), out, imshow=ui.imshow, wait_key=ui.wait_key, set_mouse=ui.set_mouse,
        named_window=lambda name: None, destroy=lambda: None,
    )  # fmt: skip
    return ui, ann


def test_run_labeler_scripted_session(video: Path, tmp_path: Path) -> None:
    out = tmp_path / "labels" / "clip.json"
    script: list[Any] = (
        ["D", "s", "D", "D", "D", "p", "D", "D", "D", "D", "e", "2", "c", "d", "d", "c"]
        + [("click", 100, 50), ("click", 100, 100), ("click", 150, 100)]
        + [None, "w", " ", None, None, " ", "q"]
    )
    ui, ann = _run(video, out, script)
    assert ui.shown > 10
    saved = load_annotation(out)
    assert saved == ann
    assert [(r.start_frame, r.peak_frame, r.end_frame, r.label) for r in saved.reps] == [
        (10, 40, 80, "incomplete_rom")
    ]
    assert len(saved.compensation_segments) == 1 and saved.reps[0].flags == ["TRUNK_LEAN"]
    assert saved.angle_annotations[0].angle_deg == pytest.approx(90)


def test_run_labeler_play_until_end(video: Path, tmp_path: Path) -> None:
    ui, ann = _run(video, tmp_path / "p.json", [" "] + [None] * 200)
    assert ann.reps == [] and ui.shown >= 120


def test_frame_source_and_render(video: Path) -> None:
    src = FrameSource(video)
    assert src.n_frames == 120 and src.fps == pytest.approx(30)
    a, b = src.get(5), src.get(6)
    assert a.shape == (240, 320, 3) and src.get(6) is b and src.get(2).shape == a.shape
    img = render(a, _state(120))
    assert img.shape == a.shape and not np.array_equal(img, a)
    src.close()
    with pytest.raises(ValueError):
        FrameSource(video.with_name("missing.mp4"))


def test_cli_label_uses_run_labeler(
    video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rehabpose.data.labeling.tool as tool
    from rehabpose.cli import app

    seen: dict[str, Any] = {}

    def fake_run(v: Path, state: LabelState, out: Path) -> Any:
        seen.update(video=v, state=state, out=out)
        return state.to_annotation()

    monkeypatch.setattr(tool, "run_labeler", fake_run)
    out = tmp_path / "x.json"
    r = CliRunner().invoke(
        app,
        [
            "label",
            str(video),
            "--exercise",
            "shoulder_abduction",
            "--subject",
            "P07",
            "--side",
            "L",
            "--out",
            str(out),
        ],
    )
    assert r.exit_code == 0, r.output
    st = seen["state"]
    assert (st.subject, st.side, st.view, st.n_frames, st.video_id) == (
        "P07",
        "L",
        "frontal",
        120,
        "clip",
    )
    bad = CliRunner().invoke(app, ["label", str(video), "--exercise", "squat", "--subject", "P1"])
    assert bad.exit_code != 0
