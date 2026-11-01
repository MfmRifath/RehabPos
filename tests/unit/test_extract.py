"""Video I/O + extract pipeline with a mocked estimator, and the `rehabpose extract` CLI."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from typer.testing import CliRunner

from rehabpose.cli import app
from rehabpose.core.landmarks import Frame
from rehabpose.data import video_io
from rehabpose.data.extract import extract_video, load_frames, read_meta
from rehabpose.pose.base import PoseEstimator
from rehabpose.settings import get_settings
from tests.fixtures.synthetic import SyntheticVideo, write_synthetic_video


class MockEstimator:
    """Returns ground-truth frames by timestamp; 'misses' every 7th frame."""

    def __init__(self, truth: list[Frame]) -> None:
        self.by_ts = {f.ts_ms: f for f in truth}
        self.calls: list[int] = []
        self.closed = False

    def process(self, frame_bgr: NDArray[np.uint8], ts_ms: int) -> Frame | None:
        assert frame_bgr.dtype == np.uint8 and frame_bgr.ndim == 3
        self.calls.append(ts_ms)
        idx = len(self.calls) - 1
        return None if idx % 7 == 6 else self.by_ts[ts_ms]

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def video(tmp_path: Path) -> SyntheticVideo:
    return write_synthetic_video(tmp_path / "stick.mp4", n_frames=30, fps=30.0)


def test_mock_satisfies_protocol(video: SyntheticVideo) -> None:
    assert isinstance(MockEstimator(video.frames), PoseEstimator)


def test_video_io_roundtrip(video: SyntheticVideo) -> None:
    info = video_io.probe(video.path)
    assert (info.width, info.height) == (640, 360)
    assert info.fps == pytest.approx(30.0)
    frames = list(video_io.iter_frames(video.path))
    assert len(frames) == 30
    assert [ts for _, ts, _ in frames[:4]] == [0, 33, 67, 100]
    assert frames[0][2].shape == (360, 640, 3)


def test_video_io_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        video_io.probe(tmp_path / "missing.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        video_io.probe(bad)


def test_extract_pipeline_with_mock(video: SyntheticVideo, tmp_path: Path) -> None:
    est = MockEstimator(video.frames)
    out = tmp_path / "out" / "lm.parquet"
    res = extract_video(video.path, est, out, meta={"model": "mock"})
    assert res.n_frames == 30 and res.n_detected == 30 - 4  # misses at 6, 13, 20, 27
    assert est.calls == sorted(est.calls)

    loaded = load_frames(out)
    assert len(loaded) == 30
    assert [i for i, f in enumerate(loaded) if f is None] == [6, 13, 20, 27]
    for i, fr in enumerate(loaded):
        if fr is None:
            continue
        truth = video.frames[i]
        np.testing.assert_allclose(fr.lm, truth.lm, atol=1e-6)  # float32 storage
        np.testing.assert_allclose(fr.world, truth.world, atol=1e-6)
        assert (fr.frame_idx, fr.ts_ms, fr.img_w, fr.img_h) == (i, truth.ts_ms, 640, 360)

    meta = read_meta(out)
    assert meta["model"] == "mock" and meta["fps"] == pytest.approx(30.0)
    assert meta["source"].endswith("stick.mp4")


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("REHABPOSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REHABPOSE_MODELS_DIR", str(tmp_path / "models"))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_cli_extract_with_mock(
    video: SyntheticVideo, cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rehabpose.pose as pose

    est = MockEstimator(video.frames)
    monkeypatch.setattr(pose, "create_estimator", lambda variant, models_dir, download: est)
    out = cli_env / "x.parquet"
    result = CliRunner().invoke(
        app, ["extract", str(video.path), "--model", "lite", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "30 frames, 26 with a person" in result.output
    assert out.is_file() and est.closed
    assert read_meta(out)["model"] == "lite"


def test_cli_extract_missing_model_exits_2(video: SyntheticVideo, cli_env: Path) -> None:
    result = CliRunner().invoke(app, ["extract", str(video.path)])
    assert result.exit_code == 2
    assert not (cli_env / "data" / "interim").exists()


def test_cli_extract_rejects_bad_variant(video: SyntheticVideo, cli_env: Path) -> None:
    result = CliRunner().invoke(app, ["extract", str(video.path), "--model", "huge"])
    assert result.exit_code != 0


def test_load_frames_rejects_foreign_parquet(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "other.parquet"
    pq.write_table(pa.table({"a": [1, 2]}), path)
    with pytest.raises(ValueError, match="schema mismatch"):
        load_frames(path)
