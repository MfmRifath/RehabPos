import io
from pathlib import Path
from typing import Any

import pytest

from rehabpose.pose import model_files as mf


def test_urls_and_paths() -> None:
    for v in mf.VARIANTS:
        url = mf.model_url(v)
        assert url.startswith("https://storage.googleapis.com/mediapipe-models/pose_landmarker/")
        assert url.endswith(f"pose_landmarker_{v}.task") and "float16" in url
    assert mf.model_path("full", Path("data/models")) == Path(
        "data/models/pose_landmarker_full.task"
    )
    with pytest.raises(ValueError):
        mf.model_url("huge")  # type: ignore[arg-type]


def test_missing_model_raises_with_url_and_never_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: Any, **k: Any) -> None:
        raise AssertionError("must not touch the network without consent")

    monkeypatch.setattr(mf.urllib.request, "urlopen", boom)
    with pytest.raises(mf.ModelFileMissingError) as exc:
        mf.resolve_model("lite", tmp_path)
    assert mf.model_url("lite") in str(exc.value)
    assert "--download-model" in str(exc.value)


def test_existing_model_resolves(tmp_path: Path) -> None:
    p = mf.model_path("heavy", tmp_path)
    p.write_bytes(b"x")
    assert mf.resolve_model("heavy", tmp_path) == p


class _FakeResp(io.BytesIO):
    def __init__(self, data: bytes, length: int | None = None) -> None:
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data) if length is None else length)}

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def test_download_with_consent_writes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        mf.urllib.request, "urlopen", lambda url, timeout: _FakeResp(b"model-bytes")
    )
    path = mf.resolve_model("full", tmp_path, download=True)
    assert path.read_bytes() == b"model-bytes"
    assert not list(tmp_path.glob("*.part"))


def test_download_refuses_over_50mb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mf.urllib.request, "urlopen", lambda url, timeout: _FakeResp(b"", length=60 * 1024 * 1024)
    )
    with pytest.raises(RuntimeError, match="50 MB"):
        mf.download_model("full", tmp_path)
    assert not mf.model_path("full", tmp_path).exists()
