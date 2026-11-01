"""`rehabpose serve`: TLS options reach uvicorn; a non-loopback bind warns about no login."""

import logging
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from rehabpose.cli import app

runner = CliRunner()


@pytest.fixture
def uvicorn_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("uvicorn.run", lambda _app, **kw: calls.append(kw))
    monkeypatch.setattr("rehabpose.api.main.create_app", lambda _settings: object())
    return calls


def _pem_pair(tmp_path: Path) -> tuple[str, str]:
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert.write_text("cert")
    key.write_text("key")
    return str(cert), str(key)


def test_plain_serve_has_no_tls(uvicorn_calls: list[dict[str, Any]]) -> None:
    assert runner.invoke(app, ["serve", "--port", "9001"]).exit_code == 0
    assert uvicorn_calls[0]["port"] == 9001
    assert uvicorn_calls[0]["ssl_certfile"] is None


def test_tls_files_reach_uvicorn(uvicorn_calls: list[dict[str, Any]], tmp_path: Path) -> None:
    cert, key = _pem_pair(tmp_path)
    result = runner.invoke(app, ["serve", "--ssl-certfile", cert, "--ssl-keyfile", key])
    assert result.exit_code == 0
    assert uvicorn_calls[0]["ssl_certfile"] == cert
    assert uvicorn_calls[0]["ssl_keyfile"] == key


def test_tls_needs_both_files(uvicorn_calls: list[dict[str, Any]], tmp_path: Path) -> None:
    cert, _ = _pem_pair(tmp_path)
    assert runner.invoke(app, ["serve", "--ssl-certfile", cert]).exit_code != 0
    assert not uvicorn_calls


def test_missing_cert_file_is_rejected(uvicorn_calls: list[dict[str, Any]], tmp_path: Path) -> None:
    _, key = _pem_pair(tmp_path)
    args = ["serve", "--ssl-certfile", str(tmp_path / "nope.pem"), "--ssl-keyfile", key]
    assert runner.invoke(app, args).exit_code != 0
    assert not uvicorn_calls


@pytest.mark.parametrize(("host", "warns"), [("127.0.0.1", False), ("0.0.0.0", True)])
def test_lan_bind_warns_about_no_login(
    uvicorn_calls: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
    host: str,
    warns: bool,
) -> None:
    # The CLI callback resets the root logger's handlers, so attach to the package logger.
    logger = logging.getLogger("rehabpose")
    logger.addHandler(caplog.handler)
    try:
        assert runner.invoke(app, ["serve", "--host", host]).exit_code == 0
    finally:
        logger.removeHandler(caplog.handler)
    assert ("no login" in caplog.text) is warns
