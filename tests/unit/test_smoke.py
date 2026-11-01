"""Phase 0 smoke tests: package import, CLI wiring, settings precedence."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

import rehabpose
from rehabpose.cli import app
from rehabpose.settings import CONFIG_ENV_VAR, Settings

SUBCOMMANDS = ["extract", "analyze", "serve", "data", "train", "evaluate", "bench", "label"]


def test_package_metadata() -> None:
    assert rehabpose.__version__
    assert "not a medical device" in rehabpose.DISCLAIMER


def test_cli_help_lists_subcommands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in SUBCOMMANDS:
        assert name in result.output


def test_settings_yaml_then_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "app.yaml"
    cfg.write_text("port: 9001\npose_model: lite\n")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(cfg))

    assert Settings().port == 9001
    assert Settings().pose_model == "lite"

    monkeypatch.setenv("REHABPOSE_PORT", "9002")
    assert Settings().port == 9002


def test_relative_paths_resolve_against_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rehabpose.settings import project_root

    root = project_root()
    assert (root / "configs" / "app.yaml").is_file()  # source checkout detected
    monkeypatch.chdir(tmp_path)  # e.g. running `rehabpose serve` from frontend/
    s = Settings()
    assert s.exercises_dir == root / "configs" / "exercises"
    assert s.data_dir == root / "data" and s.frontend_dist == root / "frontend" / "dist"
    assert s.db_url == f"sqlite:///{root / 'data' / 'rehabpose.db'}"
    assert Settings(data_dir=tmp_path / "abs").data_dir == tmp_path / "abs"  # absolute kept
    assert Settings(db_url="sqlite:///:memory:").db_url == "sqlite:///:memory:"

    monkeypatch.setenv("REHABPOSE_HOME", str(tmp_path))
    assert Settings(data_dir=Path("d")).data_dir == tmp_path / "d"
