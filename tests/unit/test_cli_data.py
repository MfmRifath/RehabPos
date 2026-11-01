"""`rehabpose data ...` commands on fixture datasets (data dir redirected via env)."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from rehabpose.cli import app
from rehabpose.data.splits import load_plan
from rehabpose.settings import get_settings
from tests.fixtures.datasets import make_intellirehab, make_own


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    d = tmp_path / "data"
    monkeypatch.setenv("REHABPOSE_DATA_DIR", str(d))
    get_settings.cache_clear()
    yield d
    get_settings.cache_clear()


def run(*args: str) -> Result:
    return CliRunner().invoke(app, ["data", *args])


def test_list_and_info(data_dir: Path) -> None:
    make_intellirehab(data_dir)
    r = run("list")
    assert r.exit_code == 0
    out = r.output
    assert "intellirehab" in out and "CC BY 4.0" in out and "kimore" in out
    r = run("info", "uiprmd")
    assert (
        r.exit_code == 0
        and "ODC PDDL" in r.output
        and "Layout verified on real files: False" in r.output
    )
    assert run("info", "nope").exit_code != 0


def test_stats_requires_license_then_works(data_dir: Path) -> None:
    make_intellirehab(data_dir)
    r = run("stats", "intellirehab")
    assert r.exit_code == 2
    r = run("stats", "intellirehab", "--i-accept-license")
    assert r.exit_code == 0, r.output
    stats = json.loads(r.output[r.output.index("{") :])
    assert stats["sequences"] == 4 and stats["subjects"] == 2


def test_stats_missing_dataset_exit_2(data_dir: Path) -> None:
    assert run("stats", "kimore", "--i-accept-license").exit_code == 2


def test_extract_all_skeletons(data_dir: Path) -> None:
    make_intellirehab(data_dir)
    r = run("extract-all", "intellirehab", "--i-accept-license")
    assert r.exit_code == 0, r.output
    assert json.loads(r.output.strip().splitlines()[-1]) == {"written": 4}
    assert len(list((data_dir / "interim" / "intellirehab").glob("*.parquet"))) == 4


def test_splits_and_schema(data_dir: Path, tmp_path: Path) -> None:
    make_own(data_dir, n_subjects=4)
    out = tmp_path / "splits.yaml"
    r = run("splits", "own", "--out", str(out))
    assert r.exit_code == 0, r.output
    plan = load_plan(out)
    assert plan.method == "loso" and len(plan.test) == 1 and len(plan.subjects) == 4
    r = run("schema")
    assert r.exit_code == 0 and json.loads(r.output)["title"] == "Annotation"
