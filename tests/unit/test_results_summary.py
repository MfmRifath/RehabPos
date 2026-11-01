"""Docs quote only numbers found in reports/results_summary.json, which `summarize` builds."""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from rehabpose.data.loaders import REGISTRY
from rehabpose.ml.results import results_table, summarize, unmeasured_decimals

ROOT = Path(__file__).resolve().parents[2]
SUMMARY: dict[str, Any] = json.loads((ROOT / "reports" / "results_summary.json").read_text())


def _section(name: str, heading: str) -> str:
    text = (ROOT / name).read_text()
    start = text.index(f"\n## {heading}")
    end = text.find("\n## ", start + 1)
    return text[start : end if end != -1 else None]


def _write_report(root: Path, stem: str, commit: str, mcf1: float, run_id: str = "abc123") -> None:
    eval_dir = root / "reports" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "synthetic": True,
        "n_windows": 10,
        "n_test_windows": 2,
        "split": {"test_subjects": ["S1", "S2"]},
        "cv_val_macro_f1": {"mean": 0.5, "std": 0.1, "n": 15},
        "test_macro_f1_by_seed": [mcf1],
        "test_seed0_macro_f1_ci95": [0.4, 0.6],
        "test_seed0": {"per_class_f1": {"GOOD": 0.5}},
        "latency_ms": {"cpu": {"median": 1.0, "p95": 2.0}},
        "engine": {
            "rep_count_mae": 0.0,
            "angle_agreement": {},
            "rules_rep_labels": {"macro_f1": 0.9, "accuracy": 0.9, "missed_reps": 0},
        },
    }
    (eval_dir / f"{stem}.json").write_text(json.dumps(report))
    (eval_dir / f"{stem}.md").write_text(
        f"- Run: `x` · MLflow run id: `{run_id}`\n- Environment: rehabpose 0.1.0, git {commit}; dev"
    )


def test_summarize_takes_newest_report_at_the_exact_commit(tmp_path: Path) -> None:
    _write_report(tmp_path, "rep_a-20260101-000000", "abc", 0.1)
    _write_report(tmp_path, "rep_a-20260102-000000", "abc", 0.2, run_id="newest")
    _write_report(tmp_path, "rep_a-20260103-000000", "abc-dirty", 0.3)
    _write_report(tmp_path, "rep_a-20260104-000000", "other", 0.4)
    (tmp_path / "reports" / "benchmarks.json").write_text("{}")
    ablations = tmp_path / "reports" / "abl.json"
    ablations.write_text("{}")

    summary = summarize(tmp_path, "abc", ablations)

    config = summary["configs"]["rep_a"]
    assert config["test_macro_f1_by_seed"] == [0.2]
    assert config["mlflow_run_id"] == "newest"
    assert summary["engine"]["rule_baseline"]["macro_f1"] == 0.9
    assert summary["synthetic"] is True


def test_summarize_without_reports_at_commit_fails(tmp_path: Path) -> None:
    (tmp_path / "reports" / "eval").mkdir(parents=True)
    with pytest.raises(ValueError, match="no evaluation reports"):
        summarize(tmp_path, "abc", tmp_path / "abl.json")


def test_unmeasured_decimals_checks_value_and_precision() -> None:
    summary = {"a": 2.5678, "b": [-6.63], "c": {"d": 10}}
    assert unmeasured_decimals("MAE 2.57 and -6.63 and 10.0", summary) == []
    assert unmeasured_decimals("MAE 2.58", summary) == ["2.58"]


def test_summary_is_all_synthetic_at_one_commit() -> None:
    assert SUMMARY["synthetic"] is True
    assert len(SUMMARY["configs"]) == 10
    assert re.fullmatch(r"[0-9a-f]{7}", SUMMARY["commit"])


@pytest.mark.parametrize("name", ["README.md", "docs/10_model_card.md"])
def test_docs_hold_the_generated_results_table(name: str) -> None:
    assert results_table(SUMMARY) in (ROOT / name).read_text()


@pytest.mark.parametrize(
    ("name", "heading"),
    [
        ("README.md", "Results"),
        ("docs/10_model_card.md", "Metrics"),
        ("docs/portfolio.md", "CV bullets"),
    ],
)
def test_quoted_numbers_are_measured(name: str, heading: str) -> None:
    assert unmeasured_decimals(_section(name, heading), SUMMARY) == []


@pytest.mark.parametrize(
    ("name", "heading"), [("README.md", "Results"), ("docs/10_model_card.md", "Metrics")]
)
def test_results_are_labelled_synthetic(name: str, heading: str) -> None:
    assert "SYNTHETIC" in _section(name, heading)


def test_non_commercial_dataset_licence_is_printed() -> None:
    assert "BY-NC" in REGISTRY["rehab24_6"].info.license
    readme = _section("README.md", "Datasets and licences")
    assert "CC BY-NC 4.0" in readme
