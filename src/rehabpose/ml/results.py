"""Collect measured numbers from reports/ into a committed summary the docs are checked against."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

CONFIG_RE = re.compile(r"^(?P<config>.+)-\d{8}-\d{6}$")
COMMIT_RE = re.compile(r"\bgit (\S+?)[;,\s]")
RUN_ID_RE = re.compile(r"MLflow run id: `(\w+)`")


def _config_summary(path: Path, md_text: str, root: Path) -> dict[str, Any]:
    """Fields of one evaluation report that the docs quote (macro-F1, CI, latency in ms)."""
    m = json.loads(path.read_text())
    run_id = RUN_ID_RE.search(md_text)
    test = m["test_seed0"]
    return {
        "report": path.with_suffix(".md").relative_to(root).as_posix(),
        "mlflow_run_id": run_id.group(1) if run_id else None,
        "synthetic": bool(m["synthetic"]),
        "test_subjects": m["split"]["test_subjects"],
        "n_windows": m["n_windows"],
        "n_test_windows": m["n_test_windows"],
        "cv_macro_f1": {k: m["cv_val_macro_f1"][k] for k in ("mean", "std", "n")},
        "test_macro_f1_by_seed": m["test_macro_f1_by_seed"],
        "test_seed0_ci95": m["test_seed0_macro_f1_ci95"],
        "per_class_f1": test["per_class_f1"],
        "per_class_auroc": test.get("per_class_auroc"),
        "latency_p95_ms": {d: v["p95"] for d, v in m["latency_ms"].items()},
    }


def _latest_reports(eval_dir: Path, commit: str) -> dict[str, tuple[Path, str]]:
    """Newest report per config whose commit is exactly `commit` (dirty runs do not count)."""
    found: dict[str, tuple[Path, str]] = {}
    for path in sorted(eval_dir.glob("*.json")):
        match = CONFIG_RE.match(path.stem)
        md = path.with_suffix(".md")
        if match is None or not md.is_file():
            continue
        text = md.read_text()
        run_commit = COMMIT_RE.search(text)
        if run_commit and run_commit.group(1) == commit:
            found[match["config"]] = (path, text)
    return found


def summarize(root: Path, commit: str, ablations: Path) -> dict[str, Any]:
    """Build the summary from reports/eval, one ablation report and reports/benchmarks.json."""
    reports = _latest_reports(root / "reports" / "eval", commit)
    if not reports:
        raise ValueError(f"no evaluation reports at commit {commit}")
    configs = {
        name: _config_summary(path, text, root) for name, (path, text) in sorted(reports.items())
    }
    engine: dict[str, Any] = {}
    rep_configs = sorted(name for name in reports if name.startswith("rep_"))
    if rep_configs:
        eng = json.loads(reports[rep_configs[0]][0].read_text())["engine"]
        rules = eng["rules_rep_labels"]
        engine = {
            "rep_count_mae": eng["rep_count_mae"],
            "angle_agreement": eng["angle_agreement"],
            "rule_baseline": {k: rules[k] for k in ("macro_f1", "accuracy", "missed_reps")},
        }
    return {
        "synthetic": all(c["synthetic"] for c in configs.values()),
        "commit": commit,
        "configs": configs,
        "engine": engine,
        "ablations": {
            "report": ablations.with_suffix(".md").relative_to(root).as_posix(),
            "results": json.loads(ablations.read_text()),
        },
        "benchmarks": json.loads((root / "reports" / "benchmarks.json").read_text()),
    }


def results_table(summary: dict[str, Any]) -> str:
    """Markdown table of every config: CV and test macro-F1, seed-0 CI and p95 latency (ms)."""
    rows = [
        "| Config | MLflow run | CV macro-F1 | Test macro-F1 by seed | Seed 0 95 % CI "
        "| p95 ms CPU / MPS |",
        "|---|---|---|---|---|---|",
    ]
    for name, c in summary["configs"].items():
        cv = c["cv_macro_f1"]
        seeds = ", ".join(f"{v:.3f}" for v in c["test_macro_f1_by_seed"])
        low, high = c["test_seed0_ci95"]
        latency = c["latency_p95_ms"]
        mps = f"{latency['mps']:.2f}" if "mps" in latency else "n/a"
        rows.append(
            f"| `{name}` | `{(c['mlflow_run_id'] or 'none')[:8]}` "
            f"| {cv['mean']:.3f} ± {cv['std']:.3f} | {seeds} | {low:.3f} to {high:.3f} "
            f"| {latency['cpu']:.2f} / {mps} |"
        )
    return "\n".join(rows)


def _flatten(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, dict):
        return [v for item in value.values() for v in _flatten(item)]
    if isinstance(value, list):
        return [v for item in value for v in _flatten(item)]
    return []


def unmeasured_decimals(text: str, summary: dict[str, Any]) -> list[str]:
    """Decimal numbers in `text` that no summary value rounds to at the same precision."""
    values = [abs(v) for v in _flatten(summary)]
    missing = []
    for token in re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", text):
        places = len(token.partition(".")[2])
        if not any(f"{v:.{places}f}" == token for v in values):
            missing.append(token)
    return missing
