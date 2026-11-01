"""Evaluation report (markdown) and figures; every page carries the SYNTHETIC banner when due."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rehabpose import DISCLAIMER
from rehabpose.ml.train import TrainConfig

SERIES = "#2a78d6"  # categorical slot 1 (validated)
RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"]  # blue 100..700 (sequential)
INK, MUTED, GRID = "#0b0b0b", "#5f5e58", "#e4e3df"


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def confusion_figure(
    path: Path, cm: list[list[int]], classes: tuple[str, ...] | list[str], title: str
) -> Path:
    """Counts heatmap, single-hue sequential ramp, values annotated (text in ink, not colour)."""
    from matplotlib.colors import LinearSegmentedColormap

    plt = _plt()
    a = np.asarray(cm, dtype=float)
    fig, ax = plt.subplots(figsize=(4.6, 3.9), dpi=160)
    ax.imshow(a, cmap=LinearSegmentedColormap.from_list("blue", RAMP), vmin=0)
    ax.set_xticks(
        range(len(classes)), [c.replace("_", "\n") for c in classes], fontsize=7, color=MUTED
    )
    ax.set_yticks(
        range(len(classes)), [c.replace("_", " ") for c in classes], fontsize=7, color=MUTED
    )
    ax.set_xlabel("predicted", fontsize=8, color=MUTED)
    ax.set_ylabel("true", fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    hi = a.max() if a.size else 1
    for (i, j), v in np.ndenumerate(a):
        ax.text(
            j,
            i,
            f"{int(v)}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if v > 0.55 * hi else INK,
        )
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def bland_altman_figure(
    path: Path, pairs: list[tuple[float, float]], title: str, max_points: int = 4000
) -> Path:
    """Mean vs difference with bias and 95 % limits of agreement (direct labels)."""
    plt = _plt()
    a = np.asarray(pairs, dtype=float)
    if len(a) > max_points:
        a = a[np.random.default_rng(0).choice(len(a), max_points, replace=False)]
    mean, diff = a.mean(1), a[:, 0] - a[:, 1]
    bias, sd = diff.mean(), diff.std(ddof=1)
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=160)
    ax.scatter(mean, diff, s=6, color=SERIES, alpha=0.35, linewidths=0)
    for y, lab, style in (
        (bias, f"bias {bias:+.2f}°", "-"),
        (bias + 1.96 * sd, f"+1.96 SD {bias + 1.96 * sd:+.2f}°", "--"),
        (bias - 1.96 * sd, f"-1.96 SD {bias - 1.96 * sd:+.2f}°", "--"),
    ):
        ax.axhline(y, color=MUTED, linewidth=1, linestyle=style)
        ax.annotate(
            lab,
            xy=(1.0, y),
            xycoords=("axes fraction", "data"),
            xytext=(-2, 3),
            textcoords="offset points",
            ha="right",
            fontsize=7,
            color=MUTED,
        )
    ax.set_xlabel("mean of engine and truth (deg)", fontsize=8, color=MUTED)
    ax.set_ylabel("engine - truth (deg)", fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=7, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def figures(
    out_dir: Path,
    run_name: str,
    cfg: TrainConfig,
    test: dict[str, Any],
    angle_pairs: dict[str, list[tuple[float, float]]],
) -> list[Path]:
    """Write figures next to the report."""
    fig_dir = Path(out_dir) / f"{run_name}_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = []
    if cfg.task == "rep":
        out.append(
            confusion_figure(
                fig_dir / "confusion.png",
                test["confusion"],
                list(test["per_class_f1"]),
                f"SYNTHETIC test confusion: {cfg.model}",
            )
        )
    for key, pairs in angle_pairs.items():
        if pairs:
            out.append(
                bland_altman_figure(
                    fig_dir / f"bland_altman_{key}.png",
                    pairs,
                    f"SYNTHETIC engine {key} angle vs ground truth",
                )
            )
    return out


def _f(v: Any, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def write_report(
    out_dir: Path,
    run_name: str,
    cfg: TrainConfig,
    m: dict[str, Any],
    figs: list[Path],
    run_id: str | None,
    env: dict[str, str],
    command: str,
    bundle_dir: Path | None,
) -> Path:
    """Markdown report under reports/eval/<run_name>.md."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t, eng = m["test_seed0"], m["engine"]
    exercises = ", ".join(cfg.data.exercises)
    ci_lo, ci_hi = (_f(v) for v in m["test_seed0_macro_f1_ci95"])
    ci = (
        f"95 % CI over subjects {ci_lo}–{ci_hi}"
        if len(m["split"]["test_subjects"]) >= 2
        else "95 % CI n/a: needs >= 2 test subjects"
    )
    lines = [
        f"# Evaluation: {cfg.name} ({cfg.task}, {cfg.model})",
        "",
        "> "
        + (
            "**SYNTHETIC — not a result.** Trained and evaluated on generated skeletons only; "
            "these numbers say nothing about real users and must not be quoted as results."
            if m.get("synthetic")
            else ""
        ),
        "",
        f"- Run: `{run_name}` · MLflow run id: `{run_id or 'not logged'}`",
        f"- Command: `{command or 'n/a'}`",
        f"- Environment: {', '.join(f'{k} {v}' for k, v in env.items())}; device {m['device']}",
        f"- Data: {cfg.data.source}, {cfg.data.n_subjects} subjects, exercises {exercises}; "
        f"{m['n_windows']} windows ({m['n_test_windows']} test)",
        f"- Split: {m['split']['method']} with {m['split']['folds']} folds over non-test subjects; "
        f"held-out test subjects {', '.join(m['split']['test_subjects'])}",
        f"- Class counts (all): {json.dumps(m['class_counts'])}",
        f"- Elapsed: {m.get('elapsed_s', 'n/a')} s",
        "",
        "## Model selection (cross-validation)",
        "",
        f"Validation macro-F1 over folds x seeds (n={m['cv_val_macro_f1']['n']}): "
        f"{_f(m['cv_val_macro_f1']['mean'])} ± {_f(m['cv_val_macro_f1']['std'])}. "
        "Early stopping used these validation folds, so this estimate is optimistic.",
        "",
        "## Held-out test subjects",
        "",
        "| metric | value |",
        "|---|---|",
        f"| macro-F1 (seed 0) | {_f(t['macro_f1'])} ({ci}) |",
        f"| macro-F1 by seed | {', '.join(_f(v) for v in m['test_macro_f1_by_seed'])} |",
    ]
    if cfg.task == "rep":
        lines += [
            f"| accuracy (seed 0) | {_f(t['accuracy'])} |",
            f"| AUROC one-vs-rest (seed 0) | {_f(t['auroc_ovr'])} |",
        ]
    else:
        lines.append(f"| mean AUROC over codes (seed 0) | {_f(t['mean_auroc'])} |")
    lines += [
        "",
        "Per-class F1 (seed 0): " + ", ".join(f"{k} {_f(v)}" for k, v in t["per_class_f1"].items()),
        "",
    ]
    if cfg.task == "rep":
        rules = eng["rules_rep_labels"]
        lines += [
            "## Baseline: rule engine on the same test sets",
            "",
            f"Rule labels vs ground truth: macro-F1 {_f(rules['macro_f1'])}, "
            f"accuracy {_f(rules['accuracy'])}, "
            f"reps missed by the engine {rules['missed_reps']}. Per class: "
            + ", ".join(f"{k} {_f(v)}" for k, v in rules["per_class_f1"].items()),
            "",
        ]
    lines += [
        "## Deterministic pipeline on test sets (T3/T4)",
        "",
        f"- Rep-count MAE per set: {_f(eng['rep_count_mae'])} "
        f"(exact sets: {_f(eng['rep_count_exact'])})",
    ]
    for k, a in eng["angle_agreement"].items():
        lines.append(
            f"- {k} angle vs ground truth: MAE {_f(a['mae'], 2)}°, RMSE {_f(a['rmse'], 2)}°, "
            f"bias {_f(a['bias'], 2)}°, 95 % LoA [{_f(a['loa_low'], 2)}, {_f(a['loa_high'], 2)}]° "
            f"(n={int(a['n'])} frames)"
        )
    lines += [
        "",
        "## Inference latency (one window, model only)",
        "",
        "| device | median ms | p95 ms |",
        "|---|---|---|",
    ]
    lines += [
        f"| {d} | {_f(v['median'], 3)} | {_f(v['p95'], 3)} |" for d, v in m["latency_ms"].items()
    ]
    if "export_parity" in m:
        lines += [
            "",
            "## Export",
            "",
            f"Bundle: `{bundle_dir}` (TorchScript + ONNX + meta.json). Parity vs eager PyTorch: "
            + ", ".join(f"{k} {v:.2e}" for k, v in m["export_parity"].items()),
        ]
    if "export_error" in m:
        lines += ["", f"**Export problem:** {m['export_error']}"]
    if figs:
        lines += ["", "## Figures", ""] + [
            f"![{p.stem}]({p.relative_to(out_dir).as_posix()})" for p in figs
        ]
    lines += ["", "---", "", DISCLAIMER, ""]
    path = out_dir / f"{run_name}.md"
    path.write_text("\n".join(lines))
    (out_dir / f"{run_name}.json").write_text(json.dumps(m, indent=2, default=str))
    return path
