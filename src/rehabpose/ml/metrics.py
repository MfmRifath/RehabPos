"""Classification metrics with subject-level bootstrap CIs, and Bland-Altman agreement."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

FloatArray = NDArray[np.float64]


def rep_metrics(y: NDArray[Any], proba: FloatArray, classes: Sequence[str]) -> dict[str, Any]:
    """Multi-class: macro-F1, per-class F1, accuracy, one-vs-rest AUROC, confusion matrix."""
    labels = list(range(len(classes)))
    pred = proba.argmax(1)
    out: dict[str, Any] = {
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float((pred == y).mean()) if len(y) else float("nan"),
        "per_class_f1": dict(
            zip(
                classes,
                map(float, f1_score(y, pred, labels=labels, average=None, zero_division=0)),
                strict=True,
            )
        ),
        "confusion": confusion_matrix(y, pred, labels=labels).tolist(),
    }
    try:
        out["auroc_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr", labels=labels))
    except ValueError:
        out["auroc_ovr"] = float("nan")  # a class is missing from y
    return out


def stream_metrics(
    y: FloatArray, proba: FloatArray, classes: Sequence[str], threshold: float = 0.5
) -> dict[str, Any]:
    """Multi-label: per-code F1/AUROC; macro-F1 over codes that occur in y."""
    pred = proba >= threshold
    per_f1, per_auc = {}, {}
    for i, c in enumerate(classes):
        yt = y[:, i] > 0.5
        per_f1[c] = float(f1_score(yt, pred[:, i], zero_division=0))
        per_auc[c] = (
            float(roc_auc_score(yt, proba[:, i])) if 0 < yt.sum() < len(yt) else float("nan")
        )
    present = [c for i, c in enumerate(classes) if (y[:, i] > 0.5).any()]
    return {
        "macro_f1": float(np.mean([per_f1[c] for c in present])) if present else float("nan"),
        "per_class_f1": per_f1,
        "per_class_auroc": per_auc,
        "mean_auroc": float(np.nanmean(list(per_auc.values()))) if per_auc else float("nan"),
        "positives": {c: int((y[:, i] > 0.5).sum()) for i, c in enumerate(classes)},
    }


def bootstrap_ci(
    groups: NDArray[Any],
    metric: Callable[[NDArray[np.int64]], float],
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI of `metric(row_indices)` resampling *subjects* with replacement."""
    rng = np.random.default_rng(seed)
    subjects = np.unique(groups)
    rows = {s: np.flatnonzero(groups == s) for s in subjects}
    vals = []
    for _ in range(n):
        pick = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([rows[s] for s in pick])
        v = metric(idx)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bland_altman(measured: FloatArray, truth: FloatArray) -> dict[str, float]:
    """Bias and 95 % limits of agreement of measured - truth; plus MAE and RMSE."""
    d = np.asarray(measured, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    if d.size == 0:
        return {k: float("nan") for k in ("n", "bias", "loa_low", "loa_high", "mae", "rmse")}
    sd = float(d.std(ddof=1)) if d.size > 1 else 0.0
    bias = float(d.mean())
    return {
        "n": float(d.size),
        "bias": bias,
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
        "mae": float(np.abs(d).mean()),
        "rmse": float(np.sqrt((d**2).mean())),
    }
