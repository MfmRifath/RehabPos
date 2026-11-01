"""LightGBM baseline on per-rep summary features.

LightGBM links Homebrew's libomp while torch bundles its own; with torch imported, a
multi-threaded LightGBM fit segfaults on macOS (checked 2026-11-01). n_jobs=1 avoids it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def summary_features(x: NDArray[Any]) -> FloatArray:
    """(N, T, F) -> (N, 5F): mean, std, min, max and last-first over time."""
    a = np.asarray(x, dtype=np.float64)
    return np.concatenate([a.mean(1), a.std(1), a.min(1), a.max(1), a[:, -1] - a[:, 0]], axis=1)


def make_lgbm(seed: int, n_classes: int, **kw: Any) -> Any:
    """LGBMClassifier with safe threading (n_jobs=1)."""
    import lightgbm as lgb

    params: dict[str, Any] = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 5,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": 1,
        "verbose": -1,
    }
    params.update(kw)
    if n_classes > 2:
        params["objective"] = "multiclass"
    return lgb.LGBMClassifier(**params)
