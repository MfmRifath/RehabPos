"""Subject-wise splits: held-out test subjects + GroupKFold (or LOSO for few subjects).

The unit of splitting is the subject; no subject may appear in more than one partition.
Subjects named "unknown:*" (dataset without subject ids) are rejected because they would
make leakage undetectable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from numpy.typing import NDArray

IntArray = NDArray[np.int64]


class LeakageError(ValueError):
    """A subject appears in more than one partition."""


@dataclass(frozen=True)
class Fold:
    """One cross-validation fold (subject ids)."""

    index: int
    train: tuple[str, ...]
    val: tuple[str, ...]


@dataclass(frozen=True)
class SplitPlan:
    """Held-out test subjects plus CV folds over the remaining subjects."""

    method: Literal["group_kfold", "loso"]
    seed: int
    test: tuple[str, ...]
    folds: tuple[Fold, ...]

    @property
    def subjects(self) -> tuple[str, ...]:
        """All subjects covered by the plan."""
        return tuple(sorted({*self.test, *self.folds[0].train, *self.folds[0].val}))


def make_split_plan(
    subjects: Iterable[str],
    test_fraction: float = 0.2,
    k: int = 5,
    seed: int = 0,
    loso_below: int = 10,
) -> SplitPlan:
    """Plan splits over unique subjects (deterministic for a given seed).

    ~test_fraction of subjects (>= 1 when there are >= 3 subjects) are held out; the rest
    form GroupKFold-style folds of near-equal size, or LOSO when fewer than `loso_below`
    remain.
    """
    uniq = sorted(set(subjects))
    unknown = [s for s in uniq if s.startswith("unknown:")]
    if unknown:
        raise ValueError(f"{len(unknown)} sequences lack subject ids; cannot split by subject")
    if len(uniq) < 2:
        raise ValueError("need at least 2 subjects")
    rng = np.random.default_rng(seed)
    order = [uniq[i] for i in rng.permutation(len(uniq))]
    n_test = max(1, round(test_fraction * len(uniq))) if len(uniq) >= 3 else 0
    test, rest = tuple(sorted(order[:n_test])), order[n_test:]

    method: Literal["group_kfold", "loso"]
    if len(rest) < loso_below:
        method = "loso"
        groups = [[s] for s in rest]
    else:
        method = "group_kfold"
        n = min(k, len(rest))
        groups = [rest[i::n] for i in range(n)]
    folds = tuple(
        Fold(
            index=i,
            train=tuple(sorted(s for j, g in enumerate(groups) if j != i for s in g)),
            val=tuple(sorted(g)),
        )
        for i, g in enumerate(groups)
    )
    plan = SplitPlan(method, seed, test, folds)
    assert_no_leakage(plan)
    return plan


def assert_no_leakage(plan: SplitPlan) -> None:
    """Raise LeakageError if any subject is shared between partitions of any fold."""
    test = set(plan.test)
    for f in plan.folds:
        train, val = set(f.train), set(f.val)
        pairs = ((train, val, "train/val"), (train, test, "train/test"), (val, test, "val/test"))
        for a, b, name in pairs:
            shared = a & b
            if shared:
                raise LeakageError(f"fold {f.index}: {name} share subjects {sorted(shared)}")


def fold_indices(
    groups: Sequence[str], plan: SplitPlan, fold: int
) -> tuple[IntArray, IntArray, IntArray]:
    """Row indices (train, val, test) for per-row subject ids `groups` in fold `fold`."""
    f = plan.folds[fold]
    g = np.asarray(groups)
    idx = np.arange(len(g), dtype=np.int64)
    tr, va, te = np.isin(g, f.train), np.isin(g, f.val), np.isin(g, plan.test)
    if (tr & va).any() or (tr & te).any() or (va & te).any():
        raise LeakageError("row assigned to more than one partition")
    missing = sorted(set(g[~(tr | va | te)].tolist()))
    if missing:
        raise ValueError(f"subjects not in the plan: {missing}")
    return idx[tr], idx[va], idx[te]


def save_plan(plan: SplitPlan, path: Path) -> Path:
    """Write a plan as YAML (e.g. configs/train/splits.yaml)."""
    data = {
        "method": plan.method,
        "seed": plan.seed,
        "test": list(plan.test),
        "folds": [
            {"index": f.index, "train": list(f.train), "val": list(f.val)} for f in plan.folds
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def load_plan(path: Path) -> SplitPlan:
    """Read a YAML plan and re-check it for leakage."""
    data = yaml.safe_load(Path(path).read_text())
    plan = SplitPlan(
        method=data["method"],
        seed=int(data["seed"]),
        test=tuple(data["test"]),
        folds=tuple(Fold(f["index"], tuple(f["train"]), tuple(f["val"])) for f in data["folds"]),
    )
    assert_no_leakage(plan)
    return plan
