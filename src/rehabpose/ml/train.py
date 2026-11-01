"""Training: config (configs/train/*.yaml), torch loop (AdamW, cosine, early stopping on val
macro-F1, class weights, per-batch augmentation) and the LightGBM baseline. Device mps -> cpu."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from rehabpose.core.features import n_features
from rehabpose.ml.datasets import DEFAULT_EXERCISES, WindowSet, featurize
from rehabpose.ml.metrics import rep_metrics, stream_metrics
from rehabpose.ml.models import TORCH_MODELS, build_model
from rehabpose.ml.models.lgbm import make_lgbm, summary_features
from rehabpose.ml.transforms import AugmentConfig

log = logging.getLogger(__name__)


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Cfg):
    """Data source. Only 'synthetic' is available until real data is approved."""

    source: Literal["synthetic", "own"] = "synthetic"
    n_subjects: int = Field(default=16, ge=4)
    sets_per_exercise: int = Field(default=2, ge=1)
    reps_per_set: int = Field(default=8, ge=2)
    exercises: list[str] = Field(default_factory=lambda: list(DEFAULT_EXERCISES))
    seed: int = 0


class StreamConfig(_Cfg):
    """Causal streaming windows."""

    length: int = Field(default=45, ge=8, le=256)
    stride: int = Field(default=5, ge=1)


class OptimConfig(_Cfg):
    """Optimiser and early stopping."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = Field(default=30, ge=1)
    batch_size: int = Field(default=64, ge=1)
    patience: int = Field(default=6, ge=1)


class CVConfig(_Cfg):
    """Subject-wise split plan."""

    test_fraction: float = Field(default=0.2, ge=0.0, lt=1.0)
    k: int = Field(default=5, ge=2)


class TrainConfig(_Cfg):
    """One experiment."""

    name: str
    task: Literal["rep", "stream"]
    model: str
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    space: Literal["2d", "3d"] = "2d"
    rep_length: int = Field(default=64, ge=8)
    data: DataConfig = Field(default_factory=DataConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)
    augment: dict[str, float] | None = Field(default_factory=dict)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    cv: CVConfig = Field(default_factory=CVConfig)
    threshold: float = Field(default=0.5, gt=0, lt=1)
    device: Literal["auto", "mps", "cpu"] = "auto"

    def augment_config(self) -> AugmentConfig | None:
        """Augmentation (None = off)."""
        return None if self.augment is None else AugmentConfig(**self.augment)

    @property
    def n_features(self) -> int:
        """Feature vector length."""
        return n_features(self.space)


def load_config(path: Path) -> TrainConfig:
    """Load and validate a training YAML."""
    cfg = TrainConfig.model_validate(yaml.safe_load(Path(path).read_text()))
    if cfg.model != "lgbm" and cfg.model not in TORCH_MODELS:
        raise ValueError(f"unknown model {cfg.model!r}")
    if cfg.model == "lgbm" and cfg.task != "rep":
        raise ValueError("lgbm supports the rep task only")
    return cfg


def resolve_device(pref: str) -> torch.device:
    """mps if available (auto/mps), else cpu. No CUDA paths."""
    if pref in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed torch and numpy (MPS shares the torch generator)."""
    torch.manual_seed(seed)
    np.random.seed(seed)


@dataclass
class FitResult:
    """A trained model with its early-stopping history."""

    model: Any
    kind: Literal["torch", "lgbm"]
    best_epoch: int
    best_val: float
    history: list[dict[str, float]] = field(default_factory=list)


def _metric(
    task: str, y: NDArray[Any], proba: NDArray[np.float64], classes: tuple[str, ...], thr: float
) -> float:
    m = rep_metrics(y, proba, classes) if task == "rep" else stream_metrics(y, proba, classes, thr)
    v = m["macro_f1"]
    return float(v) if np.isfinite(v) else 0.0


def _loss(task: str, y: NDArray[Any], n_out: int, device: torch.device) -> torch.nn.Module:
    if task == "rep":
        counts = np.bincount(y.astype(int), minlength=n_out).astype(np.float64)
        w = counts.sum() / np.maximum(counts, 1) / n_out
        return torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    pos = y.sum(0)
    pw = (len(y) - pos) / np.maximum(pos, 1)
    return torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(np.clip(pw, 1, 50), dtype=torch.float32, device=device)
    )


def predict_proba(
    fit: FitResult, ws: WindowSet, cfg: TrainConfig, device: torch.device, batch: int = 512
) -> NDArray[np.float64]:
    """Class probabilities (softmax) for rep, per-code probabilities (sigmoid) for stream."""
    x = featurize(ws, cfg.rep_length)
    if fit.kind == "lgbm":
        return np.asarray(fit.model.predict_proba(summary_features(x)), dtype=np.float64)
    model: torch.nn.Module = fit.model.to(device).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            logits = model(torch.from_numpy(x[i : i + batch]).to(device))
            p = torch.softmax(logits, 1) if cfg.task == "rep" else torch.sigmoid(logits)
            outs.append(p.float().cpu().numpy())
    return np.concatenate(outs).astype(np.float64) if outs else np.zeros((0, len(ws.classes)))


def fit(
    cfg: TrainConfig, train: WindowSet, val: WindowSet, seed: int, device: torch.device
) -> FitResult:
    """Train one model; early stopping keeps the epoch with the best validation macro-F1."""
    seed_everything(seed)
    classes = train.classes
    n_out = len(classes)
    if cfg.model == "lgbm":
        import lightgbm as lgb

        xtr, xva = (
            summary_features(featurize(train, cfg.rep_length)),
            summary_features(featurize(val, cfg.rep_length)),
        )
        clf = make_lgbm(seed, n_out)
        clf.fit(
            xtr, train.y, eval_set=[(xva, val.y)], callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        res = FitResult(clf, "lgbm", int(clf.best_iteration_ or clf.n_estimators), 0.0)
        res.best_val = _metric(
            cfg.task, val.y, predict_proba(res, val, cfg, device), classes, cfg.threshold
        )
        return res

    model = build_model(cfg.model, cfg.n_features, n_out, **cfg.model_kwargs).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.optim.epochs)
    loss_fn = _loss(cfg.task, train.y, n_out, device)
    aug = cfg.augment_config()
    rng = np.random.default_rng(seed)
    y_all = torch.as_tensor(train.y, dtype=torch.long if cfg.task == "rep" else torch.float32)
    best = FitResult(copy.deepcopy(model).cpu(), "torch", -1, -1.0)
    stale = 0
    for epoch in range(cfg.optim.epochs):
        model.train()
        order = rng.permutation(len(train))
        total = 0.0
        for i in range(0, len(order), cfg.optim.batch_size):
            idx = order[i : i + cfg.optim.batch_size]
            xb = torch.from_numpy(featurize(train, cfg.rep_length, rng, aug, idx)).to(device)
            yb = y_all[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(idx)
        sched.step()
        cur = FitResult(model, "torch", epoch, 0.0)
        val_f1 = _metric(
            cfg.task, val.y, predict_proba(cur, val, cfg, device), val.classes, cfg.threshold
        )
        best.history.append(
            {"epoch": epoch, "train_loss": total / max(len(train), 1), "val_macro_f1": val_f1}
        )
        if val_f1 > best.best_val:
            best.model, best.best_epoch, best.best_val = copy.deepcopy(model).cpu(), epoch, val_f1
            stale = 0
        else:
            stale += 1
            if stale >= cfg.optim.patience:
                break
    log.debug(
        "fit %s seed %d: best epoch %d val macro-F1 %.3f",
        cfg.model,
        seed,
        best.best_epoch,
        best.best_val,
    )
    return best
