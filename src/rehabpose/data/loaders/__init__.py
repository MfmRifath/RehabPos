"""Dataset loaders (never download data; see docs/05_data_plan.md)."""

from __future__ import annotations

from pathlib import Path

from rehabpose.data.loaders.base import (
    DatasetError,
    DatasetInfo,
    DatasetLoader,
    DatasetMissingError,
    LayoutError,
    LicenseNotAcceptedError,
    Sequence,
)
from rehabpose.data.loaders.dib_upper import DibUpperLoader
from rehabpose.data.loaders.intellirehab import IntelliRehabLoader
from rehabpose.data.loaders.kimore import KimoreLoader
from rehabpose.data.loaders.own import OwnLoader
from rehabpose.data.loaders.rehab24_6 import Rehab246Loader
from rehabpose.data.loaders.trsp import TrspLoader
from rehabpose.data.loaders.uco import UcoLoader
from rehabpose.data.loaders.uiprmd import UiPrmdLoader

REGISTRY: dict[str, type[DatasetLoader]] = {
    cls.info.name: cls
    for cls in (
        OwnLoader,
        UcoLoader,
        Rehab246Loader,
        DibUpperLoader,
        IntelliRehabLoader,
        UiPrmdLoader,
        KimoreLoader,
        TrspLoader,
    )
}


def get_loader(
    name: str, data_dir: Path = Path("data"), accept_license: bool = False
) -> DatasetLoader:
    """Instantiate a registered loader reading from <data_dir>/raw/<name>/."""
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(REGISTRY)}")
    if name == "own":
        return OwnLoader(data_dir, accept_license)
    return REGISTRY[name](Path(data_dir) / "raw", accept_license)


__all__ = [
    "REGISTRY",
    "DatasetError",
    "DatasetInfo",
    "DatasetLoader",
    "DatasetMissingError",
    "LayoutError",
    "LicenseNotAcceptedError",
    "Sequence",
    "get_loader",
]
