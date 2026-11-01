"""UI-PRMD (ODC PDDL): Vicon + Kinect segmented movements, correct/incorrect.

ASSUMED layout (unverified):
  data/raw/uiprmd/<Segmented Movements|Incorrect Segmented Movements>/Kinect/Positions/
      m<movement>_s<subject>_e<episode>_positions[_inc].txt
with comma-separated joint positions per row. The Kinect joint count/order is NOT assumed:
sequences are returned with joint_format "unknown_<J>" until a verified joint map exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rehabpose.data.loaders.base import (
    FLOAT_RE,
    DatasetInfo,
    LayoutError,
    PatternLoader,
    Sequence,
)


class UiPrmdLoader(PatternLoader):
    """UI-PRMD Kinect positions (includes shoulder movements per the spec)."""

    info = DatasetInfo(
        name="uiprmd",
        title="UI-PRMD",
        license="ODC PDDL (verify)",
        citation="UI-PRMD dataset (use the citation given on the dataset page)",
        url="https://zenodo.org (search: UI-PRMD)",
        access="open",
        instructions=(
            "Download UI-PRMD from its Zenodo/web page (record id to be confirmed; "
            "scripts/download_zenodo.sh <record-id> uiprmd after approval) and extract under "
            "data/raw/uiprmd/."
        ),
    )
    patterns = ("**/*.txt",)
    regex = (
        r"(?P<split>Incorrect Segmented Movements|Segmented Movements)/Kinect/Positions/"
        r"m(?P<exercise>\d+)_s(?P<subject>\d+)_e(?P<episode>\d+)_positions(?:_inc)?\.txt$"
    )

    def read(self, path: Path, groups: dict[str, str]) -> Sequence:
        """Parse rows of J*3 numbers; joint order unknown until verified."""
        rows = [
            [float(v) for v in FLOAT_RE.findall(line)]
            for line in path.read_text(errors="replace").splitlines()
        ]
        rows = [r for r in rows if r]
        widths = {len(r) for r in rows}
        if not rows or len(widths) != 1 or next(iter(widths)) % 3:
            raise LayoutError(f"{path}: expected constant rows of J*3 numbers, got widths {widths}")
        skel = np.asarray(rows, dtype=np.float64).reshape(len(rows), -1, 3)
        seq = self.base_sequence(
            path, groups, joint_format=f"unknown_{skel.shape[1]}", skeleton=skel
        )
        seq.labels["correct"] = not groups["split"].lower().startswith("incorrect")
        return seq
