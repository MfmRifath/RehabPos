"""Toronto Rehab Stroke Pose (Kaggle derekdb/toronto-robot-stroke-posture-dataset).

ASSUMED layout (unverified): data/raw/trsp/<subject>/<sequence>.csv with Kinect v2 25-joint
numeric columns (x, y, z[, state] per joint) and an optional per-frame text column
named "label" (none / lean-forward / shoulder-elevation / trunk-rotation per the spec).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rehabpose.data.joint_maps import KINECT_V2
from rehabpose.data.loaders.base import DatasetInfo, LayoutError, PatternLoader, Sequence

FRAME_LABEL_TO_CODE = {
    "none": None,
    "lean-forward": "TRUNK_LEAN",
    "shoulder-elevation": "SHOULDER_HIKE",
    "trunk-rotation": "TRUNK_ROTATION",
}


class TrspLoader(PatternLoader):
    """Kinect v2 frames with frame-level compensation labels."""

    info = DatasetInfo(
        name="trsp",
        title="Toronto Rehab Stroke Pose",
        license="Kaggle dataset terms (check before use)",
        citation="Toronto Rehab Stroke Pose dataset (use the citation on the Kaggle page)",
        url="https://www.kaggle.com/datasets/derekdb/toronto-robot-stroke-posture-dataset",
        access="kaggle",
        instructions=(
            "Download with the Kaggle CLI (needs your credentials; scripts/download_kaggle_trsp.sh "
            "after approval) into data/raw/trsp/."
        ),
    )
    patterns = ("**/*.csv",)
    regex = r"(?P<subject>[^/]+)/[^/]+\.csv$"

    def read(self, path: Path, groups: dict[str, str]) -> Sequence:
        """Parse numeric joint columns and the optional label column."""
        df = pd.read_csv(path)
        label_col = next((c for c in df.columns if str(c).strip().lower() == "label"), None)
        num = df.drop(columns=[label_col] if label_col else []).select_dtypes("number")
        j = len(KINECT_V2.joints)
        width = num.shape[1]
        if width not in (j * 3, j * 4, j * 3 + 1, j * 4 + 1):
            raise LayoutError(f"{path}: {width} numeric columns, expected 25 joints x 3 or 4")
        vals = num.to_numpy(dtype=np.float64)
        if width % j:
            vals = vals[:, 1:]  # leading index/timestamp column
        skel = vals.reshape(len(vals), j, -1)[:, :, :3]
        seq = self.base_sequence(path, groups, joint_format=KINECT_V2.name, skeleton=skel)
        if label_col:
            raw = [str(v).strip().lower() for v in df[label_col]]
            unknown = sorted(set(raw) - set(FRAME_LABEL_TO_CODE))
            if unknown:
                raise LayoutError(f"{path}: unknown frame labels {unknown}")
            seq.labels["frame_codes"] = [FRAME_LABEL_TO_CODE[v] for v in raw]
        return seq
