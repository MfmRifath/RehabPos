"""IntelliRehabDS (Zenodo 10.5281/zenodo.4610859; CC BY 4.0): Kinect skeleton sequences.

ASSUMED layout (unverified): one text file per sequence named
  <subject>_<date>_<gesture>_<repetition>_<correctness>_<position>.txt
containing Kinect v2 25-joint rows (x, y, z[, state]). Gesture codes are kept raw;
map them in configs/datasets/intellirehab.yaml (exercise_map) after checking the README.
"""

from __future__ import annotations

from pathlib import Path

from rehabpose.data.joint_maps import KINECT_V2
from rehabpose.data.loaders.base import DatasetInfo, PatternLoader, Sequence, read_skeleton_rows


class IntelliRehabLoader(PatternLoader):
    """Kinect v2 sequences with correctness labels (per the spec)."""

    info = DatasetInfo(
        name="intellirehab",
        title="IntelliRehabDS",
        license="CC BY 4.0",
        citation="IntelliRehabDS, Zenodo doi:10.5281/zenodo.4610859",
        url="https://doi.org/10.5281/zenodo.4610859",
        access="open",
        instructions=(
            "Download from Zenodo (doi:10.5281/zenodo.4610859; scripts/download_zenodo.sh, only "
            "after approval) and extract under data/raw/intellirehab/."
        ),
    )
    patterns = ("**/*.txt", "**/*.csv")
    regex = (
        r"(?P<subject>\d+)_(?P<date>\d+)_(?P<exercise>\d+)_(?P<repetition>\d+)_"
        r"(?P<correctness>\d+)_(?P<position>[A-Za-z0-9]+)\.(?:txt|csv)$"
    )

    def read(self, path: Path, groups: dict[str, str]) -> Sequence:
        """Parse the Kinect 25-joint rows."""
        skel = read_skeleton_rows(path, len(KINECT_V2.joints))
        return self.base_sequence(path, groups, joint_format=KINECT_V2.name, skeleton=skel)
