"""KIMORE (Kinect v2; 78 subjects; 5 low-back exercises; clinical scores; check the EULA).

ASSUMED layout (unverified):
  data/raw/kimore/<group>/[<subgroup>/]<subject>/Es<k>/Raw/JointPosition*.csv
with Kinect v2 25 joints x (x, y, z, tracking state). Low-back exercises have no RehabPose
exercise id; KIMORE is used for pretraining only. Clinical scores are not parsed yet.
"""

from __future__ import annotations

from pathlib import Path

from rehabpose.data.joint_maps import KINECT_V2
from rehabpose.data.loaders.base import DatasetInfo, PatternLoader, Sequence, read_skeleton_rows


class KimoreLoader(PatternLoader):
    """KIMORE Kinect v2 joint positions."""

    info = DatasetInfo(
        name="kimore",
        title="KIMORE",
        license="Per the spec: EULA, not confirmed (check before use; never redistribute)",
        citation=(
            "Capecci et al., IEEE TNSRE 27(7):1436-1448 (2019), doi:10.1109/TNSRE.2019.2923060"
        ),
        url="(provided with the EULA by the dataset authors)",
        access="eula",
        instructions="Obtain KIMORE under its EULA and place it under data/raw/kimore/.",
    )
    patterns = ("**/JointPosition*.csv",)
    regex = (
        r"(?P<group>[^/]+)/(?:(?P<subgroup>[^/]+)/)?(?P<subject>[^/]+)/Es(?P<exercise>\d)/Raw/"
        r"JointPosition[^/]*\.csv$"
    )

    def read(self, path: Path, groups: dict[str, str]) -> Sequence:
        """Parse Kinect 25-joint rows."""
        skel = read_skeleton_rows(path, len(KINECT_V2.joints))
        return self.base_sequence(path, groups, joint_format=KINECT_V2.name, skeleton=skel)
