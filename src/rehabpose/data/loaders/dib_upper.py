"""Upper-limb stroke rehab exercise videos (Data in Brief 2026; Mendeley 10.17632/49h9dcwx5v.1).

ASSUMED layout (unverified): data/raw/dib_upper/<exercise>/<complete|incomplete>/<video>.
Subject ids are unknown under this layout, so sequences get subject "unknown:<stem>" and
subject-wise splitting is NOT possible until real subject ids are parsed.
"""

from __future__ import annotations

from pathlib import Path

from rehabpose.data.loaders.base import DatasetInfo, Sequence, VideoLoader


class DibUpperLoader(VideoLoader):
    """491 webcam/mobile videos, 4 exercises, complete/incomplete (per the spec)."""

    info = DatasetInfo(
        name="dib_upper",
        title="Upper-limb stroke rehabilitation exercise video dataset (Data in Brief 2026)",
        license="CC BY 4.0",
        citation="Nandana et al., Data in Brief 66:112819 (2026), doi:10.1016/j.dib.2026.112819",
        url="https://data.mendeley.com/datasets/49h9dcwx5v/1",
        access="open",
        instructions=(
            "Download from Mendeley Data (doi:10.17632/49h9dcwx5v.1) and extract under "
            "data/raw/dib_upper/ (see scripts/download_mendeley.sh)."
        ),
    )
    regex = r"(?P<exercise>[^/]+)/(?P<completion>complete|incomplete)/[^/]+\.(?:mp4|avi|mkv|mov)$"

    def read(self, path: Path, groups: dict[str, str]) -> Sequence:
        """Video sequence with a binary complete/incomplete label."""
        seq = super().read(path, groups)
        seq.labels["complete"] = groups["completion"].lower() == "complete"
        return seq
