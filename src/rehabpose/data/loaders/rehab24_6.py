"""REHAB24-6 (Zenodo record 13305826): RGB videos + mocap, segmented reps, correctness.

ASSUMED layout (unverified): data/raw/rehab24_6/<subject>/<exercise>/<video>. Rep
segmentation/correctness files are not parsed until the real format is known.
"""

from __future__ import annotations

from rehabpose.data.loaders.base import DatasetInfo, VideoLoader


class Rehab246Loader(VideoLoader):
    """REHAB24-6 RGB videos (10 subjects, 6 exercises per the spec)."""

    info = DatasetInfo(
        name="rehab24_6",
        title="REHAB24-6",
        license="CC BY-NC 4.0: non-commercial research only (Zenodo record 13305826)",
        citation="Cernek, Sedmidubsky, Budikova, SISAP 2024, doi:10.1007/978-3-031-75823-2_2",
        url="https://zenodo.org/records/13305826",
        access="open",
        instructions=(
            "Download from Zenodo record 13305826 (scripts/download_zenodo.sh 13305826 rehab24_6, "
            "only after approval) and extract under data/raw/rehab24_6/."
        ),
    )
    regex = r"(?P<subject>[^/]+)/(?P<exercise>[^/]+)/[^/]+\.(?:mp4|avi|mkv|mov)$"
