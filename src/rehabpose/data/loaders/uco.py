"""UCO Physical Rehabilitation (Sensors 2023) and UCOPhyRehab++ (Scientific Data 2026).

ASSUMED layout (unverified): data/raw/uco/<subject>/<exercise>/<video>. Override in
configs/datasets/uco.yaml once the real layout is known. Videos are pose-extracted by
`rehabpose data extract-all uco`.
"""

from __future__ import annotations

from rehabpose.data.loaders.base import DatasetInfo, VideoLoader


class UcoLoader(VideoLoader):
    """UCO RGB videos (5 views, 27 subjects, 8 exercises per the Sensors 2023 paper)."""

    info = DatasetInfo(
        name="uco",
        title="UCO Physical Rehabilitation / UCOPhyRehab++",
        license="UCO videos: on request, authors' terms. UCOPhyRehab++ on Zenodo: CC BY 4.0",
        citation=(
            "Aguilar-Ortega et al., Sensors 23(21):8862 (2023), doi:10.3390/s23218862; "
            "Aguilar-Ortega et al., Scientific Data 13:1035 (2026), "
            "doi:10.1038/s41597-026-07362-5"
        ),
        url="https://github.com/AVAuco/ucophyrehab",
        access="on_request",
        instructions=(
            "Request access via github.com/AVAuco/ucophyrehab (UCOPhyRehab++ is on Zenodo), "
            "then place the files under data/raw/uco/."
        ),
    )
    regex = r"(?P<subject>[^/]+)/(?P<exercise>[^/]+)/[^/]+\.(?:mp4|avi|mkv|mov)$"
