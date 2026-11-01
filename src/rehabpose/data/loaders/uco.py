"""UCO Physical Rehabilitation (Sensors 2023) and UCOPhyRehab++ (Scientific Data 2026).

ASSUMED layout (unverified): data/raw/uco/<subject>/<exercise>/<video>. Override in
configs/datasets/uco.yaml once the real layout is known. Videos are pose-extracted by
`rehabpose data extract-all uco`.
"""

from __future__ import annotations

from rehabpose.data.loaders.base import DatasetInfo, VideoLoader


class UcoLoader(VideoLoader):
    """UCO RGB videos (5 views, 27 subjects, 16 exercises per the spec)."""

    info = DatasetInfo(
        name="uco",
        title="UCO Physical Rehabilitation / UCOPhyRehab++",
        license="Available on request from the authors; terms set by them (verify)",
        citation=(
            "UCO Physical Rehabilitation dataset, Sensors (2023); UCOPhyRehab++, "
            "Scientific Data (2026). Use the full citations from the dataset pages."
        ),
        url="https://github.com/AVAuco/ucophyrehab",
        access="on_request",
        instructions=(
            "Request access via github.com/AVAuco/ucophyrehab (UCOPhyRehab++ is on Zenodo), "
            "then place the files under data/raw/uco/."
        ),
    )
    regex = r"(?P<subject>[^/]+)/(?P<exercise>[^/]+)/[^/]+\.(?:mp4|avi|mkv|mov)$"
