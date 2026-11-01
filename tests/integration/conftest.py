"""App fixture: temporary DB/data dir, real exercise configs, mocked pose estimator."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from rehabpose.api.main import create_app
from rehabpose.core.landmarks import Frame
from rehabpose.settings import Settings

ROOT = Path(__file__).resolve().parents[2]


class TruthEstimator:
    """Returns pre-registered ground-truth frames by timestamp (stands in for MediaPipe)."""

    truth: dict[int, Frame] = {}

    def process(self, frame_bgr: NDArray[np.uint8], ts_ms: int) -> Frame | None:
        return self.truth.get(ts_ms)

    def close(self) -> None:
        pass


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        db_url=f"sqlite:///{tmp_path / 'test.db'}",
        data_dir=tmp_path / "data",
        exercises_dir=ROOT / "configs" / "exercises",
        frontend_dist=tmp_path / "no-frontend",
        models_dir=tmp_path / "models",
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    TruthEstimator.truth = {}
    app = create_app(settings, estimator_factory=TruthEstimator)
    with TestClient(app) as c:
        yield c


def new_session(
    client: TestClient, exercise: str = "shoulder_flexion", side: str = "R", **kw: str
) -> str:
    r = client.post(
        "/api/sessions", json={"user_id": "u1", "exercise_id": exercise, "side": side, **kw}
    )
    assert r.status_code == 201, r.text
    sid: str = r.json()["id"]
    return sid
