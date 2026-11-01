"""Application settings: defaults < configs/app.yaml < REHABPOSE_* env vars < init kwargs."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

DEFAULT_CONFIG_PATH = Path("configs/app.yaml")
CONFIG_ENV_VAR = "REHABPOSE_CONFIG"
HOME_ENV_VAR = "REHABPOSE_HOME"

PoseModelVariant = Literal["lite", "full", "heavy"]


def project_root() -> Path:
    """Base for relative paths: $REHABPOSE_HOME, else the source checkout, else the cwd.

    Makes `rehabpose serve` behave the same from any working directory (e.g. frontend/).
    """
    env = os.environ.get(HOME_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / DEFAULT_CONFIG_PATH).is_file():
        return checkout
    return Path.cwd()


def config_path() -> Path:
    """Return the YAML config path, overridable via the REHABPOSE_CONFIG env var."""
    env = os.environ.get(CONFIG_ENV_VAR)
    return Path(env) if env else project_root() / DEFAULT_CONFIG_PATH


class Settings(BaseSettings):
    """Runtime configuration for the CLI and API server."""

    model_config = SettingsConfigDict(
        env_prefix="REHABPOSE_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    data_dir: Path = Path("data")
    models_dir: Path = Path("data/models")
    exercises_dir: Path = Path("configs/exercises")
    db_url: str = "sqlite:///data/rehabpose.db"
    pose_model: PoseModelVariant = "full"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    frontend_dist: Path = Path("frontend/dist")
    max_upload_mb: int = Field(default=500, gt=0)
    session_retention_days: int | None = Field(default=None, ge=0)  # None: keep until deleted

    @model_validator(mode="after")
    def _absolute_paths(self) -> Settings:
        """Resolve relative paths (and a relative sqlite file) against project_root()."""
        root = project_root()
        for name in ("data_dir", "models_dir", "exercises_dir", "frontend_dist"):
            value: Path = getattr(self, name)
            if not value.is_absolute():
                setattr(self, name, root / value)
        prefix = "sqlite:///"
        if self.db_url.startswith(prefix) and not self.db_url.startswith(prefix + "/"):
            rel = self.db_url[len(prefix) :]
            if rel and rel != ":memory:":
                self.db_url = f"{prefix}{root / rel}"
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_settings = YamlConfigSettingsSource(settings_cls, yaml_file=config_path())
        return (init_settings, env_settings, yaml_settings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached Settings instance."""
    return Settings()


def torch_device() -> str:
    """Return the preferred torch device: "mps" on Apple Silicon if available, else "cpu"."""
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"
