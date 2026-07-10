from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FACE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "face-service"
    app_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # Model
    model_name: str = "buffalo_l"
    model_root: str = "/models"
    det_thresh: float = 0.5
    det_size: int = 640
    preferred_device: Literal["auto", "cuda", "cpu"] = "auto"

    # Gallery / 1:N
    data_dir: str = "/data"
    similarity_threshold: float = Field(default=0.42, ge=-1.0, le=1.0)
    identify_top_k: int = Field(default=5, ge=1, le=100)
    max_templates_per_person: int = Field(default=5, ge=1, le=20)

    # Security (optional)
    api_key: Optional[str] = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
