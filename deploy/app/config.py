"""Application configuration (12-factor, env-driven)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "Face Access Control API"
    app_version: str = "1.0.0"
    api_prefix: str = "/v1"
    debug: bool = False

    # deployment: development allows skip_liveness; production forbids it
    environment: Literal["development", "production"] = Field(
        default="development",
        alias="ENVIRONMENT",
    )

    # Security
    api_key: str | None = Field(default=None, description="Optional X-API-Key; unset = no auth")
    api_key_header: str = "X-API-Key"

    # Models (relative to cwd: deploy/ locally, /service in container)
    det_model_dir: str = Field(default="./models/detection", alias="DET_MODEL_DIR")
    recog_model_dir: str = Field(default="./models/recog", alias="RECOG_MODEL_DIR")
    antispoof_dir: str = Field(default="./models/antispoof", alias="ANTISPOOF_MODEL_DIR")
    # Optional explicit ONNX filename (or absolute path); unset = auto-pick from model dir
    det_model_name: str | None = Field(default=None, alias="DET_MODEL_NAME")
    recog_model_name: str | None = Field(
        default="glint360k_r100.onnx",
        alias="RECOG_MODEL_NAME",
        description="Default recognition model when multiple .onnx files exist",
    )
    det_thresh: float = 0.5
    det_size: int = 640

    # Inference device: auto | cuda | cpu
    device: Literal["auto", "cuda", "cpu"] = "auto"

    # 1:N gallery
    data_dir: str = Field(default="./data", alias="DATA_DIR")
    identify_threshold: float = Field(default=0.42, alias="IDENTIFY_THRESHOLD")
    identify_top_k: int = Field(default=5, alias="IDENTIFY_TOP_K")
    max_faces_per_person: int = Field(default=5, alias="MAX_FACES_PER_PERSON")

    # 1:1 verification
    verify_threshold: float = Field(default=0.42, alias="VERIFY_THRESHOLD")

    # Stranger detection / intrusion alert
    stranger_alert_enabled: bool = Field(default=True, alias="STRANGER_ALERT_ENABLED")

    # Event audit log
    event_log_enabled: bool = Field(default=True, alias="EVENT_LOG_ENABLED")
    event_log_retention_days: int = Field(default=90, alias="EVENT_LOG_RETENTION_DAYS")

    # Quality gates (access-control defaults)
    min_det_score: float = 0.5
    min_face_size_px: int = 80

    # Liveness (passive RGB, MiniFASNet ensemble)
    liveness_enabled: bool = Field(default=True, alias="LIVENESS_ENABLED")
    liveness_on_enroll: bool = Field(default=True, alias="LIVENESS_ON_ENROLL")
    liveness_on_identify: bool = Field(default=True, alias="LIVENESS_ON_IDENTIFY")
    liveness_on_verify: bool = Field(default=True, alias="LIVENESS_ON_VERIFY")
    liveness_threshold: float = Field(default=0.5, alias="LIVENESS_THRESHOLD")
    liveness_require_both_models: bool = Field(
        default=True,
        alias="LIVENESS_REQUIRE_BOTH",
        description="Both MiniFASNet V1SE and V2 must pass when available",
    )

    # Multi-face policy
    multi_face_policy: Literal["largest", "reject"] = "largest"

    # Server
    host: str = "0.0.0.0"
    port: int = Field(default=8123, alias="PORT")
    workers: int = 1

    @property
    def allow_skip_liveness(self) -> bool:
        """Production deployments must not bypass liveness checks."""
        return self.environment != "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
