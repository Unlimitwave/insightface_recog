"""Pydantic request/response schemas (OpenAPI)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


class LivenessInfo(BaseModel):
    passed: bool
    score: float
    method: str = "rgb_passive_minifasnet"
    model_scores: dict[str, float] = Field(default_factory=dict)


class FaceQualityInfo(BaseModel):
    det_score: float
    bbox: list[float]
    face_width_px: float
    face_height_px: float


class PersonCreateRequest(BaseModel):
    person_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\-]+$")
    display_name: str = Field(..., min_length=1, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PersonResponse(BaseModel):
    person_id: str
    display_name: str
    metadata: dict[str, Any]
    face_count: int
    created_at: datetime
    updated_at: datetime


class PersonListResponse(BaseModel):
    total: int
    items: list[PersonResponse]


class EnrolledFaceResponse(BaseModel):
    face_id: str
    person_id: str
    created_at: datetime
    quality: FaceQualityInfo
    liveness: LivenessInfo | None = None


class EnrollResponse(BaseModel):
    request_id: str
    person_id: str
    enrolled: list[EnrolledFaceResponse]
    total_faces: int


class IdentifyCandidate(BaseModel):
    rank: int
    person_id: str
    display_name: str
    similarity: float
    matched: bool


class IdentifyResponse(BaseModel):
    request_id: str
    matched: bool
    person_id: str | None = None
    display_name: str | None = None
    similarity: float | None = None
    threshold: float
    is_stranger: bool = Field(
        description="True when face detected but no gallery match above threshold",
    )
    alert: bool = Field(
        description="True when stranger_alert_enabled and is_stranger (intrusion alert)",
    )
    candidates: list[IdentifyCandidate]
    quality: FaceQualityInfo
    liveness: LivenessInfo | None = None
    latency_ms: dict[str, float] = Field(default_factory=dict)


class VerifyResponse(BaseModel):
    request_id: str
    verified: bool
    person_id: str
    display_name: str
    similarity: float
    threshold: float
    matched_face_id: str | None = None
    quality: FaceQualityInfo
    liveness: LivenessInfo | None = None
    latency_ms: dict[str, float] = Field(default_factory=dict)


class EventResponse(BaseModel):
    event_id: str
    request_id: str
    event_type: str
    created_at: datetime
    matched: bool
    is_stranger: bool
    person_id: str | None = None
    display_name: str | None = None
    similarity: float | None = None
    threshold: float
    liveness_passed: bool | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventListResponse(BaseModel):
    total: int
    items: list[EventResponse]


class DailyStatsItem(BaseModel):
    date: str
    total: int
    matched: int
    strangers: int
    verify_pass: int


class PersonActivityItem(BaseModel):
    person_id: str
    display_name: str
    event_count: int
    last_seen_at: datetime | None = None


class GalleryStats(BaseModel):
    person_count: int
    face_count: int


class EventStats(BaseModel):
    total_events: int
    identify_count: int
    verify_count: int
    matched_count: int
    stranger_count: int
    pass_rate: float = Field(description="identify matched / identify total in period")
    daily: list[DailyStatsItem]
    top_persons: list[PersonActivityItem]


class StatsResponse(BaseModel):
    period_days: int
    gallery: GalleryStats
    events: EventStats


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    gallery_size: int
    person_count: int
    liveness_enabled: bool
    liveness_models_loaded: bool


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, bool]
