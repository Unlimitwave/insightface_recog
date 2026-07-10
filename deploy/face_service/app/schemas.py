from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    device: str
    onnx_providers: List[str]
    gallery_size: int
    person_count: int


class EnrollResponse(BaseModel):
    person_id: str
    template_count: int
    message: str = "enrolled"


class PersonSummary(BaseModel):
    person_id: str
    display_name: Optional[str] = None
    template_count: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class PersonListResponse(BaseModel):
    total: int
    persons: List[PersonSummary]


class IdentifyCandidate(BaseModel):
    person_id: str
    display_name: Optional[str] = None
    similarity: float = Field(ge=-1.0, le=1.0)


class IdentifyResponse(BaseModel):
    matched: bool
    person_id: Optional[str] = None
    display_name: Optional[str] = None
    similarity: float = Field(ge=-1.0, le=1.0)
    threshold: float
    candidates: List[IdentifyCandidate]
    latency_ms: float


class DeletePersonResponse(BaseModel):
    person_id: str
    deleted_templates: int
    message: str = "deleted"


class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None


class EngineInfo(BaseModel):
    device: str
    providers: List[str]
    model_name: str
    embedding_dim: int
