"""FastAPI dependencies and middleware helpers."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, Header, Request

from ..config import Settings, get_settings
from ..core.errors import AppError, ErrorCode
from ..engine import FaceEngine, InsightFaceEngine
from ..services.face_service import FaceService
from ..services.event_store import EventStore
from ..services.gallery import GalleryStore

_engine: FaceEngine | None = None
_gallery: GalleryStore | None = None
_event_store: EventStore | None = None
_face_service: FaceService | None = None


def init_services(settings: Settings) -> None:
    global _engine, _gallery, _event_store, _face_service
    _engine = InsightFaceEngine(settings)
    _gallery = GalleryStore(settings.data_dir, embedding_dim=_engine.embedding_dim)
    _event_store = (
        EventStore(settings.data_dir, retention_days=settings.event_log_retention_days)
        if settings.event_log_enabled
        else None
    )
    _face_service = FaceService(_engine, _gallery, settings, event_store=_event_store)


def get_engine() -> FaceEngine:
    assert _engine is not None, "Engine not initialized"
    return _engine


def get_gallery() -> GalleryStore:
    assert _gallery is not None, "Gallery not initialized"
    return _gallery


def get_face_service() -> FaceService:
    assert _face_service is not None, "FaceService not initialized"
    return _face_service


async def verify_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise AppError(
            ErrorCode.UNAUTHORIZED,
            "Invalid or missing API key",
            status_code=401,
        )


def resolve_skip_liveness(settings: Settings, skip_liveness: bool) -> bool:
    """Reject skip_liveness in production; allowed in development only."""
    if skip_liveness and not settings.allow_skip_liveness:
        raise AppError(
            ErrorCode.INVALID_REQUEST,
            "skip_liveness is disabled in production",
            status_code=403,
        )
    return skip_liveness


async def get_request_id(
    request: Request,
    x_request_id: Annotated[str | None, Header(alias="X-Request-ID")] = None,
) -> str:
    rid = x_request_id or request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = rid
    return rid
