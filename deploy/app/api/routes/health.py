"""Health and readiness probes (Kubernetes-style)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from ...config import Settings, get_settings
from ...schemas.api import HealthResponse, ReadyResponse
from ..deps import get_engine, get_gallery

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    settings: Annotated[Settings, Depends(get_settings)],
    engine=Depends(get_engine),
    gallery=Depends(get_gallery),
) -> HealthResponse:
    liveness_loaded = False
    if hasattr(engine, "liveness_engine") and engine.liveness_engine:
        liveness_loaded = engine.liveness_engine.available

    return HealthResponse(
        status="ok",
        version=settings.app_version,
        device=engine.device_label,
        gallery_size=gallery.total_faces,
        person_count=gallery.person_count(),
        liveness_enabled=settings.liveness_enabled,
        liveness_models_loaded=liveness_loaded,
    )


@router.get("/ready", response_model=ReadyResponse)
def ready(
    settings: Annotated[Settings, Depends(get_settings)],
    engine=Depends(get_engine),
    gallery=Depends(get_gallery),
) -> ReadyResponse:
    liveness_ok = True
    if settings.liveness_enabled and (
        settings.liveness_on_enroll or settings.liveness_on_identify
    ):
        liveness_ok = (
            hasattr(engine, "liveness_engine")
            and engine.liveness_engine is not None
            and engine.liveness_engine.available
        )

    checks = {
        "engine": engine is not None,
        "gallery": gallery is not None,
        "liveness_models": liveness_ok or not settings.liveness_enabled,
    }
    return ReadyResponse(ready=all(checks.values()), checks=checks)
