"""1:N identification API (access control probe)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, UploadFile

from ...schemas.api import IdentifyResponse
from ..deps import get_face_service, get_request_id, get_settings, resolve_skip_liveness, verify_api_key
from ...config import Settings
from ...services.face_service import FaceService

router = APIRouter(tags=["identify"], dependencies=[Depends(verify_api_key)])


@router.post("/identify", response_model=IdentifyResponse)
async def identify(
    service: Annotated[FaceService, Depends(get_face_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    request_id: Annotated[str, Depends(get_request_id)],
    image: UploadFile = File(..., description="Probe face image from camera"),
    skip_liveness: bool = Query(
        False,
        description="Skip liveness (development only; forbidden in production)",
    ),
) -> IdentifyResponse:
    data = await image.read()
    return service.identify(
        data,
        request_id,
        skip_liveness=resolve_skip_liveness(settings, skip_liveness),
    )
