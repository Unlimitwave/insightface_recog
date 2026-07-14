"""1:1 face verification API (probe vs enrolled person templates)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, UploadFile

from ...schemas.api import VerifyResponse
from ..deps import get_face_service, get_request_id, get_settings, resolve_skip_liveness, verify_api_key
from ...config import Settings
from ...services.face_service import FaceService

router = APIRouter(tags=["verify"], dependencies=[Depends(verify_api_key)])


@router.post("/verify", response_model=VerifyResponse)
async def verify(
    service: Annotated[FaceService, Depends(get_face_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    request_id: Annotated[str, Depends(get_request_id)],
    person_id: str = Query(..., description="Expected person ID to verify against"),
    image: UploadFile = File(..., description="Probe face image"),
    skip_liveness: bool = Query(
        False,
        description="Skip liveness (development only; forbidden in production)",
    ),
) -> VerifyResponse:
    """1:1 验证：探针图与指定人员底库模板比对，返回 verified 与 similarity。"""
    data = await image.read()
    return service.verify(
        data,
        person_id,
        request_id,
        skip_liveness=resolve_skip_liveness(settings, skip_liveness),
    )
