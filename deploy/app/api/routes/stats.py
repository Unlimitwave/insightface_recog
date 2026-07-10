"""Gallery statistics and recognition event audit API."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query

from ...schemas.api import EventListResponse, StatsResponse
from ..deps import get_face_service, verify_api_key
from ...services.face_service import FaceService

router = APIRouter(tags=["analytics"], dependencies=[Depends(verify_api_key)])


@router.get("/stats", response_model=StatsResponse)
def stats(
    service: Annotated[FaceService, Depends(get_face_service)],
    days: int = Query(7, ge=1, le=365, description="Statistics window in days"),
) -> StatsResponse:
    """底库规模 + 识别/验证事件汇总（日趋势、通过率、人员活跃度）。"""
    return service.get_stats(days=days)


@router.get("/events", response_model=EventListResponse)
def list_events(
    service: Annotated[FaceService, Depends(get_face_service)],
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    event_type: Literal["identify", "verify"] | None = Query(
        None, description="Filter by event type"
    ),
    person_id: str | None = Query(None, description="Filter by matched person_id"),
    is_stranger: bool | None = Query(None, description="Filter stranger (intrusion) events"),
) -> EventListResponse:
    """识别/验证事件历史，支持分页与类型/人员/陌生人筛选。"""
    return service.list_events(
        offset=offset,
        limit=limit,
        event_type=event_type,
        person_id=person_id,
        is_stranger=is_stranger,
    )
