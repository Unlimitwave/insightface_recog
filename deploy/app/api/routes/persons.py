"""人员注册与人脸录入 API 路由。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, UploadFile

from ...schemas.api import EnrollResponse, PersonCreateRequest, PersonListResponse, PersonResponse
from ..deps import get_face_service, get_request_id, get_settings, resolve_skip_liveness, verify_api_key
from ...config import Settings
from ...services.face_service import FaceService

# 创建路由模块：所有路径自动加上 /persons 前缀，
# 且整个路由下的接口都需要通过 API Key 验证（verify_api_key）
router = APIRouter(prefix="/persons", tags=["persons"], dependencies=[Depends(verify_api_key)])


@router.post("", response_model=PersonResponse, status_code=201)
def create_person(
    body: PersonCreateRequest,           # 请求体：包含 display_name, metadata 等
    service: Annotated[FaceService, Depends(get_face_service)],  # 依赖注入：获取人脸服务单例
) -> PersonResponse:
    """创建人员。POST /v1/persons  返回 201 表示已创建。"""
    return service.create_person(body)


@router.get("", response_model=PersonListResponse)
def list_persons(
    service: Annotated[FaceService, Depends(get_face_service)],
    offset: int = Query(0, ge=0),           # 分页偏移量，>=0，默认从 0 开始
    limit: int = Query(100, ge=1, le=500),  # 每页条数，1~500 之间，默认 100
) -> PersonListResponse:
    """分页查询人员列表。GET /v1/persons?offset=0&limit=100"""
    return service.list_persons(offset=offset, limit=limit)


@router.get("/{person_id}", response_model=PersonResponse)
def get_person(
    person_id: str,                                          # URL 路径参数：人员 ID
    service: Annotated[FaceService, Depends(get_face_service)],
) -> PersonResponse:
    """查询单个人员详情。GET /v1/persons/{person_id}"""
    return service.get_person(person_id)


@router.delete("/{person_id}", status_code=204)
def delete_person(
    person_id: str,
    service: Annotated[FaceService, Depends(get_face_service)],
) -> None:
    """删除人员及其所有关联人脸。DELETE /v1/persons/{person_id}  返回 204 无内容。"""
    service.delete_person(person_id)


@router.post("/{person_id}/faces", response_model=EnrollResponse)
async def enroll_faces(
    person_id: str,
    service: Annotated[FaceService, Depends(get_face_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    request_id: Annotated[str, Depends(get_request_id)],  # 自动注入请求 ID，用于日志追踪
    images: list[UploadFile] = File(..., description="One or more face photos (JPEG/PNG)"),
    skip_liveness: bool = Query(
        False,
        description="Skip liveness on enrollment (development only; forbidden in production)",
    ),
) -> EnrollResponse:
    """为人脸录入人脸照片。POST /v1/persons/{person_id}/faces

    接收 multipart/form-data 上传的图片列表，对每张图片：
    1. 人脸检测 + 质量筛选
    2. 活体检测（可通过 skip_liveness=true 跳过）
    3. 提取 512 维特征向量
    4. 存入底库（SQLite + FAISS 索引）
    """
    # 将 FastAPI 的 UploadFile 对象转为 (文件名, 文件流) 元组列表
    files = [(f.filename or "upload", f.file) for f in images]
    return service.enroll_faces(
        person_id,
        files,
        request_id,
        skip_liveness=resolve_skip_liveness(settings, skip_liveness),
    )


@router.delete("/{person_id}/faces/{face_id}", status_code=204)
def delete_face(
    person_id: str,
    face_id: str,
    service: Annotated[FaceService, Depends(get_face_service)],
) -> None:
    """删除某人员下的某张人脸。DELETE /v1/persons/{person_id}/faces/{face_id}"""
    from ...core.errors import AppError, ErrorCode

    try:
        service.gallery.delete_face(person_id, face_id)
    except KeyError as exc:
        # 将 Python 原生 KeyError 转换为业务级别的 AppError
        raise AppError(
            ErrorCode.FACE_NOT_FOUND,
            f"Face not found: {face_id}",
            status_code=404,
        ) from exc
