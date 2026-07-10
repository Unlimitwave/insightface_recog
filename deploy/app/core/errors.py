"""Standard API error codes and HTTP exceptions."""

from __future__ import annotations

from enum import Enum
from typing import Any

from fastapi import HTTPException, status


class ErrorCode(str, Enum):
    INVALID_REQUEST = "INVALID_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    FACE_NOT_DETECTED = "FACE_NOT_DETECTED"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    LOW_FACE_QUALITY = "LOW_FACE_QUALITY"
    LIVENESS_FAILED = "LIVENESS_FAILED"
    LIVENESS_UNAVAILABLE = "LIVENESS_UNAVAILABLE"
    PERSON_NOT_FOUND = "PERSON_NOT_FOUND"
    FACE_NOT_FOUND = "FACE_NOT_FOUND"
    GALLERY_EMPTY = "GALLERY_EMPTY"
    NO_MATCH = "NO_MATCH"
    NO_ENROLLED_FACES = "NO_ENROLLED_FACES"
    EVENT_LOG_DISABLED = "EVENT_LOG_DISABLED"
    ENROLLMENT_LIMIT = "ENROLLMENT_LIMIT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AppError(HTTPException):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            status_code=status_code,
            detail={
                "error": {
                    "code": code.value,
                    "message": message,
                    "details": details or {},
                }
            },
        )
        self.error_code = code
