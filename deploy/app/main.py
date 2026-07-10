"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .api.deps import init_services
from .api.routes import health, identify, persons, stats, verify
from .config import get_settings
from .core.errors import AppError
from .core.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理函数。
    在应用启动时执行初始化，关闭时执行清理。
    """
    settings = get_settings()
    setup_logging(settings.debug)
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    init_services(settings)
    yield
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-oriented face access control API. "
            "1:N identify, 1:1 verify, passive RGB liveness, event audit. "
            "GPU preferred, CPU fallback."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    prefix = settings.api_prefix
    app.include_router(health.router, prefix=prefix)
    app.include_router(persons.router, prefix=prefix)
    app.include_router(identify.router, prefix=prefix)
    app.include_router(verify.router, prefix=prefix)
    app.include_router(stats.router, prefix=prefix)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": "Request validation failed",
                    "details": {"errors": exc.errors()},
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Internal server error",
                    "details": {},
                }
            },
        )

    return app


app = create_app()
