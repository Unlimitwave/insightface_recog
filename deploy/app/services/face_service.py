"""Business logic for enrollment and 1:N identification."""

from __future__ import annotations

import logging
import time
from typing import BinaryIO

import cv2
import numpy as np

from ..config import Settings
from ..core.errors import AppError, ErrorCode
from ..engine.base import FaceEngine
from ..schemas.api import (
    EnrollResponse,
    EnrolledFaceResponse,
    EventListResponse,
    EventResponse,
    EventStats,
    FaceQualityInfo,
    GalleryStats,
    IdentifyCandidate,
    IdentifyResponse,
    LivenessInfo,
    PersonActivityItem,
    PersonCreateRequest,
    PersonListResponse,
    PersonResponse,
    StatsResponse,
    VerifyResponse,
    DailyStatsItem,
)
from .event_store import EventStore
from .gallery import GalleryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# decode_image_bytes: 将 HTTP 上传的图片字节流转为 OpenCV 的 BGR 图像(ndarray)
#   步骤: bytes → uint8 数组 → cv2.imdecode 解码(支持 JPEG/PNG 等常见格式)
#   注意: 如果图片数据损坏或不支持的格式, cv2.imdecode 返回 None, 此时抛出异常
# ---------------------------------------------------------------------------
def decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise AppError(ErrorCode.INVALID_REQUEST, "Invalid or unsupported image data")
    return image


# ---------------------------------------------------------------------------
# FaceService: 业务逻辑层, 负责协调 engine(人脸分析引擎) 和 gallery(底库存储/搜索)
# 对外提供核心能力:
#   - 人员管理: create_person / get_person / list_persons / delete_person
#   - 人脸注册: enroll_faces(多张图片注册到一个人员)
#   - 人脸识别: identify(1:N 识别, 含陌生人检测/告警)
#   - 人脸验证: verify(1:1 比对, 探针 vs 指定人员底库模板)
#   - 审计统计: list_events / get_stats
#   - 辅助方法: _quality_info / _liveness_info / _person_to_response / _log_event
# ---------------------------------------------------------------------------
class FaceService:
    def __init__(
        self,
        engine: FaceEngine,
        gallery: GalleryStore,
        settings: Settings,
        event_store: EventStore | None = None,
    ) -> None:
        self.engine = engine
        self.gallery = gallery
        self.settings = settings
        self.event_store = event_store

    def create_person(self, req: PersonCreateRequest) -> PersonResponse:
        try:
            record = self.gallery.create_person(
                req.person_id, req.display_name, req.metadata
            )
        except ValueError as exc:
            raise AppError(ErrorCode.INVALID_REQUEST, str(exc)) from exc
        return self._person_to_response(record)

    def get_person(self, person_id: str) -> PersonResponse:
        try:
            record = self.gallery.get_person(person_id)
        except KeyError as exc:
            raise AppError(
                ErrorCode.PERSON_NOT_FOUND, f"Person not found: {person_id}", status_code=404
            ) from exc
        return self._person_to_response(record)

    def list_persons(self, offset: int = 0, limit: int = 100) -> PersonListResponse:
        items, total = self.gallery.list_persons(offset=offset, limit=min(limit, 500))
        return PersonListResponse(
            total=total,
            items=[self._person_to_response(r) for r in items],
        )

    def delete_person(self, person_id: str) -> None:
        try:
            self.gallery.delete_person(person_id)
        except KeyError as exc:
            raise AppError(
                ErrorCode.PERSON_NOT_FOUND, f"Person not found: {person_id}", status_code=404
            ) from exc

    # -------------------------------------------------------------------
    # enroll_faces: 人脸注册 - 为一个人员注册多张人脸图像
    # 完整流程:
    #   ① 容量检查 — 当前已有 face_count + 本次欲注册数量 > max_faces_per_person → 拒绝
    #   ② 活体开关 — 检查是否需要活体检测(配置启用 + 注册时需要 + 未跳过)
    #   ③ 逐图处理 —
    #       a. 读取图片字节 → decode_image_bytes 解码为 BGR 图像
    #       b. engine.analyze()  → 检测+质量校验+活体检查(可选)+提取 512 维特征
    #       c. gallery.add_face() → 特征向量 L2 归一化 → 写入 SQLite + FAISS 索引
    #   ④ 返回 EnrollResponse(包含每张人脸的质量分/活体结果/创建时间等)
    # -------------------------------------------------------------------
    def enroll_faces(
        self,
        person_id: str,
        files: list[tuple[str, BinaryIO]],
        request_id: str,
        skip_liveness: bool = False,
    ) -> EnrollResponse:
        try:
            person = self.gallery.get_person(person_id)
        except KeyError as exc:
            raise AppError(
                ErrorCode.PERSON_NOT_FOUND, f"Person not found: {person_id}", status_code=404
            ) from exc

        if person.face_count + len(files) > self.settings.max_faces_per_person:
            raise AppError(
                ErrorCode.ENROLLMENT_LIMIT,
                f"Max {self.settings.max_faces_per_person} faces per person",
                details={
                    "current": person.face_count,
                    "requested": len(files),
                    "max": self.settings.max_faces_per_person,
                },
            )

        check_liveness = (
            self.settings.liveness_enabled
            and self.settings.liveness_on_enroll
            and not skip_liveness
        )

        enrolled: list[EnrolledFaceResponse] = []
        for filename, file_obj in files:
            data = file_obj.read()
            image = decode_image_bytes(data)
            t0 = time.perf_counter()
            result = self.engine.analyze(image, check_liveness=check_liveness)
            _ = time.perf_counter() - t0

            stored = self.gallery.add_face(person_id, result.embedding)
            enrolled.append(
                EnrolledFaceResponse(
                    face_id=stored.face_id,
                    person_id=person_id,
                    created_at=stored.created_at,
                    quality=self._quality_info(result.quality),
                    liveness=self._liveness_info(result.liveness),
                )
            )
            logger.info("Enrolled face %s for %s from %s", stored.face_id, person_id, filename)

        updated = self.gallery.get_person(person_id)
        return EnrollResponse(
            request_id=request_id,
            person_id=person_id,
            enrolled=enrolled,
            total_faces=updated.face_count,
        )

    # -------------------------------------------------------------------
    # identify: 1:N 人脸识别 - 根据一张照片在底库中查找最相似的人员
    # 完整流程:
    #   ① 底库空检查 — 如果底库中没有任何人脸特征, 直接返回错误(503)
    #   ② 活体开关 — 同 enroll_faces, 检查是否需要活体检测
    #   ③ 人脸分析 — engine.analyze() 同上(检测+质量+活体+特征提取)
    #   ④ FAISS 搜索 — gallery.search() 用余弦相似度(归一化内积)在底库中搜索 top_k 最相似
    #   ⑤ 阈值匹配 — 将搜索结果逐一与 identify_threshold 比较, >= 阈值才算匹配
    #   ⑥ 构建返回 — 包含最佳候选人 + 完整候选列表 + 质量/活体信息 + 耗时统计
    # -------------------------------------------------------------------
    def identify(
        self,
        image_bytes: bytes,
        request_id: str,
        skip_liveness: bool = False,
    ) -> IdentifyResponse:
        if self.gallery.total_faces == 0:
            raise AppError(
                ErrorCode.GALLERY_EMPTY,
                "Gallery is empty; enroll persons before identification",
                status_code=503,
            )

        check_liveness = (
            self.settings.liveness_enabled
            and self.settings.liveness_on_identify
            and not skip_liveness
        )

        image = decode_image_bytes(image_bytes)
        t0 = time.perf_counter()
        result = self.engine.analyze(image, check_liveness=check_liveness)
        infer_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        hits = self.gallery.search(result.embedding, top_k=self.settings.identify_top_k)
        search_ms = (time.perf_counter() - t1) * 1000

        threshold = self.settings.identify_threshold
        candidates: list[IdentifyCandidate] = []
        for rank, hit in enumerate(hits, start=1):
            matched = hit.similarity >= threshold
            candidates.append(
                IdentifyCandidate(
                    rank=rank,
                    person_id=hit.person_id,
                    display_name=hit.display_name,
                    similarity=round(hit.similarity, 6),
                    matched=matched,
                )
            )

        best = candidates[0] if candidates else None
        matched = best is not None and best.matched
        is_stranger = not matched
        alert = is_stranger and self.settings.stranger_alert_enabled
        total_ms = infer_ms + search_ms

        self._log_event(
            request_id=request_id,
            event_type="identify",
            matched=matched,
            is_stranger=is_stranger,
            threshold=threshold,
            person_id=best.person_id if matched else None,
            display_name=best.display_name if matched else None,
            similarity=best.similarity if best else None,
            liveness_passed=result.liveness.passed if result.liveness else None,
            latency_ms=total_ms,
            metadata={"alert": alert, "candidate_count": len(candidates)},
        )

        return IdentifyResponse(
            request_id=request_id,
            matched=matched,
            person_id=best.person_id if matched else None,
            display_name=best.display_name if matched else None,
            similarity=best.similarity if best else None,
            threshold=threshold,
            is_stranger=is_stranger,
            alert=alert,
            candidates=candidates,
            quality=self._quality_info(result.quality),
            liveness=self._liveness_info(result.liveness),
            latency_ms={
                "inference": round(infer_ms, 2),
                "search": round(search_ms, 2),
                "total": round(total_ms, 2),
            },
        )

    # -------------------------------------------------------------------
    # verify: 1:1 人脸验证 — 探针图与指定 person_id 的底库模板比对
    # 完整流程:
    #   ① 校验人员存在且已注册人脸
    #   ② engine.analyze() 提取探针特征
    #   ③ gallery.verify_against_person() 与该人员所有模板取最高相似度
    #   ④ 与 verify_threshold 比较得出 verified
    #   ⑤ 写入事件日志
    # -------------------------------------------------------------------
    def verify(
        self,
        image_bytes: bytes,
        person_id: str,
        request_id: str,
        skip_liveness: bool = False,
    ) -> VerifyResponse:
        try:
            person = self.gallery.get_person(person_id)
        except KeyError as exc:
            raise AppError(
                ErrorCode.PERSON_NOT_FOUND, f"Person not found: {person_id}", status_code=404
            ) from exc

        if person.face_count == 0:
            raise AppError(
                ErrorCode.NO_ENROLLED_FACES,
                f"Person has no enrolled faces: {person_id}",
                status_code=422,
            )

        check_liveness = (
            self.settings.liveness_enabled
            and self.settings.liveness_on_verify
            and not skip_liveness
        )

        image = decode_image_bytes(image_bytes)
        t0 = time.perf_counter()
        result = self.engine.analyze(image, check_liveness=check_liveness)
        infer_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        try:
            similarity, matched_face_id = self.gallery.verify_against_person(
                person_id, result.embedding
            )
        except ValueError as exc:
            raise AppError(ErrorCode.NO_ENROLLED_FACES, str(exc)) from exc
        verify_ms = (time.perf_counter() - t1) * 1000

        threshold = self.settings.verify_threshold
        verified = similarity >= threshold
        total_ms = infer_ms + verify_ms

        self._log_event(
            request_id=request_id,
            event_type="verify",
            matched=verified,
            is_stranger=False,
            threshold=threshold,
            person_id=person_id,
            display_name=person.display_name,
            similarity=round(similarity, 6),
            liveness_passed=result.liveness.passed if result.liveness else None,
            latency_ms=total_ms,
            metadata={"matched_face_id": matched_face_id},
        )

        return VerifyResponse(
            request_id=request_id,
            verified=verified,
            person_id=person_id,
            display_name=person.display_name,
            similarity=round(similarity, 6),
            threshold=threshold,
            matched_face_id=matched_face_id,
            quality=self._quality_info(result.quality),
            liveness=self._liveness_info(result.liveness),
            latency_ms={
                "inference": round(infer_ms, 2),
                "verify": round(verify_ms, 2),
                "total": round(total_ms, 2),
            },
        )

    def list_events(
        self,
        offset: int = 0,
        limit: int = 50,
        event_type: str | None = None,
        person_id: str | None = None,
        is_stranger: bool | None = None,
    ) -> EventListResponse:
        if self.event_store is None:
            raise AppError(
                ErrorCode.EVENT_LOG_DISABLED,
                "Event logging is disabled",
                status_code=503,
            )
        items, total = self.event_store.list_events(
            offset=offset,
            limit=min(limit, 500),
            event_type=event_type,
            person_id=person_id,
            is_stranger=is_stranger,
        )
        return EventListResponse(
            total=total,
            items=[self._event_to_response(e) for e in items],
        )

    def get_stats(self, days: int = 7) -> StatsResponse:
        gallery_stats = GalleryStats(
            person_count=self.gallery.person_count(),
            face_count=self.gallery.total_faces,
        )
        if self.event_store is None:
            empty_events = EventStats(
                total_events=0,
                identify_count=0,
                verify_count=0,
                matched_count=0,
                stranger_count=0,
                pass_rate=0.0,
                daily=[],
                top_persons=[],
            )
            return StatsResponse(period_days=days, gallery=gallery_stats, events=empty_events)

        summary = self.event_store.stats_summary(days=days)
        return StatsResponse(
            period_days=summary.period_days,
            gallery=gallery_stats,
            events=EventStats(
                total_events=summary.total_events,
                identify_count=summary.identify_count,
                verify_count=summary.verify_count,
                matched_count=summary.matched_count,
                stranger_count=summary.stranger_count,
                pass_rate=summary.pass_rate,
                daily=[
                    DailyStatsItem(
                        date=d.date,
                        total=d.total,
                        matched=d.matched,
                        strangers=d.strangers,
                        verify_pass=d.verify_pass,
                    )
                    for d in summary.daily
                ],
                top_persons=[
                    PersonActivityItem(
                        person_id=p.person_id,
                        display_name=p.display_name,
                        event_count=p.event_count,
                        last_seen_at=p.last_seen_at,
                    )
                    for p in summary.top_persons
                ],
            ),
        )

    def _log_event(self, **kwargs) -> None:
        if not self.settings.event_log_enabled or self.event_store is None:
            return
        try:
            self.event_store.record(**kwargs)
        except Exception:
            logger.exception("Failed to record event request_id=%s", kwargs.get("request_id"))

    # 将引擎内部的质量对象转为 API 响应格式
    @staticmethod
    def _quality_info(q) -> FaceQualityInfo:
        return FaceQualityInfo(
            det_score=q.det_score,
            bbox=q.bbox,
            face_width_px=q.face_width_px,
            face_height_px=q.face_height_px,
        )

    # 将活体检测结果对象转为 API 响应格式(可能为 None 表示未执行活体检测)
    @staticmethod
    def _liveness_info(liveness) -> LivenessInfo | None:
        if liveness is None:
            return None
        return LivenessInfo(
            passed=liveness.passed,
            score=liveness.score,
            method=liveness.method,
            model_scores=liveness.model_scores,
        )

    # 将 GalleryStore 返回的 PersonRecord 转为 API 响应格式
    @staticmethod
    def _person_to_response(record) -> PersonResponse:
        return PersonResponse(
            person_id=record.person_id,
            display_name=record.display_name,
            metadata=record.metadata,
            face_count=record.face_count,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    @staticmethod
    def _event_to_response(event) -> EventResponse:
        return EventResponse(
            event_id=event.event_id,
            request_id=event.request_id,
            event_type=event.event_type,
            created_at=event.created_at,
            matched=event.matched,
            is_stranger=event.is_stranger,
            person_id=event.person_id,
            display_name=event.display_name,
            similarity=event.similarity,
            threshold=event.threshold,
            liveness_passed=event.liveness_passed,
            latency_ms=event.latency_ms,
            metadata=event.metadata,
        )
