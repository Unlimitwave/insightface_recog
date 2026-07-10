"""InsightFace + ONNXRuntime engine with GPU-first / CPU fallback."""

from __future__ import annotations

import logging
import os

import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

from ..config import Settings
from ..core.errors import AppError, ErrorCode
from .base import FaceAnalysisResult, FaceEngine, FaceQuality, LivenessResult
from .liveness import LivenessEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# resolve_providers: 根据 device 参数决定 ONNX Runtime 的执行后端
#   - device="auto" 或 "cuda": 优先用 GPU(CUDAExecutionProvider)，不可用时回退到 CPU
#   - device="cpu": 强制使用 CPU 推理
# 返回值: (providers 列表, ctx_id, 可读的设备标签)
#   - ctx_id=0 表示使用第 0 块 GPU, ctx_id=-1 表示使用 CPU
# ---------------------------------------------------------------------------
def resolve_providers(device: str) -> tuple[list[str], int, str]:
    """Return (providers, ctx_id, device_label). GPU preferred when device=auto."""
    # 获取当前环境可用的 ONNX Runtime 执行提供者(如 CUDA, TensorRT, CPU 等)
    available = ort.get_available_providers()
    logger.info("ONNXRuntime available providers: %s", available)
    want_cuda = device in ("auto", "cuda")
    if want_cuda and "CUDAExecutionProvider" in available:
        # GPU 可用: 优先级 CUDA > CPU(作为 fallback)
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0, "cuda:0"
    if device == "cuda" and "CUDAExecutionProvider" not in available:
        logger.warning("CUDA requested but unavailable; falling back to CPU")
    # 纯 CPU 推理
    return ["CPUExecutionProvider"], -1, "cpu"


# ---------------------------------------------------------------------------
# InsightFaceEngine: 整个人脸系统的核心引擎
# 功能涵盖三大部分:
#   (1) 人脸检测 + 特征提取  通过 InsightFace 的 FaceAnalysis 完成
#   (2) 活体检测(可选)       通过 LivenessEngine(静默反欺骗模型) 完成
#   (3) 质量校验              检测置信度阈值 + 人脸最小尺寸检查
# 初始化流程: 解析运行设备 → 加载 InsightFace 模型 → 配置检测/识别模块 → 初始化活体引擎
# ---------------------------------------------------------------------------
class InsightFaceEngine(FaceEngine):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # 步骤1: 确定 ONNX Runtime 的推理后端(GPU / CPU)
        self.providers, self.ctx_id, self._device_label = resolve_providers(settings.device)
        root = os.path.expanduser(settings.insightface_root)

        # 步骤2: 创建 InsightFace 应用实例, 只开启 detection(检测) + recognition(识别) 模块
        self.app = FaceAnalysis(
            name=settings.model_pack,
            root=root,
            providers=self.providers,
            allowed_modules=["detection", "recognition"],
        )
        # 步骤3: 准备模型 - 设置上下文ID(0=GPU, -1=CPU), 检测阈值, 检测尺寸
        self.app.prepare(
            ctx_id=self.ctx_id,
            det_thresh=settings.det_thresh,
            det_size=(settings.det_size, settings.det_size),
        )

        # 步骤4: 如果配置启用了活体检测, 则初始化静默反欺骗引擎
        self.liveness_engine: LivenessEngine | None = None
        if settings.liveness_enabled:
            self.liveness_engine = LivenessEngine(settings, self.providers)

        # InsightFace 默认输出 512 维特征向量
        self._embedding_dim = 512

        logger.info(
            "InsightFaceEngine ready: pack=%s device=%s providers=%s liveness=%s",
            settings.model_pack,
            self._device_label,
            self.providers,
            self.liveness_engine.available if self.liveness_engine else False,
        )

    @property
    def device_label(self) -> str:
        return self._device_label

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    # -----------------------------------------------------------------------
    # analyze: 对单张图片执行完整的人脸分析流水线
    # 流水线流程:
    #   ① 人脸检测 — InsightFace 检测图中所有人脸, 得到 bbox + 关键点
    #   ② 多脸策略 — 多脸时按 multi_face_policy 决定: reject(拒绝) / largest(选最大的)
    #   ③ 质量校验 — 检测置信度 >= min_det_score, 人脸像素 >= min_face_size_px
    #   ④ 活体检测(可选) — 两张模型(MiniFASNetV1SE + V2) 的 RGB 被动活体判断
    #   ⑤ 特征提取 — 取 normed_embedding(已归一化的 512 维向量), 再做一次 L2 归一化保底
    #   ⑥ 返回 FaceAnalysisResult(embedding, quality, liveness, raw)
    # -----------------------------------------------------------------------
    def analyze(
        self,
        image_bgr: np.ndarray,
        *,
        check_liveness: bool = False,
    ) -> FaceAnalysisResult:
        # ① 人脸检测: 调用 InsightFace 检测图中所有人脸
        faces = self.app.get(image_bgr)
        if not faces:
            raise AppError(ErrorCode.FACE_NOT_DETECTED, "No face detected in image")

        # ② 多脸处理: 根据配置决定是拒绝还是选一张继续
        if len(faces) > 1 and self.settings.multi_face_policy == "reject":
            raise AppError(
                ErrorCode.MULTIPLE_FACES,
                "Multiple faces detected; set multi_face_policy=largest or use a single-face image",
                details={"face_count": len(faces)},
            )

        # ③ 选脸: 单脸直接用, 多脸则选 bbox 面积最大的(策略=largest)
        face = self._select_face(faces)
        bbox = face.bbox.astype(float)
        fw = float(bbox[2] - bbox[0])
        fh = float(bbox[3] - bbox[1])
        quality = FaceQuality(
            det_score=float(face.det_score),
            bbox=bbox.tolist(),
            face_width_px=fw,
            face_height_px=fh,
        )
        # ④ 质量校验: 检测置信度 + 人脸最小尺寸
        self._validate_quality(quality)

        # ⑤ 活体检测(可选): 仅当调用方显式要求 check_liveness=True 时执行
        liveness: LivenessResult | None = None
        if check_liveness:
            if not self.liveness_engine or not self.liveness_engine.available:
                raise AppError(
                    ErrorCode.LIVENESS_UNAVAILABLE,
                    "Liveness models not installed; run scripts/download_models.sh",
                    status_code=503,
                )
            liveness = self.liveness_engine.check(image_bgr, bbox)
            if not liveness.passed:
                raise AppError(
                    ErrorCode.LIVENESS_FAILED,
                    "Liveness check failed (possible spoof / printed photo / screen replay)",
                    details={
                        "score": liveness.score,
                        "threshold": self.settings.liveness_threshold,
                        "model_scores": liveness.model_scores,
                    },
                )

        # ⑥ 特征提取: normed_embedding 已经是 InsightFace 归一化后的向量
        #    这里再做一次 L2 归一化，确保向量模长为 1(与 FAISS 内积搜索一致)
        emb = np.asarray(face.normed_embedding, dtype=np.float32)
        if emb.ndim != 1:
            emb = emb.reshape(-1)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return FaceAnalysisResult(embedding=emb, quality=quality, liveness=liveness, raw=face)

    # -----------------------------------------------------------------------
    # _select_face: 多脸时选择策略
    #   - 只有一张脸: 直接返回
    #   - 多张脸: 按 bbox 面积 (宽×高) 选最大的一张(通常离镜头最近的主脸)
    # -----------------------------------------------------------------------
    def _select_face(self, faces: list) -> object:
        if len(faces) == 1:
            return faces[0]
        # largest face by bbox area
        def area(f) -> float:
            b = f.bbox
            return float((b[2] - b[0]) * (b[3] - b[1]))

        return max(faces, key=area)

    # -----------------------------------------------------------------------
    # _validate_quality: 人脸质量双重校验
    #   (1) 检测置信度(det_score) < min_det_score → 认为检测不可靠, 拒绝
    #   (2) 人脸像素最小边长 < min_face_size_px        → 人脸太小看不清, 拒绝
    # -----------------------------------------------------------------------
    def _validate_quality(self, quality: FaceQuality) -> None:
        if quality.det_score < self.settings.min_det_score:
            raise AppError(
                ErrorCode.LOW_FACE_QUALITY,
                "Face detection confidence too low",
                details={
                    "det_score": quality.det_score,
                    "min_det_score": self.settings.min_det_score,
                },
            )
        min_side = min(quality.face_width_px, quality.face_height_px)
        if min_side < self.settings.min_face_size_px:
            raise AppError(
                ErrorCode.LOW_FACE_QUALITY,
                "Face too small in image",
                details={
                    "face_min_side_px": min_side,
                    "min_face_size_px": self.settings.min_face_size_px,
                },
            )
