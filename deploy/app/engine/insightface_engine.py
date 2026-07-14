"""InsightFace + ONNXRuntime engine with GPU-first / CPU fallback."""

from __future__ import annotations

import glob
import logging
import os

import numpy as np
import onnxruntime as ort
from insightface.app.common import Face
from insightface.model_zoo import model_zoo

from ..config import Settings
from ..core.errors import AppError, ErrorCode
from .base import FaceAnalysisResult, FaceEngine, FaceQuality, LivenessResult
from .liveness import LivenessEngine

logger = logging.getLogger(__name__)

_DETECTION_HINTS = ("det_", "scrfd")
_RECOGNITION_HINTS = ("w600k", "r50", "glintr", "glint360k", "arcface", "recog")
# Preferred recognition model when multiple ONNX files exist in the same directory.
_RECOG_PREFERRED_MODELS = ("glint360k_r100.onnx", "w600k_r50.onnx")


def resolve_providers(device: str) -> tuple[list[str], int, str]:
    """Return (providers, ctx_id, device_label). GPU preferred when device=auto."""
    available = ort.get_available_providers()
    logger.info("ONNXRuntime available providers: %s", available)
    want_cuda = device in ("auto", "cuda")
    if want_cuda and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0, "cuda:0"
    if device == "cuda" and "CUDAExecutionProvider" not in available:
        logger.warning("CUDA requested but unavailable; falling back to CPU")
    return ["CPUExecutionProvider"], -1, "cpu"


def _find_onnx_model(
    model_dir: str,
    role: str,
    hints: tuple[str, ...],
    *,
    model_name: str | None = None,
    preferred_models: tuple[str, ...] = (),
) -> str:
    """Resolve a single ONNX file under model_dir (detection or recognition)."""
    expanded = os.path.expanduser(model_dir)
    if not os.path.isdir(expanded):
        raise RuntimeError(f"{role} model directory not found: {expanded}")

    if model_name:
        candidate = os.path.expanduser(model_name)
        if os.path.isfile(candidate):
            return candidate
        path = os.path.join(expanded, model_name)
        if os.path.isfile(path):
            return path
        raise RuntimeError(f"{role} model not found: {model_name} (dir={expanded})")

    files = sorted(glob.glob(os.path.join(expanded, "*.onnx")))
    if not files:
        raise RuntimeError(f"No .onnx model in {expanded}; run scripts/download_models.sh")

    def _pick_preferred(candidates: list[str]) -> str | None:
        for pref in preferred_models:
            pref_lower = pref.lower()
            for path in candidates:
                if os.path.basename(path).lower() == pref_lower:
                    return path
        return None

    # Honor preferred model filenames before hint-based matching.
    preferred_hit = _pick_preferred(files)
    if preferred_hit is not None:
        logger.info("Using preferred %s model: %s", role, os.path.basename(preferred_hit))
        return preferred_hit

    matching = [
        path
        for path in files
        if any(hint in os.path.basename(path).lower() for hint in hints)
    ]

    if len(matching) == 1:
        return matching[0]

    if len(matching) > 1:
        preferred_hit = _pick_preferred(matching)
        if preferred_hit is not None:
            logger.info(
                "Multiple %s models; selected preferred %s",
                role,
                os.path.basename(preferred_hit),
            )
            return preferred_hit
        chosen = matching[0]
        logger.warning(
            "Multiple %s models in %s; using %s (set %s_MODEL_NAME to override)",
            role,
            expanded,
            os.path.basename(chosen),
            role.lower(),
        )
        return chosen

    if len(files) == 1:
        return files[0]

    raise RuntimeError(
        f"Multiple ONNX models in {expanded}; set an explicit model name "
        f"(e.g. RECOG_MODEL_NAME=glint360k_r100.onnx) or keep only one .onnx per role"
    )


class InsightFaceEngine(FaceEngine):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.providers, self.ctx_id, self._device_label = resolve_providers(settings.device)

        det_path = _find_onnx_model(
            settings.det_model_dir,
            "Detection",
            _DETECTION_HINTS,
            model_name=settings.det_model_name,
        )
        rec_path = _find_onnx_model(
            settings.recog_model_dir,
            "Recognition",
            _RECOGNITION_HINTS,
            model_name=settings.recog_model_name,
            preferred_models=_RECOG_PREFERRED_MODELS,
        )

        self.det_model = model_zoo.get_model(det_path, providers=self.providers)
        self.rec_model = model_zoo.get_model(rec_path, providers=self.providers)
        if self.det_model is None or self.rec_model is None:
            raise RuntimeError(
                f"Failed to load face models from {settings.det_model_dir} and {settings.recog_model_dir}"
            )

        det_size = (settings.det_size, settings.det_size)
        self.det_model.prepare(
            ctx_id=self.ctx_id,
            input_size=det_size,
            det_thresh=settings.det_thresh,
        )
        self.rec_model.prepare(ctx_id=self.ctx_id)

        self.liveness_engine: LivenessEngine | None = None
        if settings.liveness_enabled:
            self.liveness_engine = LivenessEngine(settings, self.providers)

        self._embedding_dim = 512

        logger.info(
            "FaceEngine ready: det=%s rec=%s device=%s providers=%s liveness=%s",
            os.path.basename(det_path),
            os.path.basename(rec_path),
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

    def analyze(
        self,
        image_bgr: np.ndarray,
        *,
        check_liveness: bool = False,
    ) -> FaceAnalysisResult:
        faces = self._detect_and_recognize(image_bgr)
        if not faces:
            raise AppError(ErrorCode.FACE_NOT_DETECTED, "No face detected in image")

        if len(faces) > 1 and self.settings.multi_face_policy == "reject":
            raise AppError(
                ErrorCode.MULTIPLE_FACES,
                "Multiple faces detected; set multi_face_policy=largest or use a single-face image",
                details={"face_count": len(faces)},
            )

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
        self._validate_quality(quality)

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

        emb = np.asarray(face.normed_embedding, dtype=np.float32)
        if emb.ndim != 1:
            emb = emb.reshape(-1)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return FaceAnalysisResult(embedding=emb, quality=quality, liveness=liveness, raw=face)

    def _detect_and_recognize(self, image_bgr: np.ndarray) -> list[Face]:
        bboxes, kpss = self.det_model.detect(image_bgr, max_num=0, metric="default")
        if bboxes.shape[0] == 0:
            return []

        faces: list[Face] = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = kpss[i] if kpss is not None else None
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            self.rec_model.get(image_bgr, face)
            faces.append(face)
        return faces

    def _select_face(self, faces: list) -> object:
        if len(faces) == 1:
            return faces[0]

        def area(f) -> float:
            b = f.bbox
            return float((b[2] - b[0]) * (b[3] - b[1]))

        return max(faces, key=area)

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
