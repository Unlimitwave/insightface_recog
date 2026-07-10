"""Face analysis engine abstraction (swap backend without changing API)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FaceQuality:
    det_score: float
    bbox: list[float]
    face_width_px: float
    face_height_px: float


@dataclass
class LivenessResult:
    passed: bool
    score: float
    method: str = "rgb_passive_minifasnet"
    model_scores: dict[str, float] = field(default_factory=dict)
    reason: str | None = None


@dataclass
class FaceAnalysisResult:
    embedding: np.ndarray
    quality: FaceQuality
    liveness: LivenessResult | None = None
    raw: Any = None


class FaceEngine(ABC):
    @property
    @abstractmethod
    def device_label(self) -> str:
        """Human-readable inference device, e.g. cuda:0 or cpu."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass

    @abstractmethod
    def analyze(
        self,
        image_bgr: np.ndarray,
        *,
        check_liveness: bool = False,
    ) -> FaceAnalysisResult:
        """Detect largest face, optional liveness, return L2-normalized embedding."""

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        return float(np.dot(a, b))
