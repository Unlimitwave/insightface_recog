"""Passive RGB liveness via MiniFASNet V1SE + V2 ensemble (industry common practice)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from ..config import Settings
from .base import LivenessResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MODEL_SPECS: 两个被动 RGB 活体检测模型的定义
#   - MiniFASNetV1SE: scale=4.0, 裁剪人脸时会按 bbox 外扩 4 倍以获取更多上下文
#   - MiniFASNetV2:   scale=2.7, 同上, 外扩比例稍小
# 这两个模型来自 Silent-Face-Anti-Spoofing 开源项目, 业界常用于静默活体检测
# ---------------------------------------------------------------------------
MODEL_SPECS = {
    "MiniFASNetV1SE": {"filename": "MiniFASNetV1SE.onnx", "scale": 4.0},
    "MiniFASNetV2": {"filename": "MiniFASNetV2.onnx", "scale": 2.7},
}


# _AntiSpoofModel 数据类: 封装一个活体检测 ONNX 模型的运行时信息
#   - session:       ONNX Runtime 推理会话
#   - input_name:    模型输入节点名(如 "input")
#   - output_name:   模型输出节点名(如 "output")
#   - input_size:    模型要求的输入尺寸 (H, W), 人脸裁剪后会 resize 到这个尺寸
#   - scale:         人脸 bbox 外扩倍数, 用于在裁剪时引入周围背景上下文
@dataclass
class _AntiSpoofModel:
    name: str
    session: ort.InferenceSession
    input_name: str
    output_name: str
    input_size: tuple[int, int]
    scale: float


class LivenessEngine:
    """Silent-Face-Anti-Spoofing ONNX models (RGB passive liveness)."""

    # -------------------------------------------------------------------
    # 初始化: 从 antispoof_dir 目录加载两个 ONNX 模型
    #  1. 遍历 MODEL_SPECS 中的模型定义
    #  2. 检查对应的 .onnx 文件是否存在, 不存在则跳过(warning)
    #  3. 创建 ONNX Runtime 推理会话, 获取输入/输出的名称和尺寸
    #  4. 记录每个模型的 scale 参数(用于后续人脸区域裁剪)
    # -------------------------------------------------------------------
    def __init__(self, settings: Settings, providers: list[str]) -> None:
        self.settings = settings
        self.models: list[_AntiSpoofModel] = []
        model_dir = os.path.expanduser(settings.antispoof_dir)

        for name, spec in MODEL_SPECS.items():
            path = os.path.join(model_dir, spec["filename"])
            if not os.path.isfile(path):
                logger.warning("Liveness model missing: %s (skip)", path)
                continue
            session = ort.InferenceSession(path, providers=providers)
            inp = session.get_inputs()[0]
            out = session.get_outputs()[0]
            h, w = int(inp.shape[2]), int(inp.shape[3])
            self.models.append(
                _AntiSpoofModel(
                    name=name,
                    session=session,
                    input_name=inp.name,
                    output_name=out.name,
                    input_size=(h, w),
                    scale=float(spec["scale"]),
                )
            )
            logger.info("Loaded liveness model %s from %s", name, path)

    @property
    def available(self) -> bool:
        return len(self.models) > 0

    # -------------------------------------------------------------------
    # check: 活体检测主入口
    # 流程:
    #   ① 从 bbox_xyxy 解析人脸坐标 (x1, y1, x2, y2)
    #   ② 对每个已加载的模型, 调用 _predict_real_prob 得到"真人概率"(0~1)
    #   ③ 每个模型单独判断: 概率 >= liveness_threshold → 视为真人
    #   ④ 合并策略:
    #      - liveness_require_both_models=True  → 两个模型都通过才算真人(严格模式)
    #      - liveness_require_both_models=False → 任一模型通过即算真人(宽松模式)
    #   ⑤ 计算所有模型的平均分作为该次检测的综合得分
    #   ⑥ 返回 LivenessResult(passed, score, model_scores, reason)
    # -------------------------------------------------------------------
    def check(self, image_bgr: np.ndarray, bbox_xyxy: np.ndarray) -> LivenessResult:
        if not self.models:
            return LivenessResult(
                passed=False,
                score=0.0,
                reason="liveness_models_not_installed",
            )

        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
        scores: dict[str, float] = {}
        passes: dict[str, bool] = {}

        for model in self.models:
            real_prob = self._predict_real_prob(image_bgr, [x1, y1, x2, y2], model)
            scores[model.name] = real_prob
            passes[model.name] = real_prob >= self.settings.liveness_threshold

        if self.settings.liveness_require_both_models and len(self.models) >= 2:
            passed = all(passes.values())
        else:
            passed = any(passes.values())

        avg_score = float(np.mean(list(scores.values()))) if scores else 0.0
        return LivenessResult(
            passed=passed,
            score=avg_score,
            model_scores=scores,
            reason=None if passed else "spoof_detected",
        )

    # -------------------------------------------------------------------
    # _predict_real_prob: 用单个模型预测人脸是"真人"的概率
    # 步骤:
    #   ① 将 bbox_xyxy 转为 bbox_xywh(宽高)格式
    #   ② _crop_face: 按 scale 倍数外扩裁剪人脸区域, 再 resize 到模型的 input_size
    #   ③ 图像格式转换: HWC → CHW(NCHW), float32
    #   ④ ONNX 推理: 得到 logits(原始分类得分, 形状 (1, 2))
    #   ⑤ softmax: logits → 概率分布 [假人脸概率, 真人概率]
    #   ⑥ 返回 probs[0, 1], 即 label=1(Real) 的概率(Silent-Face 约定 label 1 = 真人)
    # -------------------------------------------------------------------
    def _predict_real_prob(
        self,
        image_bgr: np.ndarray,
        bbox_xyxy: list[float],
        model: _AntiSpoofModel,
    ) -> float:
        x1, y1, x2, y2 = bbox_xyxy
        bbox_xywh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        crop = self._crop_face(image_bgr, bbox_xywh, model.scale, model.input_size)
        tensor = crop.astype(np.float32)
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        logits = model.session.run([model.output_name], {model.input_name: tensor})[0]
        probs = self._softmax(logits)
        # label 1 = Real (Silent-Face-Anti-Spoofing convention)
        return float(probs[0, 1])

    # -------------------------------------------------------------------
    # _crop_face: 从原图中裁剪并缩放人脸区域
    # 参数说明:
    #   - bbox_xywh: [x, y, w, h] 格式的人脸框
    #   - scale:     外扩倍数, 以人脸中心为基准向外扩大(如 4.0 代表 4 倍宽高), 
    #                目的是引入更多周围背景, 帮助模型判断是否是翻拍/屏幕
    #   - input_size: 模型要求的输入尺寸 (H, W), 最终 resize 到这个尺寸
    # 边界保护: 裁剪坐标限制在原图范围内, 防止越界
    # -------------------------------------------------------------------
    @staticmethod
    def _crop_face(
        image: np.ndarray,
        bbox_xywh: list[int],
        scale: float,
        input_size: tuple[int, int],
    ) -> np.ndarray:
        src_h, src_w = image.shape[:2]
        x, y, box_w, box_h = bbox_xywh
        scale = min((src_h - 1) / max(box_h, 1), (src_w - 1) / max(box_w, 1), scale)
        new_w = box_w * scale
        new_h = box_h * scale
        center_x = x + box_w / 2
        center_y = y + box_h / 2
        x1 = max(0, int(center_x - new_w / 2))
        y1 = max(0, int(center_y - new_h / 2))
        x2 = min(src_w - 1, int(center_x + new_w / 2))
        y2 = min(src_h - 1, int(center_y + new_h / 2))
        cropped = image[y1 : y2 + 1, x1 : x2 + 1]
        return cv2.resize(cropped, (input_size[1], input_size[0]))

    # ---------------------------------------------------------------
    # _softmax: 将模型的原始输出(logits)转换为概率分布(0~1)
    # 公式: softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
    # 减 max 是为了数值稳定性, 防止 exp 溢出
    # ---------------------------------------------------------------
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
