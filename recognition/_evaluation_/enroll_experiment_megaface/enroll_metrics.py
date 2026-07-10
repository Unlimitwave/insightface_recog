"""多注册张数实验 —— 标准化指标计算模块。

实现方式对齐：
  - recognition/_evaluation_/metrics/production_metrics.py（共享指标库）
  - python-package/insightface/gui/core/evaluation.py（企业 1:1 评测）
  - recognition/_evaluation_/megaface/enrollment_ablation.py（MegaFace 消融）
  - docs/人脸识别算法指标与评测指南.md 第 3 节

指标命名（生产上线常用）：
  1:N 识别：Rank-1 / Rank-5 / mAP / TPIR@FPIR / FNIR(漏检率) / FPIR(误检率)
  1:1 验证：TAR@FAR / FRR@FAR(拒识率) / EER / 最佳阈值 Accuracy
"""

from __future__ import print_function

import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

_EVAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _EVAL_ROOT not in sys.path:
    sys.path.insert(0, _EVAL_ROOT)

from metrics.production_metrics import (
    FAR_TARGETS_11,
    FPIR_TARGETS_1N,
    compute_best_threshold_metrics,
    compute_eer,
    compute_tar_at_far,
    far_label,
    linear_interp_logx,
)

# Re-export for backward compatibility
__all__ = [
    "FAR_TARGETS_11",
    "FPIR_TARGETS_1N",
    "far_label",
    "linear_interp_logx",
    "compute_tar_at_far",
    "compute_eer",
    "compute_best_threshold_metrics",
    "count_higher_scores",
    "compute_rank1_pct",
    "compute_rankk_pct",
    "compute_map_pct",
    "evaluate_identification_1n",
    "evaluate_verification_11",
]


def count_higher_scores(
    identity_scores: np.ndarray,
    probe_cols: np.ndarray,
    distractor_scores: np.ndarray,
) -> np.ndarray:
    """统计每个探针有多少 gallery 条目得分严格高于「正确身份 Max 分」。

    Gallery = 80 个身份 Max 分 + 全部 distractor 分。
    该计数 + 1 = 正确身份在全局排序中的名次（1-based rank）。

    用于向量化计算 MegaFace 风格 Rank-1 / Rank-5 / mAP。
    """
    num_probes = identity_scores.shape[0]
    correct_scores = identity_scores[np.arange(num_probes), probe_cols]

    # 其他身份中得分高于正确身份的数量
    masked = identity_scores.copy()
    masked[np.arange(num_probes), probe_cols] = -np.inf
    higher_id = np.sum(masked > correct_scores[:, None], axis=1)

    # distractor 中得分高于正确身份的数量
    higher_dist = np.sum(distractor_scores > correct_scores[:, None], axis=1)

    return higher_id + higher_dist


def compute_rank1_pct(higher_counts: np.ndarray) -> float:
    """Rank-1：正确身份全局排名第 1（没有任何 gallery 条目得分更高）。"""
    hits = int(np.sum(higher_counts == 0))
    return 100.0 * hits / max(1, len(higher_counts))


def compute_rankk_pct(higher_counts: np.ndarray, k: int) -> float:
    """Rank-K：正确身份出现在全局 Top-K 内（至多 K-1 个条目得分更高）。"""
    hits = int(np.sum(higher_counts <= (k - 1)))
    return 100.0 * hits / max(1, len(higher_counts))


def compute_map_pct(higher_counts: np.ndarray) -> float:
    """mAP（单正样本简化）：每个探针 AP = 1/rank，mAP = mean(AP)。

    当 gallery 中仅正确身份为 relevant 时，Average Precision 等于 1/rank。
    rank = higher_counts + 1（1-based）。
    """
    ranks = higher_counts.astype(np.float64) + 1.0
    return float(np.mean(1.0 / ranks) * 100.0)


def evaluate_identification_1n(
    identity_scores: np.ndarray,
    probe_cols: np.ndarray,
    distractor_scores: np.ndarray,
    fpir_targets: Sequence[float] = FPIR_TARGETS_1N,
) -> Dict[str, object]:
    """1:N 识别指标汇总（MegaFace 百万 distractor 协议）。

    Step 1. 计算 Rank-1 / Rank-5 / mAP（检索排序类，不依赖阈值）
    Step 2. 构造 TPIR@FPIR 正负样本：
              正样本 = 探针 vs 正确身份 Max 分
              负样本 = max(最佳错误身份分, distractor 最高分)
    Step 3. 扫描阈值得到 TPIR / FNIR(漏检) / FPIR(误检)
    """
    num_probes = identity_scores.shape[0]
    higher_counts = count_higher_scores(identity_scores, probe_cols, distractor_scores)

    # --- 检索类指标 ---
    rank1 = compute_rank1_pct(higher_counts)
    rank5 = compute_rankk_pct(higher_counts, k=5)
    map_score = compute_map_pct(higher_counts)

    # --- 阈值类指标（TPIR@FPIR） ---
    pos_scores = identity_scores[np.arange(num_probes), probe_cols]
    masked = identity_scores.copy()
    masked[np.arange(num_probes), probe_cols] = -np.inf
    best_wrong_enroll = np.max(masked, axis=1)
    max_distractor = np.max(distractor_scores, axis=1)
    neg_scores = np.maximum(best_wrong_enroll, max_distractor)

    tpir_detail = compute_tar_at_far(
        pos_scores.tolist(), neg_scores.tolist(), fpir_targets
    )

    tpir_at_fpir = {}
    fnir_at_fpir = {}
    fpir_at_fpir = {}
    threshold_at_fpir = {}
    for target in fpir_targets:
        key = far_label(target)
        detail = tpir_detail.get(target)
        if detail is None:
            tpir_at_fpir[key] = None
            fnir_at_fpir[key] = None
            fpir_at_fpir[key] = None
            threshold_at_fpir[key] = None
        else:
            tpir_at_fpir[key] = detail["tar"] * 100.0
            fnir_at_fpir[key] = detail["frr"] * 100.0
            fpir_at_fpir[key] = detail["far"] * 100.0
            threshold_at_fpir[key] = detail["threshold"]

    return {
        "probe_count": int(num_probes),
        "rank1": rank1,
        "rank5": rank5,
        "map": map_score,
        "tpir_at_fpir": tpir_at_fpir,
        "fnir_at_fpir": fnir_at_fpir,
        "fpir_at_fpir": fpir_at_fpir,
        "threshold_at_fpir": threshold_at_fpir,
        "tpir_at_fpir_detail": {
            far_label(t): tpir_detail.get(t) for t in fpir_targets
        },
    }


def evaluate_verification_11(
    identity_scores: np.ndarray,
    probe_cols: np.ndarray,
    far_targets: Sequence[float] = FAR_TARGETS_11,
) -> Dict[str, object]:
    """1:1 验证指标汇总（不含 distractor）。

    Step 1. 正样本：每条探针 vs 本人 k 张注册照的 Max 分（1 条/探针）
    Step 2. 负样本：每条探针 vs 每个错误身份的 Max 分（79 条/探针）
    Step 3. 扫描阈值得到 TAR@FAR / FRR@FAR / EER / 最佳 Accuracy
    """
    num_probes = identity_scores.shape[0]

    # 正样本：探针与正确身份列的 Max 分
    pos_scores = identity_scores[np.arange(num_probes), probe_cols]

    # 负样本：排除正确身份列后展平（每条探针贡献 num_identities-1 条）
    wrong_mask = np.ones_like(identity_scores, dtype=bool)
    wrong_mask[np.arange(num_probes), probe_cols] = False
    neg_scores = identity_scores[wrong_mask]

    positive_list = pos_scores.tolist()
    negative_list = neg_scores.tolist()

    tar_detail = compute_tar_at_far(positive_list, negative_list, far_targets)
    eer_info = compute_eer(positive_list, negative_list)
    best_info = compute_best_threshold_metrics(positive_list, negative_list)

    tar_at_far = {}
    frr_at_far = {}
    far_at_far = {}
    threshold_at_far = {}
    for target in far_targets:
        key = far_label(target)
        detail = tar_detail.get(target)
        if detail is None:
            tar_at_far[key] = None
            frr_at_far[key] = None
            far_at_far[key] = None
            threshold_at_far[key] = None
        else:
            tar_at_far[key] = detail["tar"] * 100.0
            frr_at_far[key] = detail["frr"] * 100.0
            far_at_far[key] = detail["far"] * 100.0
            threshold_at_far[key] = detail["threshold"]

    return {
        "positive_pairs": len(positive_list),
        "negative_pairs": len(negative_list),
        "tar_at_far": tar_at_far,
        "frr_at_far": frr_at_far,
        "far_at_far": far_at_far,
        "threshold_at_far": threshold_at_far,
        "tar_at_far_detail": {far_label(t): tar_detail.get(t) for t in far_targets},
        "eer": None if eer_info["eer"] is None else eer_info["eer"] * 100.0,
        "eer_threshold": eer_info["threshold"],
        "best_threshold": best_info,
    }
