"""Production-oriented face recognition metrics (shared across benchmarks).

Aligned with:
  - python-package/insightface/gui/core/evaluation.py
  - docs/人脸识别算法指标与评测指南.md §3
  - recognition/_evaluation_/enroll_experiment_megaface/enroll_metrics.py

1:1 verification: TAR@FAR / FRR@FAR / Threshold@FAR / EER / AUC
1:N identification: Rank-1/5/10 / mAP / TPIR@FPIR / FNIR(Miss) / FPIR(False Positive Retrieval)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Standard operating-point targets (NIST FRVT / IJB / enterprise eval)
# ---------------------------------------------------------------------------

FPIR_TARGETS_1N = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
FAR_TARGETS_11 = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
FAR_TARGETS_11_DISPLAY = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
FPIR_TARGETS_1N_DISPLAY = (1e-1, 1e-2, 1e-3, 1e-4)


def far_label(value: float) -> str:
    """Format FAR/FPIR target for JSON keys, e.g. 0.0001 -> '1e-4'."""
    text = "%.0e" % value
    mantissa, exponent = text.split("e")
    return "%se%s" % (mantissa, int(exponent))


def linear_interp_logx(x_t, x1, x2, y1, y2):
    """Log-space linear interpolation (MegaFace official script style)."""
    if x2 <= x1:
        return y1
    if x1 <= 0 or x2 <= 0:
        return y1
    lx, lx1, lx2 = math.log10(x_t), math.log10(x1), math.log10(x2)
    f1 = (lx - lx1) / (lx2 - lx1)
    f2 = 1.0 - f1
    return y1 * f2 + y2 * f1


def compute_tar_at_far(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    far_targets: Sequence[float],
) -> Dict[float, Optional[Dict[str, float]]]:
    """Compute TAR@FAR (1:1) or TPIR@FPIR (1:N) with paired FRR/FNIR and threshold.

    Returns {target: {"tar", "far", "frr", "threshold"}} with rates in [0, 1].
    """
    result = {far: None for far in far_targets}
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return result

    positives = np.sort(np.asarray(positive_scores, dtype=np.float64))
    negatives = np.sort(np.asarray(negative_scores, dtype=np.float64))
    thresholds = np.unique(np.concatenate([positives, negatives]))[::-1]

    for target in far_targets:
        best_tar = 0.0
        best_threshold = None
        matched_far = None

        for threshold in thresholds:
            far = float(np.mean(negatives >= threshold))
            tar = float(np.mean(positives >= threshold))
            if far <= target and tar >= best_tar:
                best_tar = tar
                best_threshold = float(threshold)
                matched_far = far

        if best_threshold is None:
            fars, tars = [], []
            for threshold in thresholds:
                fars.append(float(np.mean(negatives >= threshold)))
                tars.append(float(np.mean(positives >= threshold)))
            for i in range(1, len(fars)):
                if fars[i] >= target >= fars[i - 1] or fars[i - 1] >= target >= fars[i]:
                    best_tar = linear_interp_logx(
                        target, fars[i - 1], fars[i], tars[i - 1], tars[i]
                    )
                    matched_far = target
                    best_threshold = float(thresholds[min(i, len(thresholds) - 1)])
                    break
            else:
                best_tar = tars[-1] if fars and fars[-1] <= target else 0.0
                matched_far = fars[-1] if fars else None
                best_threshold = float(thresholds[-1]) if len(thresholds) else None

        result[target] = {
            "tar": best_tar,
            "far": matched_far if matched_far is not None else 0.0,
            "frr": 1.0 - best_tar,
            "threshold": best_threshold,
        }
    return result


def compute_eer(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> Dict[str, Optional[float]]:
    """Equal Error Rate: operating point where FAR ≈ FRR."""
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return {"eer": None, "threshold": None}

    positives = np.asarray(positive_scores, dtype=np.float64)
    negatives = np.asarray(negative_scores, dtype=np.float64)
    all_scores = np.unique(np.concatenate([positives, negatives]))

    best_diff = float("inf")
    best_eer = None
    best_threshold = None
    for threshold in all_scores:
        frr = float(np.mean(positives < threshold))
        far = float(np.mean(negatives >= threshold))
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_eer = 0.5 * (far + frr)
            best_threshold = float(threshold)
    return {"eer": best_eer, "threshold": best_threshold}


def compute_best_threshold_metrics(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> Dict[str, float]:
    """Search threshold maximizing Accuracy on paired verification scores."""
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return {}

    positives = np.sort(np.asarray(positive_scores, dtype=np.float64))
    negatives = np.sort(np.asarray(negative_scores, dtype=np.float64))
    positive_total = len(positives)
    negative_total = len(negatives)
    candidates = np.unique(np.concatenate([positives, negatives]))

    tp = positive_total - np.searchsorted(positives, candidates, side="left")
    fn = positive_total - tp
    tn = np.searchsorted(negatives, candidates, side="left")
    fp = negative_total - tn
    total = positive_total + negative_total
    accuracy = (tp + tn) / max(1, total)
    best_index = int(np.argmax(accuracy))
    return {
        "accuracy": float(accuracy[best_index]),
        "tar": float(tp[best_index] / max(1, positive_total)),
        "far": float(fp[best_index] / max(1, negative_total)),
        "frr": float(fn[best_index] / max(1, positive_total)),
        "threshold": float(candidates[best_index]),
    }


def compute_map_from_ranks(ranks: Sequence[int]) -> float:
    """mAP for single-relevant retrieval: AP_i = 1/rank_i, mAP = mean(AP)."""
    if not ranks:
        return 0.0
    ranks_arr = np.asarray(ranks, dtype=np.float64)
    return float(np.mean(1.0 / ranks_arr))


def _threshold_at_roc_index(thresholds: np.ndarray, roc_index: int, n_roc_points: int) -> Optional[float]:
    """Map ROC point index to sklearn threshold (len(thresholds) == n_roc_points - 1)."""
    if thresholds is None or len(thresholds) == 0:
        return None
    # sklearn roc_curve: thresholds[k] pairs with fpr[k+1], tpr[k+1]
    thresh_idx = min(max(int(roc_index) - 1, 0), len(thresholds) - 1)
    return float(thresholds[thresh_idx])


def summarize_verification_metrics_from_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    positive_count: int,
    negative_count: int,
    far_targets: Sequence[float] = FAR_TARGETS_11_DISPLAY,
    auc: Optional[float] = None,
    positive_scores: Optional[np.ndarray] = None,
    negative_scores: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Fast 1:1 metrics from a single sklearn ``roc_curve`` pass (IJB-scale pairs).

    Avoids re-scanning millions of unique thresholds. Use this for ``-P`` plot-only replay.
    """
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    fnr = 1.0 - tpr

    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float(0.5 * (fpr[eer_idx] + fnr[eer_idx]))
    eer_threshold = _threshold_at_roc_index(thresholds, eer_idx, len(fpr))

    fpr_flip = np.flipud(fpr)
    tpr_flip = np.flipud(tpr)
    fnr_flip = np.flipud(fnr)
    thresholds_flip = np.flipud(thresholds) if len(thresholds) else thresholds

    tar_at_far = {}
    frr_at_far = {}
    far_at_far = {}
    threshold_at_far = {}
    tar_at_far_detail = {}
    for target in far_targets:
        key = far_label(target)
        idx = int(np.argmin(np.abs(fpr_flip - target)))
        tar = float(tpr_flip[idx])
        frr = float(fnr_flip[idx])
        achieved_far = float(fpr_flip[idx])
        orig_idx = len(fpr) - 1 - idx
        thresh = _threshold_at_roc_index(thresholds, orig_idx, len(fpr))
        tar_at_far[key] = tar
        frr_at_far[key] = frr
        far_at_far[key] = achieved_far
        threshold_at_far[key] = thresh
        tar_at_far_detail[key] = {"tar": tar, "far": achieved_far, "frr": frr, "threshold": thresh}

    best_info = {}
    # Best-threshold search needs unique() over all scores — skip on IJB-scale pair counts.
    if (
        positive_scores is not None
        and negative_scores is not None
        and len(negative_scores) <= 500_000
    ):
        best_info = compute_best_threshold_metrics(positive_scores, negative_scores)

    return {
        "positive_pairs": int(positive_count),
        "negative_pairs": int(negative_count),
        "tar_at_far": tar_at_far,
        "frr_at_far": frr_at_far,
        "far_at_far": far_at_far,
        "threshold_at_far": threshold_at_far,
        "tar_at_far_detail": tar_at_far_detail,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "best_threshold": best_info,
        "auc": auc,
    }


def summarize_verification_metrics(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    far_targets: Sequence[float] = FAR_TARGETS_11_DISPLAY,
    auc: Optional[float] = None,
) -> Dict[str, object]:
    """Full 1:1 verification report dict for IJB / enterprise acceptance."""
    tar_detail = compute_tar_at_far(positive_scores, negative_scores, far_targets)
    eer_info = compute_eer(positive_scores, negative_scores)
    best_info = compute_best_threshold_metrics(positive_scores, negative_scores)

    tar_at_far = {}
    frr_at_far = {}
    far_at_far = {}
    threshold_at_far = {}
    for target in far_targets:
        key = far_label(target)
        detail = tar_detail.get(target)
        if detail is None:
            tar_at_far[key] = frr_at_far[key] = far_at_far[key] = threshold_at_far[key] = None
        else:
            tar_at_far[key] = detail["tar"]
            frr_at_far[key] = detail["frr"]
            far_at_far[key] = detail["far"]
            threshold_at_far[key] = detail["threshold"]

    return {
        "positive_pairs": len(positive_scores),
        "negative_pairs": len(negative_scores),
        "tar_at_far": tar_at_far,
        "frr_at_far": frr_at_far,
        "far_at_far": far_at_far,
        "threshold_at_far": threshold_at_far,
        "tar_at_far_detail": {far_label(t): tar_detail.get(t) for t in far_targets},
        "eer": eer_info["eer"],
        "eer_threshold": eer_info["threshold"],
        "best_threshold": best_info,
        "auc": auc,
    }


def summarize_identification_metrics(
    rank1: float,
    rank5: float,
    rank10: float,
    map_score: float,
    enrolled_probe_count: int,
    tpir_detail: Dict[float, Optional[Dict[str, float]]],
    fpir_targets: Sequence[float] = FPIR_TARGETS_1N_DISPLAY,
) -> Dict[str, object]:
    """Full 1:N identification report dict (closed-set + open-set)."""
    tpir_at_fpir = {}
    fnir_at_fpir = {}
    fpir_at_fpir = {}
    threshold_at_fpir = {}
    for target in fpir_targets:
        key = far_label(target)
        detail = tpir_detail.get(target)
        if detail is None:
            tpir_at_fpir[key] = fnir_at_fpir[key] = fpir_at_fpir[key] = threshold_at_fpir[key] = None
        else:
            tpir_at_fpir[key] = detail["tar"]
            fnir_at_fpir[key] = detail["frr"]
            fpir_at_fpir[key] = detail["far"]
            threshold_at_fpir[key] = detail["threshold"]

    return {
        "enrolled_probe_count": enrolled_probe_count,
        "rank1": rank1,
        "rank5": rank5,
        "rank10": rank10,
        "map": map_score,
        "tpir_at_fpir": tpir_at_fpir,
        "fnir_at_fpir": fnir_at_fpir,
        "fpir_at_fpir": fpir_at_fpir,
        "threshold_at_fpir": threshold_at_fpir,
        "tpir_at_fpir_detail": {far_label(t): tpir_detail.get(t) for t in fpir_targets},
    }


def _fmt_rate(val: Optional[float], as_pct: bool = False) -> str:
    if val is None:
        return "N/A"
    if as_pct:
        return "%.4f%%" % (val * 100.0)
    return "%.6f" % val


def _parse_far_key(key: str) -> float:
    try:
        return float(key)
    except ValueError:
        return 0.0


def format_verification_table(metrics: Dict[str, object], method_name: str = "model") -> str:
    """Markdown table: TAR / FRR / Threshold @ FAR + EER + AUC."""
    rows = []
    far_keys = sorted(metrics.get("tar_at_far", {}).keys(), key=_parse_far_key)
    for key in far_keys:
        rows.append(
            {
                "FAR": key,
                "TAR": metrics["tar_at_far"].get(key),
                "FRR": metrics["frr_at_far"].get(key),
                "Threshold": metrics["threshold_at_far"].get(key),
            }
        )
    import pandas as pd

    df = pd.DataFrame(rows).set_index("FAR")
    summary = pd.DataFrame(
        {
            "EER": [_fmt_rate(metrics.get("eer"))],
            "EER_Threshold": [_fmt_rate(metrics.get("eer_threshold"))],
            "AUC": [_fmt_rate(metrics.get("auc"))],
            "Pos_Pairs": [metrics.get("positive_pairs")],
            "Neg_Pairs": [metrics.get("negative_pairs")],
        },
        index=[method_name],
    )
    try:
        return "TAR / FRR / Threshold @ FAR:\n" + df.to_markdown() + "\n\nSummary:\n" + summary.to_markdown()
    except ImportError:
        return "TAR / FRR / Threshold @ FAR:\n" + df.to_string() + "\n\nSummary:\n" + summary.to_string()


def format_identification_table(metrics: Dict[str, object], gallery_name: str = "gallery") -> str:
    """Markdown table: Rank / mAP + TPIR / FNIR / FPIR @ FPIR."""
    import pandas as pd

    closed = pd.DataFrame(
        {
            "Rank-1": [_fmt_rate(metrics.get("rank1"), as_pct=True)],
            "Rank-5": [_fmt_rate(metrics.get("rank5"), as_pct=True)],
            "Rank-10": [_fmt_rate(metrics.get("rank10"), as_pct=True)],
            "mAP": [_fmt_rate(metrics.get("map"), as_pct=True)],
            "Enrolled_Probes": [metrics.get("enrolled_probe_count")],
        },
        index=[gallery_name],
    )
    open_rows = []
    for key in sorted(metrics.get("tpir_at_fpir", {}).keys(), key=_parse_far_key):
        open_rows.append(
            {
                "FPIR": key,
                "TPIR": metrics["tpir_at_fpir"].get(key),
                "FNIR(Miss)": metrics["fnir_at_fpir"].get(key),
                "FPIR(achieved)": metrics["fpir_at_fpir"].get(key),
                "Threshold": metrics["threshold_at_fpir"].get(key),
            }
        )
    open_df = pd.DataFrame(open_rows).set_index("FPIR")
    try:
        return "Closed-set (Rank / mAP):\n" + closed.to_markdown() + "\n\nOpen-set (TPIR / FNIR / FPIR):\n" + open_df.to_markdown()
    except ImportError:
        return "Closed-set (Rank / mAP):\n" + closed.to_string() + "\n\nOpen-set (TPIR / FNIR / FPIR):\n" + open_df.to_string()
