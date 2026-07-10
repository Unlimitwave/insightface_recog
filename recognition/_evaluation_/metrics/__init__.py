"""Shared production-oriented face recognition evaluation metrics."""

from .production_metrics import (
    FAR_TARGETS_11,
    FPIR_TARGETS_1N,
    compute_best_threshold_metrics,
    compute_eer,
    compute_map_from_ranks,
    compute_tar_at_far,
    far_label,
    format_identification_table,
    format_verification_table,
)

__all__ = [
    "FAR_TARGETS_11",
    "FPIR_TARGETS_1N",
    "compute_best_threshold_metrics",
    "compute_eer",
    "compute_map_from_ranks",
    "compute_tar_at_far",
    "far_label",
    "format_identification_table",
    "format_verification_table",
]
