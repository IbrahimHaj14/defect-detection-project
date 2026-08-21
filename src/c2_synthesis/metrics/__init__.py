"""Generative-fidelity metrics for C2."""

from src.c2_synthesis.metrics.fid_kid import FidelityMetrics, compute_fid_kid
from src.c2_synthesis.metrics.lpips_diversity import mean_pairwise_lpips
from src.c2_synthesis.metrics.report import FidelityRow, read_fidelity_report, write_fidelity_report

__all__ = [
    "FidelityMetrics",
    "FidelityRow",
    "compute_fid_kid",
    "mean_pairwise_lpips",
    "read_fidelity_report",
    "write_fidelity_report",
]
