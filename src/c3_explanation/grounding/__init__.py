"""Deterministic C1-mask grounding for C3."""

from src.c3_explanation.grounding.bridge import (
    REPORT_SCHEMA_PATH,
    build_skeleton_report,
    ground_mask,
)
from src.c3_explanation.grounding.severity_rules import assign_severity

__all__ = [
    "REPORT_SCHEMA_PATH",
    "assign_severity",
    "build_skeleton_report",
    "ground_mask",
]
