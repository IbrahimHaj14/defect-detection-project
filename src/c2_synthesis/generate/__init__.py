"""Defect-generation utilities for C2."""

from src.c2_synthesis.generate.generate_defects import (
    GenerationCandidate,
    GenerationResult,
    LFSAcceptanceRateError,
    assert_pixel_composite_invariant,
    generate_and_filter,
    save_samples_grid,
)
from src.c2_synthesis.generate.low_fidelity_selection import (
    LFSBatchResult,
    LFSDecision,
    filter_batch,
    lfs_score,
)

__all__ = [
    "GenerationCandidate",
    "GenerationResult",
    "LFSAcceptanceRateError",
    "LFSBatchResult",
    "LFSDecision",
    "assert_pixel_composite_invariant",
    "filter_batch",
    "generate_and_filter",
    "lfs_score",
    "save_samples_grid",
]
