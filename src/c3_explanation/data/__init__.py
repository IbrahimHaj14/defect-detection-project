"""Deterministic C3 report-corpus construction."""

from src.c3_explanation.data.corpus_split import (
    assert_no_c2_synthetic_paths,
    deterministic_stratified_split,
    test_count_for_stratum,
)
from src.c3_explanation.data.enrichment import enrich_description
from src.c3_explanation.data.report_builder import (
    CorpusBuildResult,
    build_corpus,
    build_report,
    load_dataset_config,
)

__all__ = [
    "assert_no_c2_synthetic_paths",
    "build_corpus",
    "build_report",
    "CorpusBuildResult",
    "deterministic_stratified_split",
    "enrich_description",
    "load_dataset_config",
    "test_count_for_stratum",
]
