"""Optional offline description-only enrichment interface for C3."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any


def enrich_description(
    report: Mapping[str, Any],
    *,
    enabled: bool = False,
    rephrase: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Return a copied report, changing only ``description`` when enabled.

    Phase 2 provides no enrichment model. Enabling enrichment therefore
    requires an explicitly supplied offline callable and never adds an
    inference-time dependency.
    """

    original = copy.deepcopy(dict(report))
    enriched = copy.deepcopy(original)
    if not enabled:
        return enriched
    if rephrase is None:
        raise ValueError("Enabled enrichment requires an offline rephrasing callable")

    description = rephrase(str(original["description"]))
    if not isinstance(description, str) or not description.strip():
        raise ValueError("The enrichment callable must return non-empty text")
    enriched["description"] = description

    original_without_description = copy.deepcopy(original)
    enriched_without_description = copy.deepcopy(enriched)
    original_without_description.pop("description", None)
    enriched_without_description.pop("description", None)
    if original_without_description != enriched_without_description:
        raise AssertionError("Enrichment modified a field other than description")
    return enriched
