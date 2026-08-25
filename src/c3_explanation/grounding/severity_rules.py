"""Auditable, configuration-driven severity assignment for C3."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def _threshold_config(
    defect_type: str,
    config: Mapping[str, Any],
) -> tuple[float, float]:
    try:
        severity_config = config["severity_thresholds"]
        global_config = severity_config["global"]
        overrides = severity_config.get("per_defect_type", {})
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Config requires severity_thresholds.global and "
            "severity_thresholds.per_defect_type"
        ) from error

    if not isinstance(global_config, Mapping) or not isinstance(overrides, Mapping):
        raise ValueError("Severity threshold blocks must be mappings")
    defect_override = overrides.get(defect_type, {})
    if not isinstance(defect_override, Mapping):
        raise ValueError(f"Severity override for {defect_type!r} must be a mapping")

    merged = {**global_config, **defect_override}
    try:
        minor_upper = float(merged["minor_upper_exclusive"])
        moderate_upper = float(merged["moderate_upper_exclusive"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Each effective severity policy requires numeric "
            "minor_upper_exclusive and moderate_upper_exclusive cutoffs"
        ) from error
    if not (0.0 <= minor_upper <= moderate_upper <= 100.0):
        raise ValueError(
            "Severity cutoffs must satisfy 0 <= minor <= moderate <= 100"
        )
    return minor_upper, moderate_upper


def assign_severity(
    area_pct: float,
    defect_type: str,
    config: Mapping[str, Any],
) -> str:
    """Return ``minor``, ``moderate``, or ``severe`` from YAML cutoffs.

    The global thresholds and optional per-defect-type overrides are read only
    from ``config``. Empty masks are handled by the Bridge as severity ``none``;
    this function assigns severity only when a defect is present.
    """

    try:
        area = float(area_pct)
    except (TypeError, ValueError) as error:
        raise ValueError("area_pct must be numeric") from error
    if not math.isfinite(area) or not 0.0 <= area <= 100.0:
        raise ValueError("area_pct must be finite and lie in [0, 100]")
    if not isinstance(defect_type, str) or not defect_type:
        raise ValueError("defect_type must be a non-empty string")

    minor_upper, moderate_upper = _threshold_config(defect_type, config)
    if area < minor_upper:
        return "minor"
    if area < moderate_upper:
        return "moderate"
    return "severe"
