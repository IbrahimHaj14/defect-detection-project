"""Deterministic conversion of C1 binary masks into grounded C3 facts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.c3_explanation.grounding.severity_rules import assign_severity
from src.c3_explanation.utils.json_io import validate_against_schema

REPORT_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "report_schema.yaml"

_REGION_LABELS = (
    ("upper-left quadrant", "top edge", "upper-right quadrant"),
    ("left edge", "centre", "right edge"),
    ("lower-left quadrant", "bottom edge", "lower-right quadrant"),
)


def _image_hw(image_shape: Sequence[int]) -> tuple[int, int]:
    if len(image_shape) < 2:
        raise ValueError("image_shape must provide height and width")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image height and width must be positive")
    return height, width


def _binary_mask(mask: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("C1 mask must be a single-channel 2D array")
    if array.shape != expected_shape:
        raise ValueError(
            f"Mask shape {array.shape} does not match image shape {expected_shape}"
        )
    # Project-wide C1 contract: only source values strictly above 127 are defect.
    return array if array.dtype == np.bool_ else array > 127


def _largest_component_centroid(binary: np.ndarray) -> list[float]:
    """Return [cx, cy] for the largest 8-connected component.

    Components are discovered in row-major order; retaining the first component
    on an equal-area tie makes the result deterministic.
    """

    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    largest: list[tuple[int, int]] = []
    for start_y, start_x in np.argwhere(binary):
        y0, x0 = int(start_y), int(start_x)
        if visited[y0, x0]:
            continue
        visited[y0, x0] = True
        stack = [(y0, x0)]
        component: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    neighbour_y, neighbour_x = y + dy, x + dx
                    if (
                        0 <= neighbour_y < height
                        and 0 <= neighbour_x < width
                        and binary[neighbour_y, neighbour_x]
                        and not visited[neighbour_y, neighbour_x]
                    ):
                        visited[neighbour_y, neighbour_x] = True
                        stack.append((neighbour_y, neighbour_x))
        if len(component) > len(largest):
            largest = component

    coordinates = np.asarray(largest, dtype=np.float64)
    return [float(coordinates[:, 1].mean()), float(coordinates[:, 0].mean())]


def _region_from_centroid(
    centroid: Sequence[float],
    image_shape: tuple[int, int],
) -> str:
    height, width = image_shape
    cx, cy = float(centroid[0]), float(centroid[1])
    column = min(int((3.0 * cx) / width), 2)
    row = min(int((3.0 * cy) / height), 2)
    return _REGION_LABELS[row][column]


def ground_mask(
    mask: np.ndarray,
    image_shape: Sequence[int],
    defect_type: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert one C1 mask into deterministic, parameter-free grounded facts.

    Mask values are binarised with the exact project contract ``> 127``. The
    bounding box contains the union of all blobs, while the centroid is the
    pixel-coordinate mean of the largest 8-connected component.
    """

    if not isinstance(defect_type, str) or not defect_type:
        raise ValueError("defect_type must be a non-empty string")
    height, width = _image_hw(image_shape)
    binary = _binary_mask(mask, (height, width))
    positive_count = int(np.count_nonzero(binary))

    if positive_count == 0:
        return {
            "defect_present": False,
            "bounding_box": None,
            "centroid": None,
            "area_pct": 0.0,
            "region": None,
            "severity_level": "none",
        }

    ys, xs = np.nonzero(binary)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bounding_box = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
    centroid = _largest_component_centroid(binary)
    area_pct = float(np.clip(100.0 * positive_count / (height * width), 0.0, 100.0))
    return {
        "defect_present": True,
        "bounding_box": bounding_box,
        "centroid": centroid,
        "area_pct": area_pct,
        "region": _region_from_centroid(centroid, (height, width)),
        "severity_level": assign_severity(area_pct, defect_type, config),
    }


def build_skeleton_report(facts: Mapping[str, Any], defect_type: str) -> dict[str, Any]:
    """Build and validate a report with Bridge facts and Brain placeholders.

    ``defect_type`` is the Bridge-owned known label during corpus construction;
    at inference the Brain owns it and must select from the configured known
    vocabulary. ``inspect`` is only an internal schema-valid Phase 1
    placeholder, not a Bridge inference. Phase 2 corpus construction and Phase
    4 final generation must replace/populate it. All free-text Brain fields
    remain empty for the Phase 1 skeleton.
    """

    required_facts = {
        "defect_present",
        "bounding_box",
        "centroid",
        "area_pct",
        "region",
        "severity_level",
    }
    missing = required_facts.difference(facts)
    if missing:
        raise ValueError(f"Grounding facts are missing fields: {sorted(missing)}")
    report = {
        "defect_present": bool(facts["defect_present"]),
        "defect_type": defect_type,
        "location": {
            "region": facts["region"],
            "bounding_box": facts["bounding_box"],
            "centroid": facts["centroid"],
        },
        "severity": {
            "level": facts["severity_level"],
            "affected_area_pct": float(facts["area_pct"]),
            "rationale": "",
        },
        "description": "",
        "recommended_action": {"action": "inspect", "reason": ""},
        "confidence": "",
    }
    validate_against_schema(report, REPORT_SCHEMA_PATH)
    return report
