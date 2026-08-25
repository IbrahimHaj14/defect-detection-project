"""C3 Phase 1 acceptance: deterministic mask grounding and report schema."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from jsonschema import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c3_explanation.grounding.bridge import (  # noqa: E402
    REPORT_SCHEMA_PATH,
    build_skeleton_report,
    ground_mask,
)
from src.c3_explanation.grounding.severity_rules import assign_severity  # noqa: E402
from src.c3_explanation.utils.json_io import validate_against_schema  # noqa: E402

SPLITS_PATH = REPO_ROOT / "data" / "splits" / "mvtec_ad_splits.json"


def _load_config() -> dict[str, object]:
    with REPORT_SCHEMA_PATH.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    assert isinstance(config, dict)
    return config


def _assert_close(actual: float, expected: float) -> None:
    assert np.isclose(actual, expected, rtol=0.0, atol=1e-12), (
        f"Expected {expected}, got {actual}"
    )


def _verify_threshold_contract(config: dict[str, object]) -> None:
    assert assign_severity(0.999, "scratch", config) == "minor"
    assert assign_severity(1.0, "scratch", config) == "moderate"
    assert assign_severity(4.999, "scratch", config) == "moderate"
    assert assign_severity(5.0, "scratch", config) == "severe"
    assert "none" not in {
        assign_severity(area, "scratch", config) for area in (0.0, 1.0, 5.0)
    }

    overridden = copy.deepcopy(config)
    overridden["severity_thresholds"]["per_defect_type"]["scratch"] = {
        "minor_upper_exclusive": 0.25,
        "moderate_upper_exclusive": 2.0,
    }
    assert assign_severity(0.5, "scratch", overridden) == "moderate"
    assert assign_severity(2.0, "scratch", overridden) == "severe"
    assert assign_severity(0.5, "crack", overridden) == "minor"


def _verify_threshold_binarisation(config: dict[str, object]) -> None:
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[0, 0] = 127
    mask[5, 5] = 128
    facts = ground_mask(mask, mask.shape, "scratch", config)
    assert facts["bounding_box"] == [5, 5, 1, 1]
    assert facts["centroid"] == [5.0, 5.0]
    _assert_close(facts["area_pct"], 100.0 / 36.0)
    assert facts["region"] == "lower-right quadrant"


def _verify_synthetic_masks(config: dict[str, object]) -> None:
    corner = np.zeros((9, 9), dtype=np.uint8)
    corner[0:2, 0:3] = 255
    corner_facts = ground_mask(corner, (9, 9, 3), "scratch", config)
    assert corner_facts["bounding_box"] == [0, 0, 3, 2]
    assert corner_facts["centroid"] == [1.0, 0.5]
    _assert_close(corner_facts["area_pct"], 100.0 * 6.0 / 81.0)
    assert corner_facts["region"] == "upper-left quadrant"

    centred = np.zeros((9, 9), dtype=np.uint8)
    centred[3:6, 3:6] = 255
    centre_facts = ground_mask(centred, centred.shape, "scratch", config)
    assert centre_facts["bounding_box"] == [3, 3, 3, 3]
    assert centre_facts["centroid"] == [4.0, 4.0]
    assert centre_facts["region"] == "centre"

    near_full = np.zeros((10, 12), dtype=np.uint8)
    near_full[1:9, 1:11] = 255
    full_facts = ground_mask(near_full, near_full.shape, "scratch", config)
    assert full_facts["bounding_box"] == [1, 1, 10, 8]
    assert full_facts["centroid"] == [5.5, 4.5]
    _assert_close(full_facts["area_pct"], 100.0 * 80.0 / 120.0)
    assert full_facts["region"] == "centre"

    empty = np.zeros((9, 9), dtype=np.uint8)
    empty_facts = ground_mask(empty, empty.shape, "scratch", config)
    assert empty_facts == {
        "defect_present": False,
        "bounding_box": None,
        "centroid": None,
        "area_pct": 0.0,
        "region": None,
        "severity_level": "none",
    }
    empty_report = build_skeleton_report(empty_facts, "scratch")
    validate_against_schema(empty_report, REPORT_SCHEMA_PATH)
    assert empty_report["severity"]["level"] == "none"
    invalid_empty = copy.deepcopy(empty_report)
    invalid_empty["location"]["region"] = "none"
    try:
        validate_against_schema(invalid_empty, REPORT_SCHEMA_PATH)
    except ValidationError:
        pass
    else:
        raise AssertionError('Empty-mask region string "none" must be rejected')


def _verify_multi_blob_contract(config: dict[str, object]) -> None:
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[0:2, 0:2] = 255
    mask[8:11, 8:12] = 255
    facts = ground_mask(mask, mask.shape, "scratch", config)
    assert facts["bounding_box"] == [0, 0, 12, 11], "BBox must cover the union"
    assert facts["centroid"] == [9.5, 9.0], "Centroid must use the largest blob"
    assert facts["region"] == "lower-right quadrant"
    _assert_close(facts["area_pct"], 100.0 * 16.0 / 144.0)


def _verify_schema_and_ownership(config: dict[str, object]) -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:2, 0:2] = 255
    facts = ground_mask(mask, mask.shape, "scratch", config)
    report = build_skeleton_report(facts, "scratch")
    validate_against_schema(report, REPORT_SCHEMA_PATH)
    assert report["defect_present"] == facts["defect_present"]
    assert report["location"] == {
        "region": facts["region"],
        "bounding_box": facts["bounding_box"],
        "centroid": facts["centroid"],
    }
    assert report["severity"]["level"] == facts["severity_level"]
    assert report["severity"]["affected_area_pct"] == facts["area_pct"]
    assert report["severity"]["rationale"] == ""
    assert report["description"] == ""
    assert report["recommended_action"] == {"action": "inspect", "reason": ""}
    assert report["confidence"] == ""

    invalid_present = copy.deepcopy(report)
    invalid_present["location"]["region"] = None
    try:
        validate_against_schema(invalid_present, REPORT_SCHEMA_PATH)
    except ValidationError:
        pass
    else:
        raise AssertionError("Defect-present reports must reject null geometry")

    ownership = config["field_ownership"]
    assert ownership["defect_type"] == {"corpus": "bridge", "inference": "brain"}
    for field in (
        "defect_present",
        "location.region",
        "location.bounding_box",
        "location.centroid",
        "severity.level",
        "severity.affected_area_pct",
    ):
        assert ownership[field] == "bridge"
    for field in (
        "severity.rationale",
        "description",
        "recommended_action.action",
        "recommended_action.reason",
        "confidence",
    ):
        assert ownership[field] == "brain"

    vocabularies = config["allowed_enumerations"]["defect_type"]
    assert set(vocabularies) == {"mvtec_ad", "ecf"}
    schema_union = set(config["properties"]["defect_type"]["enum"])
    assert schema_union == set(vocabularies["mvtec_ad"]) | set(vocabularies["ecf"])


def _first_real_c1_mask() -> tuple[Path, Path, str]:
    with SPLITS_PATH.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    for category in manifest.values():
        for sample in category["splits"]["test"]:
            if sample["is_anomalous"] and sample["mask_path"]:
                return (
                    REPO_ROOT / sample["image_path"],
                    REPO_ROOT / sample["mask_path"],
                    sample["label"],
                )
    raise AssertionError("No real anomalous C1 mask exists in the MVTec manifest")


def _verify_real_c1_mask(config: dict[str, object]) -> Path:
    image_path, mask_path, defect_type = _first_real_c1_mask()
    assert image_path.is_file() and mask_path.is_file()
    with Image.open(image_path) as image_file:
        image_shape = (image_file.height, image_file.width)
    with Image.open(mask_path) as mask_file:
        assert mask_file.mode == "L", "C1 mask must be single-channel"
        mask = np.asarray(mask_file)
    facts = ground_mask(mask, image_shape, defect_type, config)
    assert facts["defect_present"] is True
    report = build_skeleton_report(facts, defect_type)
    validate_against_schema(report, REPORT_SCHEMA_PATH)
    return mask_path


def main() -> None:
    config = _load_config()
    _verify_threshold_contract(config)
    _verify_threshold_binarisation(config)
    _verify_synthetic_masks(config)
    _verify_multi_blob_contract(config)
    _verify_schema_and_ownership(config)
    real_mask_path = _verify_real_c1_mask(config)

    print("C3 Phase 1 verification passed")
    print("Binary mask contract (>127): passed")
    print("Synthetic masks (corner/centre/near-full/empty): passed")
    print("Multi-blob union bbox/largest-component centroid: passed")
    print("3x3 region labelling: passed")
    print("Config-driven severity and per-type override: passed")
    print("Skeleton schema and field ownership: passed")
    print("Conditional null geometry and dataset vocabularies: passed")
    print(f"Real C1 mask: {real_mask_path.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
