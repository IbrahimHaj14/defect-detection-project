"""C3 Phase 2 acceptance: deterministic real-defect report corpora."""

from __future__ import annotations

import copy
import inspect
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c3_explanation.data.corpus_split import (  # noqa: E402
    assert_no_c2_synthetic_paths,
    deterministic_stratified_split,
    test_count_for_stratum,
)
from src.c3_explanation.data.enrichment import enrich_description  # noqa: E402
from src.c3_explanation.data.report_builder import (  # noqa: E402
    MEMBERSHIP_FIELDS,
    build_corpus,
    build_report,
    load_dataset_config,
)
from src.c3_explanation.utils.json_io import (  # noqa: E402
    load_json,
    validate_against_schema,
)

MVTEC_CONFIG_PATH = (
    REPO_ROOT / "src" / "c3_explanation" / "configs" / "datasets" / "mvtec.yaml"
)
ECF_CONFIG_PATH = (
    REPO_ROOT / "src" / "c3_explanation" / "configs" / "datasets" / "ecf.yaml"
)


def _expect_error(error_type: type[BaseException], callback: Any) -> None:
    try:
        callback()
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}")


def _filtered_config(path: Path, class_name: str) -> dict[str, Any]:
    config = load_dataset_config(path)
    selected = copy.deepcopy(config)
    selected["classes"] = {class_name: copy.deepcopy(config["classes"][class_name])}
    return selected


def _verify_split_policy() -> None:
    assert test_count_for_stratum(1, 0.2) == 0
    assert test_count_for_stratum(2, 0.2) == 1
    assert test_count_for_stratum(5, 0.5) == 3, "Must use half-up, not bankers rounding"
    assert test_count_for_stratum(20, 0.2) == 4

    samples = [
        {
            "uid": f"multi-{index}",
            "class": "product",
            "defect_type": "defect",
            "image_path": f"data/processed/example/{index}.png",
            "mask_path": f"data/processed/example/{index}_mask.png",
        }
        for index in range(5, 0, -1)
    ]
    samples.append(
        {
            "uid": "singleton",
            "class": "product",
            "defect_type": "singleton_defect",
            "image_path": "data/processed/example/singleton.png",
            "mask_path": "data/processed/example/singleton_mask.png",
        }
    )
    first = deterministic_stratified_split(
        samples, seed=42, train_fraction=0.8, test_fraction=0.2
    )
    second = deterministic_stratified_split(
        list(reversed(samples)), seed=42, train_fraction=0.8, test_fraction=0.2
    )
    assert first == second
    assert any(item["uid"] == "singleton" for item in first["train"])
    assert all(item["uid"] != "singleton" for item in first["test"])
    assert [item["uid"] for item in first["train"]] == sorted(
        item["uid"] for item in first["train"]
    )
    assert [item["uid"] for item in first["test"]] == sorted(
        item["uid"] for item in first["test"]
    )

    c2_member = copy.deepcopy(samples[0])
    c2_member["image_path"] = "outputs/synthetic/c2-example.png"
    _expect_error(AssertionError, lambda: assert_no_c2_synthetic_paths([c2_member]))


def _verify_enrichment_interface() -> None:
    source = {
        "description": "Original description.",
        "location": {"bounding_box": [1, 2, 3, 4]},
        "severity": {"level": "minor"},
    }
    disabled = enrich_description(source, enabled=False)
    assert disabled == source and disabled is not source
    enabled = enrich_description(
        source,
        enabled=True,
        rephrase=lambda text: f"{text} Offline rephrasing.",
    )
    assert enabled["description"] != source["description"]
    assert enabled["location"] == source["location"]
    assert enabled["severity"] == source["severity"]
    _expect_error(ValueError, lambda: enrich_description(source, enabled=True))


def _verify_config_contracts(
    mvtec_config: dict[str, Any],
    ecf_config: dict[str, Any],
) -> None:
    assert mvtec_config["processed_dir"] == "data/processed/mvtec_ad"
    assert mvtec_config["splits_path"] == "data/splits/mvtec_ad_splits.json"
    ecf_processed_dir = Path(ecf_config["processed_dir"])
    assert ecf_processed_dir.parts[:2] == ("data", "processed")
    assert (REPO_ROOT / ecf_processed_dir).is_dir()
    assert ecf_config["splits_path"] == "data/splits/ecf_splits.json"
    assert set(ecf_config["excluded_classes"]) == {
        "pseudo_broken_solder",
        "Serious defect",
        "solder_bead",
    }
    for config, expected_pairs in ((mvtec_config, 73), (ecf_config, 11)):
        pairs = [
            (class_name, defect_type)
            for class_name, class_config in config["classes"].items()
            for defect_type in class_config["defect_types"]
        ]
        assert len(pairs) == expected_pairs
        actions = {
            template["action_template"]["action"]
            for class_config in config["classes"].values()
            for template in class_config["defect_types"].values()
        }
        assert actions == {"inspect"}
        reasons = [
            template["action_template"]["reason_template"]
            for class_config in config["classes"].values()
            for template in class_config["defect_types"].values()
        ]
        assert all("isolate and inspect" not in reason.lower() for reason in reasons)


def _verify_explicit_class_and_dataset_vocabularies(
    mvtec_config: dict[str, Any],
    ecf_config: dict[str, Any],
) -> None:
    assert list(inspect.signature(build_report).parameters) == [
        "image_path",
        "mask_path",
        "class_name",
        "defect_type",
        "config",
    ]
    mvtec_manifest = load_json(REPO_ROOT / mvtec_config["splits_path"])
    sample = next(
        item
        for item in mvtec_manifest["bottle"]["splits"]["test"]
        if item["is_anomalous"] and item["label"] == "broken_large"
    )
    report = build_report(
        sample["image_path"],
        sample["mask_path"],
        "bottle",
        "broken_large",
        mvtec_config,
    )
    assert report["defect_type"] == "broken_large"
    assert report["recommended_action"]["action"] == "inspect"
    assert report["recommended_action"]["reason"]

    _expect_error(
        ValueError,
        lambda: build_report(
            sample["image_path"],
            sample["mask_path"],
            "cable",
            "broken_large",
            mvtec_config,
        ),
    )
    _expect_error(
        ValueError,
        lambda: build_report(
            sample["image_path"],
            sample["mask_path"],
            "bottle",
            "1.scratch",
            mvtec_config,
        ),
    )

    ecf_manifest = load_json(REPO_ROOT / ecf_config["splits_path"])
    ecf_sample = next(
        item
        for item in ecf_manifest["1.scratch"]["splits"]["test"]
        if item["is_anomalous"] and item["mask_path"]
    )
    _expect_error(
        ValueError,
        lambda: build_report(
            ecf_sample["image_path"],
            ecf_sample["mask_path"],
            "1.scratch",
            "broken_large",
            ecf_config,
        ),
    )


def _verify_artifacts(result: Any, config: dict[str, Any]) -> dict[str, Any]:
    membership = load_json(result.membership_path)
    assert set(membership) == {
        "dataset_key",
        "seed",
        "train_fraction",
        "test_fraction",
        "stratify_by",
        "train",
        "test",
    }
    assert membership["dataset_key"] == config["dataset_key"]
    assert membership["seed"] == 42
    assert membership["train_fraction"] == 0.8
    assert membership["test_fraction"] == 0.2
    assert membership["stratify_by"] == ["class", "defect_type"]
    assert len(membership["train"]) == result.train_count
    assert len(membership["test"]) == result.test_count
    schema_path = REPO_ROOT / config["report_schema_path"]

    for split_name in ("train", "test"):
        members = membership[split_name]
        assert [item["uid"] for item in members] == sorted(
            item["uid"] for item in members
        )
        for member in members:
            assert set(member) == set(MEMBERSHIP_FIELDS)
            assert "\\" not in member["image_path"]
            assert "\\" not in member["mask_path"]
            assert not member["image_path"].startswith("outputs/synthetic")
            assert not member["mask_path"].startswith("outputs/synthetic")
            assert (REPO_ROOT / member["image_path"]).is_file()
            assert (REPO_ROOT / member["mask_path"]).is_file()

            base = result.output_dir / split_name / member["uid"]
            grounding = load_json(base.with_suffix(".grounding.json"))
            report = load_json(base.with_suffix(".report.json"))
            validate_against_schema(report, schema_path)
            assert grounding["dataset"] == config["dataset_key"]
            assert grounding["class"] == member["class"]
            assert report["defect_type"] == member["defect_type"]
            assert report["defect_present"] is True
            assert grounding["bounding_box"] == report["location"]["bounding_box"]
            assert grounding["centroid"] == report["location"]["centroid"]
            assert grounding["region"] == report["location"]["region"]
            assert grounding["area_pct"] == report["severity"]["affected_area_pct"]
            assert grounding["severity_level"] == report["severity"]["level"]
            assert report["description"]
            assert report["severity"]["rationale"]
            assert report["recommended_action"]["action"] == "inspect"
            assert report["recommended_action"]["reason"]
            assert report["confidence"] == config["report_templates"][
                "confidence_template"
            ]
            lowered_reason = report["recommended_action"]["reason"].lower()
            assert "rework" not in lowered_reason
            assert "scrap" not in lowered_reason
            assert "monitor" not in lowered_reason
    assert not list(result.output_dir.rglob("*.image.png"))
    assert_no_c2_synthetic_paths([*membership["train"], *membership["test"]])
    return membership


def _build_twice_and_compare(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    first = build_corpus(config["dataset_key"], config)
    first_bytes = first.membership_path.read_bytes()
    second = build_corpus(config["dataset_key"], config)
    second_bytes = second.membership_path.read_bytes()
    assert first_bytes == second_bytes, "Membership must be byte-for-byte deterministic"
    assert first.train_count == second.train_count
    assert first.test_count == second.test_count
    assert first.skipped == second.skipped
    return second, _verify_artifacts(second, config)


def main() -> None:
    _verify_split_policy()
    _verify_enrichment_interface()
    full_mvtec = load_dataset_config(MVTEC_CONFIG_PATH)
    full_ecf = load_dataset_config(ECF_CONFIG_PATH)
    _verify_config_contracts(full_mvtec, full_ecf)

    mvtec = _filtered_config(MVTEC_CONFIG_PATH, "bottle")
    ecf = _filtered_config(ECF_CONFIG_PATH, "1.scratch")
    _verify_explicit_class_and_dataset_vocabularies(mvtec, ecf)
    mvtec_result, mvtec_membership = _build_twice_and_compare(mvtec)
    ecf_result, ecf_membership = _build_twice_and_compare(ecf)

    assert mvtec_result.total_count == 63
    assert (mvtec_result.train_count, mvtec_result.test_count) == (51, 12)
    assert ecf_result.total_count == 105
    assert (ecf_result.train_count, ecf_result.test_count) == (84, 21)
    assert not mvtec_result.skipped and not ecf_result.skipped
    mvtec_test_strata = Counter(
        (item["class"], item["defect_type"]) for item in mvtec_membership["test"]
    )
    assert mvtec_test_strata == {
        ("bottle", "broken_large"): 4,
        ("bottle", "broken_small"): 4,
        ("bottle", "contamination"): 4,
    }
    assert Counter(
        (item["class"], item["defect_type"]) for item in ecf_membership["test"]
    ) == {("1.scratch", "1.scratch"): 21}

    print("C3 Phase 2 verification passed")
    print("Half-up stratified split and singleton handling: passed")
    print("Explicit class propagation and dataset vocabularies: passed")
    print("Rich deterministic membership and POSIX paths: passed")
    print("Schema/grounding agreement and placeholder replacement: passed")
    print("C2 synthetic exclusion and source-image references: passed")
    print("Disabled description-only enrichment interface: passed")
    print(
        f"MVTec bottle corpus: total={mvtec_result.total_count} "
        f"train={mvtec_result.train_count} test={mvtec_result.test_count} "
        f"skipped={len(mvtec_result.skipped)}"
    )
    print(
        f"ECF 1.scratch corpus: total={ecf_result.total_count} "
        f"train={ecf_result.train_count} test={ecf_result.test_count} "
        f"skipped={len(ecf_result.skipped)}"
    )
    print(f"MVTec test strata: {dict(sorted(mvtec_test_strata.items()))}")
    print("ECF test strata: {('1.scratch', '1.scratch'): 21}")


if __name__ == "__main__":
    main()
