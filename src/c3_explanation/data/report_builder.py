"""Build deterministic, schema-valid C3 reports from real C1 mask evidence."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from src.c3_explanation.data.corpus_split import (
    assert_no_c2_synthetic_paths,
    deterministic_stratified_split,
)
from src.c3_explanation.grounding.bridge import build_skeleton_report, ground_mask
from src.c3_explanation.utils.json_io import (
    load_json,
    save_json,
    validate_against_schema,
)
from src.c3_explanation.utils.logging_utils import get_logger

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "datasets"
MEMBERSHIP_FIELDS = ("uid", "class", "defect_type", "image_path", "mask_path")


@dataclass(frozen=True)
class CorpusBuildResult:
    dataset_key: str
    output_dir: Path
    membership_path: Path
    train_count: int
    test_count: int
    skipped: tuple[dict[str, str], ...]

    @property
    def total_count(self) -> int:
        return self.train_count + self.test_count


def _repo_path(path_value: str | Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _repo_relative_posix(path_value: str | Path) -> str:
    resolved = _repo_path(path_value)
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"Corpus source must be inside the repository: {resolved}") from error


def _assert_under_processed(path_value: str | Path, processed_dir: Path) -> None:
    resolved = _repo_path(path_value)
    try:
        resolved.relative_to(processed_dir)
    except ValueError as error:
        raise AssertionError(
            f"Real corpus source is outside configured processed_dir: {resolved}"
        ) from error


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate one Phase 2 dataset YAML."""

    config_path = _repo_path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Dataset configuration must be a mapping: {config_path}")
    required = {
        "dataset_key",
        "processed_dir",
        "splits_path",
        "output_dir",
        "report_schema_path",
        "seed",
        "train_fraction",
        "test_fraction",
        "enrichment",
        "excluded_classes",
        "report_templates",
        "classes",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Dataset configuration is missing keys: {sorted(missing)}")
    if not isinstance(config["classes"], dict) or not config["classes"]:
        raise ValueError("Dataset configuration requires a non-empty classes mapping")
    if bool(config["enrichment"].get("enabled", True)):
        raise ValueError("Phase 2 corpus enrichment must be disabled by default")
    return config


@lru_cache(maxsize=4)
def _load_yaml_mapping(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    with path.open("r", encoding="utf-8") as input_file:
        value = yaml.safe_load(input_file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def _schema_config(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    schema_path = _repo_path(str(config["report_schema_path"]))
    return schema_path, _load_yaml_mapping(str(schema_path))


def _template_for(
    config: Mapping[str, Any],
    class_name: str,
    defect_type: str,
) -> Mapping[str, Any]:
    classes = config["classes"]
    if class_name not in classes:
        raise ValueError(
            f"Class {class_name!r} is not configured for dataset {config['dataset_key']!r}"
        )
    defect_types = classes[class_name].get("defect_types", {})
    if defect_type not in defect_types:
        raise ValueError(
            f"Defect type {defect_type!r} is not configured for class "
            f"{class_name!r} in dataset {config['dataset_key']!r}"
        )

    _, schema = _schema_config(config)
    dataset_vocabularies = schema["allowed_enumerations"]["defect_type"]
    dataset_key = str(config["dataset_key"])
    if dataset_key not in dataset_vocabularies:
        raise ValueError(f"Schema has no defect vocabulary for dataset {dataset_key!r}")
    if defect_type not in dataset_vocabularies[dataset_key]:
        raise ValueError(
            f"Defect type {defect_type!r} is outside the {dataset_key!r} vocabulary"
        )
    return defect_types[defect_type]


def _format(template: str, values: Mapping[str, Any], field_name: str) -> str:
    try:
        rendered = template.format(**values)
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid {field_name} template: {error}") from error
    if not rendered.strip():
        raise ValueError(f"{field_name} template produced empty text")
    return rendered


def build_report(
    image_path: str | Path,
    mask_path: str | Path,
    class_name: str,
    defect_type: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a fully populated report for an explicit class and defect type."""

    template = _template_for(config, class_name, defect_type)
    processed_dir = _repo_path(str(config["processed_dir"]))
    _assert_under_processed(image_path, processed_dir)
    _assert_under_processed(mask_path, processed_dir)
    resolved_image = _repo_path(image_path)
    resolved_mask = _repo_path(mask_path)
    if not resolved_image.is_file():
        raise FileNotFoundError(f"Corpus image does not exist: {resolved_image}")
    if not resolved_mask.is_file():
        raise FileNotFoundError(f"Corpus mask does not exist: {resolved_mask}")

    with Image.open(resolved_image) as image_file:
        image_shape = (image_file.height, image_file.width)
    with Image.open(resolved_mask) as mask_file:
        mask = np.asarray(mask_file)

    _, schema = _schema_config(config)
    facts = ground_mask(mask, image_shape, defect_type, schema)
    report = build_skeleton_report(facts, defect_type)
    values = {
        "class_name": class_name,
        "defect_type": defect_type,
        "region": facts["region"],
        "affected_area_pct": float(facts["area_pct"]),
        "severity_level": facts["severity_level"],
    }
    action_template = template.get("action_template")
    if not isinstance(action_template, Mapping):
        raise ValueError(
            f"action_template for {class_name}/{defect_type} must be a mapping"
        )
    action = str(action_template.get("action", ""))
    if action != "inspect":
        raise ValueError(
            "Phase 2 corpus action policy permits only the approved 'inspect' action"
        )

    report["description"] = _format(
        str(template["description_template"]), values, "description"
    )
    report["severity"]["rationale"] = _format(
        str(config["report_templates"]["severity_rationale_template"]),
        values,
        "severity rationale",
    )
    # Unconditional replacement proves this value comes from the class-scoped
    # template, never by inheriting the Phase 1 skeleton placeholder.
    report["recommended_action"] = {
        "action": action,
        "reason": _format(
            str(action_template["reason_template"]), values, "action reason"
        ),
    }
    report["confidence"] = _format(
        str(config["report_templates"]["confidence_template"]),
        values,
        "confidence",
    )

    schema_path, _ = _schema_config(config)
    validate_against_schema(report, schema_path)
    return report


def _slug(value: str) -> str:
    rendered = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return rendered or "item"


def _uid_for(
    dataset_key: str,
    class_name: str,
    defect_type: str,
    image_path: str,
) -> str:
    identity = f"{dataset_key}|{class_name}|{defect_type}|{image_path}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"{_slug(class_name)}-{_slug(defect_type)}-{digest}"


def _membership_record(
    *,
    uid: str,
    class_name: str,
    defect_type: str,
    image_path: str,
    mask_path: str,
) -> dict[str, str]:
    return {
        "uid": uid,
        "class": class_name,
        "defect_type": defect_type,
        "image_path": image_path,
        "mask_path": mask_path,
    }


def _grounding_record(
    member: Mapping[str, str],
    dataset_key: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "uid": member["uid"],
        "dataset": dataset_key,
        "class": member["class"],
        "image_path": member["image_path"],
        "mask_path": member["mask_path"],
        "bounding_box": report["location"]["bounding_box"],
        "centroid": report["location"]["centroid"],
        "area_pct": report["severity"]["affected_area_pct"],
        "region": report["location"]["region"],
        "severity_level": report["severity"]["level"],
    }


def build_corpus(
    dataset: str,
    config: Mapping[str, Any],
) -> CorpusBuildResult:
    """Build one configured real-defect corpus and its deterministic split."""

    dataset_key = str(config["dataset_key"])
    if dataset != dataset_key:
        raise ValueError(f"Dataset argument {dataset!r} != configured key {dataset_key!r}")
    if bool(config["enrichment"].get("enabled", False)):
        raise ValueError(
            "Enrichment is off by default; enabled enrichment requires the "
            "separate offline enrich_description interface and callable"
        )

    processed_dir = _repo_path(str(config["processed_dir"]))
    manifest = load_json(_repo_path(str(config["splits_path"])))
    classes = config["classes"]
    excluded = set(str(value) for value in config["excluded_classes"])
    selected: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    for class_name in sorted(classes):
        if class_name in excluded:
            raise ValueError(f"Excluded class is also configured: {class_name}")
        if class_name not in manifest:
            raise ValueError(f"Configured class is absent from manifest: {class_name}")
        class_manifest = manifest[class_name]
        for source_split in sorted(class_manifest["splits"]):
            samples = sorted(
                class_manifest["splits"][source_split],
                key=lambda item: (
                    str(item.get("image_path", "")),
                    str(item.get("label", "")),
                ),
            )
            for sample in samples:
                if not bool(sample["is_anomalous"]):
                    continue
                defect_type = str(sample["label"])
                mask_value = sample.get("mask_path")
                if not mask_value:
                    warning = {
                        "class": class_name,
                        "image_path": str(sample["image_path"]),
                        "reason": "null mask_path",
                    }
                    skipped.append(warning)
                    logger.warning(
                        "Skipping %s/%s: null mask_path (%s)",
                        dataset_key,
                        class_name,
                        sample["image_path"],
                    )
                    continue
                if not _repo_path(str(mask_value)).is_file():
                    warning = {
                        "class": class_name,
                        "image_path": str(sample["image_path"]),
                        "reason": f"missing mask file: {mask_value}",
                    }
                    skipped.append(warning)
                    logger.warning(
                        "Skipping %s/%s: missing mask %s for %s",
                        dataset_key,
                        class_name,
                        mask_value,
                        sample["image_path"],
                    )
                    continue

                _template_for(config, class_name, defect_type)
                _assert_under_processed(str(sample["image_path"]), processed_dir)
                _assert_under_processed(str(mask_value), processed_dir)
                image_path = _repo_relative_posix(str(sample["image_path"]))
                mask_path = _repo_relative_posix(str(mask_value))
                if not _repo_path(image_path).is_file():
                    raise FileNotFoundError(f"Corpus image does not exist: {image_path}")
                uid = _uid_for(dataset_key, class_name, defect_type, image_path)
                selected.append(
                    _membership_record(
                        uid=uid,
                        class_name=class_name,
                        defect_type=defect_type,
                        image_path=image_path,
                        mask_path=mask_path,
                    )
                )

    assert_no_c2_synthetic_paths(selected)
    split = deterministic_stratified_split(
        selected,
        seed=int(config["seed"]),
        train_fraction=float(config["train_fraction"]),
        test_fraction=float(config["test_fraction"]),
    )
    output_dir = _repo_path(str(config["output_dir"]))
    schema_path, _ = _schema_config(config)

    for split_name in ("train", "test"):
        split_dir = output_dir / split_name
        for member in split[split_name]:
            report = build_report(
                member["image_path"],
                member["mask_path"],
                member["class"],
                member["defect_type"],
                config,
            )
            validate_against_schema(report, schema_path)
            grounding = _grounding_record(member, dataset_key, report)
            uid = member["uid"]
            save_json(split_dir / f"{uid}.grounding.json", grounding)
            save_json(split_dir / f"{uid}.report.json", report)

    membership = {
        "dataset_key": dataset_key,
        "seed": int(config["seed"]),
        "train_fraction": float(config["train_fraction"]),
        "test_fraction": float(config["test_fraction"]),
        "stratify_by": ["class", "defect_type"],
        "train": [{field: item[field] for field in MEMBERSHIP_FIELDS} for item in split["train"]],
        "test": [{field: item[field] for field in MEMBERSHIP_FIELDS} for item in split["test"]],
    }
    assert_no_c2_synthetic_paths([*membership["train"], *membership["test"]])
    membership_path = save_json(output_dir / "split_membership.json", membership)
    logger.info(
        "Built C3 corpus dataset=%s train=%d test=%d skipped=%d output=%s",
        dataset_key,
        len(split["train"]),
        len(split["test"]),
        len(skipped),
        output_dir,
    )
    return CorpusBuildResult(
        dataset_key=dataset_key,
        output_dir=output_dir,
        membership_path=membership_path,
        train_count=len(split["train"]),
        test_count=len(split["test"]),
        skipped=tuple(skipped),
    )
