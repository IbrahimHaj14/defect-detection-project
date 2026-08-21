"""Build deterministic C2 image/mask pairs from C1 split manifests."""

from __future__ import annotations

import copy
import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypeAlias

import yaml

from src.c2_synthesis.utils.image_io import load_image_rgb, load_mask_binary
from src.c2_synthesis.utils.logging_utils import get_logger

Budget: TypeAlias = int | Literal["all"]
DefectPair: TypeAlias = tuple[str, str]

_C2_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASET_CONFIG_DIR = _C2_ROOT / "configs" / "datasets"
_REQUIRED_CONFIG_KEYS = {
    "dataset_key",
    "processed_dir",
    "splits_path",
    "classes",
    "excluded_classes",
    "generation_mode",
    "crop_size",
    "selection_seed",
    "budgets",
    "eligible_class_count",
}

logger = get_logger(__name__)


def _repo_path(posix_path: str) -> Path:
    path = Path(str(posix_path).replace("\\", "/"))
    return path if path.is_absolute() else _REPO_ROOT / path


def _require_within(path: Path, root: Path, label: str) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} is outside configured processed_dir: {path}") from error


@lru_cache(maxsize=None)
def _load_dataset_config_cached(dataset_key: str) -> dict[str, Any]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for config_path in sorted(_DATASET_CONFIG_DIR.glob("*.yaml")):
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if isinstance(config, dict) and config.get("dataset_key") == dataset_key:
            matches.append((config_path, config))

    if not matches:
        available = sorted(
            config_path.stem for config_path in _DATASET_CONFIG_DIR.glob("*.yaml")
        )
        raise KeyError(f"Unknown dataset key '{dataset_key}'. Configs: {available}")
    if len(matches) != 1:
        paths = [path.as_posix() for path, _ in matches]
        raise ValueError(f"Duplicate configs for dataset key '{dataset_key}': {paths}")

    config_path, config = matches[0]
    missing_keys = _REQUIRED_CONFIG_KEYS.difference(config)
    if missing_keys:
        raise ValueError(
            f"Dataset config {config_path.as_posix()} is missing keys: "
            f"{sorted(missing_keys)}"
        )

    classes = list(config["classes"])
    excluded = list(config["excluded_classes"])
    unknown_exclusions = sorted(set(excluded).difference(classes))
    if unknown_exclusions:
        raise ValueError(f"Excluded classes are absent from class list: {unknown_exclusions}")

    eligible_count = len([category for category in classes if category not in excluded])
    if eligible_count != int(config["eligible_class_count"]):
        raise ValueError(
            f"eligible_class_count={config['eligible_class_count']} but config contains "
            f"{eligible_count} eligible classes"
        )

    processed_dir = _repo_path(str(config["processed_dir"]))
    splits_path = _repo_path(str(config["splits_path"]))
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"Configured processed_dir does not exist: {processed_dir}")
    if not splits_path.is_file():
        raise FileNotFoundError(f"Configured splits_path does not exist: {splits_path}")

    return config


def load_dataset_config(dataset_key: str) -> dict[str, Any]:
    """Load and validate the YAML identified by its logical dataset key."""

    if not dataset_key or not dataset_key.strip():
        raise ValueError("dataset_key must be a non-empty string")
    return copy.deepcopy(_load_dataset_config_cached(dataset_key))


def get_eligible_classes(dataset_key: str) -> list[str]:
    """Return configured classes after applying only C2-local exclusions."""

    config = load_dataset_config(dataset_key)
    excluded = set(config["excluded_classes"])
    return [category for category in config["classes"] if category not in excluded]


@lru_cache(maxsize=None)
def _validate_pair(image_path: str, mask_path: str, processed_dir: str) -> bool:
    image_file = _repo_path(image_path)
    mask_file = _repo_path(mask_path)
    processed_root = _repo_path(processed_dir)

    _require_within(image_file, processed_root, "image_path")
    _require_within(mask_file, processed_root, "mask_path")
    if not image_file.is_file():
        raise FileNotFoundError(f"Defect image does not exist: {image_file}")
    if not mask_file.is_file():
        raise FileNotFoundError(f"Defect mask does not exist: {mask_file}")

    image = load_image_rgb(image_file)
    mask = load_mask_binary(mask_file)
    if image.size != (mask.shape[1], mask.shape[0]):
        raise ValueError(
            f"Image/mask dimensions differ for {image_path}: "
            f"image={image.size}, mask={(mask.shape[1], mask.shape[0])}"
        )
    if not mask.any():
        logger.warning("Skipping all-zero defect mask: %s", mask_path)
        return False
    return True


def _normalise_budget(budget: Budget) -> int | Literal["all"]:
    if budget == "all":
        return "all"
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError("budget must be a positive integer or 'all'")
    if budget <= 0:
        raise ValueError("budget must be positive")
    return budget


def build_pairs(
    dataset: str,
    category: str,
    budget: Budget,
) -> tuple[list[DefectPair], list[str]]:
    """Return deterministic defect pairs and clean training images.

    The C1 manifest remains the sole split authority. Pair selection is a
    seeded shuffle of a canonical ordering, so integer budgets are nested
    prefixes of the sequence returned for ``budget='all'``.
    """

    config = load_dataset_config(dataset)
    classes = list(config["classes"])
    excluded = set(config["excluded_classes"])
    if category not in classes:
        raise KeyError(f"Category '{category}' is not configured for dataset '{dataset}'")
    if category in excluded:
        raise ValueError(f"Category '{category}' is excluded from C2 for dataset '{dataset}'")

    splits_path = _repo_path(str(config["splits_path"]))
    processed_dir = str(config["processed_dir"])
    with splits_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    if category not in manifest:
        raise KeyError(f"Category '{category}' is absent from {splits_path.as_posix()}")
    category_manifest = manifest[category]
    splits = category_manifest.get("splits", {})

    clean_images: list[str] = []
    for entry in splits.get("train", []):
        if entry.get("is_anomalous"):
            continue
        image_path = Path(str(entry["image_path"]).replace("\\", "/")).as_posix()
        image_file = _repo_path(image_path)
        _require_within(image_file, _repo_path(processed_dir), "clean image_path")
        if not image_file.is_file():
            raise FileNotFoundError(f"Clean training image does not exist: {image_file}")
        clean_images.append(image_path)

    if not clean_images:
        raise ValueError(f"No clean train images for {dataset}/{category}")

    all_pairs: list[DefectPair] = []
    for entry in splits.get("test", []):
        if not entry.get("is_anomalous"):
            continue
        mask_value = entry.get("mask_path")
        if not mask_value:
            logger.warning(
                "Skipping anomalous sample with no mask for %s/%s: %s",
                dataset,
                category,
                entry.get("image_path"),
            )
            continue

        image_path = Path(str(entry["image_path"]).replace("\\", "/")).as_posix()
        mask_path = Path(str(mask_value).replace("\\", "/")).as_posix()
        if _validate_pair(image_path, mask_path, processed_dir):
            all_pairs.append((image_path, mask_path))

    if not all_pairs:
        raise ValueError(f"No valid defect pairs for {dataset}/{category}")

    ordered_pairs = sorted(all_pairs)
    random.Random(int(config["selection_seed"])).shuffle(ordered_pairs)

    normalised_budget = _normalise_budget(budget)
    if normalised_budget == "all":
        selected_pairs = ordered_pairs
    else:
        selected_pairs = ordered_pairs[:normalised_budget]
        if len(selected_pairs) < normalised_budget:
            logger.warning(
                "Requested budget %d for %s/%s, but only %d valid pairs exist",
                normalised_budget,
                dataset,
                category,
                len(selected_pairs),
            )

    return selected_pairs, clean_images
