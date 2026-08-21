"""Phase 1 acceptance test for C1 pairing and ECF crop extraction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c2_synthesis.data.pair_builder import (  # noqa: E402
    build_pairs,
    get_eligible_classes,
    load_dataset_config,
)
from src.c2_synthesis.data.patch_extractor import (  # noqa: E402
    DefectCrop,
    composite_crop_back,
    extract_defect_crop,
    summarise_crop_eligibility,
)
from src.c2_synthesis.utils.image_io import (  # noqa: E402
    load_image_rgb,
    load_mask_binary,
    save_image,
)

CONTACT_SHEET_ITEMS = 5
CONTACT_TILE_SIZE = 256
MASK_OVERLAY_ALPHA = 0.45


def _resolve(path: str) -> Path:
    candidate = Path(path.replace("\\", "/"))
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _assert_pair_valid(pair: tuple[str, str]) -> None:
    image_path, mask_path = pair
    assert _resolve(image_path).is_file(), f"Missing image: {image_path}"
    assert _resolve(mask_path).is_file(), f"Missing mask: {mask_path}"
    image = load_image_rgb(image_path)
    mask = load_mask_binary(mask_path)
    assert image.size == (mask.shape[1], mask.shape[0]), (
        f"Dimension mismatch: {image_path}, {mask_path}"
    )
    assert mask.any(), f"Empty mask: {mask_path}"
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})


def _verify_budget_nesting(
    dataset_key: str,
    category: str,
) -> tuple[list[tuple[str, str]], list[str]]:
    config = load_dataset_config(dataset_key)
    budgets = config["budgets"]
    assert budgets == [5, 10, 20, "all"], f"Unexpected budgets in {dataset_key} config"

    pairs_by_budget: dict[Any, list[tuple[str, str]]] = {}
    clean_by_budget: dict[Any, list[str]] = {}
    for budget in budgets:
        pairs, clean_images = build_pairs(dataset_key, category, budget)
        pairs_by_budget[budget] = pairs
        clean_by_budget[budget] = clean_images

    pair_sets = {budget: set(pairs) for budget, pairs in pairs_by_budget.items()}
    assert pair_sets[5] < pair_sets[10] < pair_sets[20] < pair_sets["all"], (
        f"Budgets are not strict nested subsets for {dataset_key}/{category}"
    )
    assert len(pairs_by_budget[5]) == 5
    assert len(pairs_by_budget[10]) == 10
    assert len(pairs_by_budget[20]) == 20

    reference_clean_pool = clean_by_budget["all"]
    assert reference_clean_pool, f"No clean training images for {dataset_key}/{category}"
    for budget in budgets:
        assert clean_by_budget[budget] == reference_clean_pool

    for pair in pairs_by_budget["all"]:
        _assert_pair_valid(pair)

    return pairs_by_budget["all"], reference_clean_pool


def _mask_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = image_array.copy()
    overlay[mask.astype(bool)] = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    blended = image_array * (1.0 - MASK_OVERLAY_ALPHA) + overlay * MASK_OVERLAY_ALPHA
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def _save_contact_sheet(
    dataset_key: str,
    category: str,
    pairs: list[tuple[str, str]],
) -> str:
    selected_pairs = pairs[:CONTACT_SHEET_ITEMS]
    tile_height = CONTACT_TILE_SIZE + 24
    sheet = Image.new(
        "RGB",
        (CONTACT_TILE_SIZE * len(selected_pairs), tile_height),
        color="white",
    )
    draw = ImageDraw.Draw(sheet)

    for index, (image_path, mask_path) in enumerate(selected_pairs):
        image = load_image_rgb(image_path)
        mask = load_mask_binary(mask_path)
        overlay = _mask_overlay(image, mask)
        thumbnail = ImageOps.contain(
            overlay,
            (CONTACT_TILE_SIZE, CONTACT_TILE_SIZE),
            method=Image.Resampling.BICUBIC,
        )
        tile = Image.new("RGB", (CONTACT_TILE_SIZE, CONTACT_TILE_SIZE), color="black")
        tile_x = (CONTACT_TILE_SIZE - thumbnail.width) // 2
        tile_y = (CONTACT_TILE_SIZE - thumbnail.height) // 2
        tile.paste(thumbnail, (tile_x, tile_y))

        sheet_x = index * CONTACT_TILE_SIZE
        sheet.paste(tile, (sheet_x, 0))
        draw.text(
            (sheet_x + 4, CONTACT_TILE_SIZE + 4),
            Path(image_path).name,
            fill="black",
        )

    safe_category = category.replace(" ", "_").replace("/", "_")
    output_path = REPO_ROOT / f"outputs/figures/c2/phase1_pairs_{safe_category}.png"
    saved_path = save_image(sheet, output_path)
    assert _resolve(saved_path).is_file()
    print(f"Contact sheet ({dataset_key}/{category}): {saved_path}")
    return saved_path


def _find_crop_examples(
    pairs: list[tuple[str, str]],
    crop_size: int,
) -> tuple[DefectCrop, DefectCrop, tuple[int, int], tuple[str, str]]:
    standard_crop: DefectCrop | None = None
    resized_crop: DefectCrop | None = None
    resized_source_size: tuple[int, int] | None = None
    dropped_pair: tuple[str, str] | None = None

    for pair in pairs:
        image = load_image_rgb(pair[0])
        mask = load_mask_binary(pair[1])
        crop = extract_defect_crop(image, mask, size=crop_size)
        if crop is None:
            dropped_pair = pair
        elif crop.resized_from_full_frame:
            resized_crop = crop
            resized_source_size = image.size
        else:
            standard_crop = crop
        if standard_crop and resized_crop and dropped_pair:
            break

    assert standard_crop is not None, "No standard 256 crop found"
    assert resized_crop is not None, "No valid oversized/resized crop found"
    assert resized_source_size is not None
    assert dropped_pair is not None, "No unfittable-square drop found"
    return standard_crop, resized_crop, resized_source_size, dropped_pair


def _verify_composite(crop: DefectCrop, full_size: tuple[int, int]) -> None:
    clean = Image.new("RGB", full_size, color=(127, 127, 127))
    generated = Image.new("RGB", crop.image.size, color=(255, 0, 0))
    composite = composite_crop_back(
        full_clean=clean,
        gen_crop=generated,
        crop_offset=crop.crop_offset,
        crop_mask=crop.mask,
        crop_bbox=crop.crop_bbox,
    )

    assert composite.size == clean.size
    clean_pixels = np.asarray(clean, dtype=np.uint8)
    composite_pixels = np.asarray(composite, dtype=np.uint8)
    crop_x, crop_y, crop_width, crop_height = crop.crop_bbox

    resized_mask = Image.fromarray(crop.mask * 255).resize(
        (crop_width, crop_height),
        resample=Image.Resampling.NEAREST,
    )
    full_mask = np.zeros((clean.height, clean.width), dtype=bool)
    full_mask[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ] = np.asarray(resized_mask, dtype=np.uint8) > 127

    assert np.array_equal(
        composite_pixels[~full_mask],
        clean_pixels[~full_mask],
    ), "Composite changed pixels outside the placed mask"


def main() -> None:
    mvtec_config = load_dataset_config("mvtec_ad")
    ecf_config = load_dataset_config("ecf")
    assert mvtec_config["dataset_key"] == "mvtec_ad"
    assert ecf_config["dataset_key"] == "ecf"
    assert len(get_eligible_classes("mvtec_ad")) == int(mvtec_config["eligible_class_count"])
    assert len(get_eligible_classes("ecf")) == int(ecf_config["eligible_class_count"]) == 11
    assert len(ecf_config["excluded_classes"]) == 3

    mvtec_pairs, mvtec_clean = _verify_budget_nesting("mvtec_ad", "bottle")
    ecf_pairs, ecf_clean = _verify_budget_nesting("ecf", "1.scratch")

    crop_size = int(ecf_config["crop_size"])
    crop_summary = summarise_crop_eligibility(
        ecf_pairs,
        size=crop_size,
        category="1.scratch",
    )
    assert crop_summary.total_pairs == len(ecf_pairs)
    assert crop_summary.usable_pairs >= 3
    assert crop_summary.crop_exclusion_count > 0

    standard_crop, resized_crop, resized_source_size, dropped_pair = _find_crop_examples(
        ecf_pairs,
        crop_size,
    )
    assert standard_crop.image.size == (crop_size, crop_size)
    assert standard_crop.mask.shape == (crop_size, crop_size)
    assert standard_crop.crop_bbox[2:] == (crop_size, crop_size)
    assert standard_crop.crop_offset == standard_crop.crop_bbox[:2]

    assert resized_crop.image.size == (crop_size, crop_size)
    assert resized_crop.mask.shape == (crop_size, crop_size)
    assert resized_crop.crop_bbox[2] > crop_size
    assert resized_crop.crop_bbox[2] == resized_crop.crop_bbox[3]
    assert resized_crop.crop_offset == resized_crop.crop_bbox[:2]

    _verify_composite(resized_crop, resized_source_size)

    mvtec_sheet = _save_contact_sheet("mvtec_ad", "bottle", mvtec_pairs)
    ecf_sheet = _save_contact_sheet("ecf", "1.scratch", ecf_pairs)

    print("Phase 1 verification passed")
    print(
        f"MVTec bottle: {len(mvtec_pairs)} valid defect pairs, "
        f"{len(mvtec_clean)} clean training images"
    )
    print(
        f"ECF 1.scratch: {len(ecf_pairs)} valid defect pairs, "
        f"{len(ecf_clean)} clean training images"
    )
    print(
        "ECF 1.scratch crop eligibility: "
        f"usable={crop_summary.usable_pairs}, "
        f"excluded={crop_summary.crop_exclusion_count}"
    )
    print(f"Resized crop_bbox: {list(resized_crop.crop_bbox)}")
    print(f"Verified dropped sample: {dropped_pair[0]}")
    print(f"Artifacts: {mvtec_sheet}, {ecf_sheet}")


if __name__ == "__main__":
    main()
