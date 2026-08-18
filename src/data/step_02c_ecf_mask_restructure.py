"""
src/data/step_02c_ecf_mask_restructure.py

Reshapes ECF masks from a flat dataset-level directory into per-category
subdirectories, matching Anomalib's expected Folder datamodule layout.

Before:
    data/processed/ecf/ground_truth/defects/Num1028.png    (flat)
    data/processed/ecf/1.scratch/test/1.scratch/Num1028.bmp

After:
    data/processed/ecf/1.scratch/ground_truth/1.scratch/Num1028.png
    data/processed/ecf/1.scratch/test/1.scratch/Num1028.bmp

The mapping matches mask stems to test image stems, automatically stripping
common dataset suffixes (e.g., _leftImg8bit) to resolve naming discrepancies.

Usage:
    python -m src.data.step_02c_ecf_mask_restructure --dry-run
    python -m src.data.step_02c_ecf_mask_restructure --execute
    python -m src.data.step_02c_ecf_mask_restructure --execute --delete-flat
"""

from __future__ import annotations

import argparse
import logging
import shutil
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ECF_ROOT = Path("data/processed/ECF-Dataset")
FLAT_MASK_DIR = ECF_ROOT / "ground_truth" / "defects"


def normalize_stem(path_or_stem: str | Path) -> str:
    """
    Strip dataset and camera suffixes so image and mask stems match cleanly.
    e.g., '5_b402_seg_000478_leftImg8bit' -> '5_b402_seg_000478'
    """
    stem = Path(path_or_stem).stem
    if stem.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        stem = Path(stem).stem

    suffixes = [
        "_leftImg8bit",
        "_rightImg8bit",
        "_gtFine_polygons",
        "_labelIds",
        "_mask",
        "_gt",
    ]
    for s in suffixes:
        if stem.endswith(s):
            stem = stem[: -len(s)]
            break
    return stem


def collect_flat_masks() -> dict[str, Path]:
    """Return a dict of {mask_stem: mask_path} from the flat directory."""
    if not FLAT_MASK_DIR.exists():
        logger.error(f"Flat mask directory not found: {FLAT_MASK_DIR}")
        return {}
    masks = {p.stem: p for p in FLAT_MASK_DIR.glob("*.png")}
    logger.info(f"Found {len(masks)} masks in flat directory: {FLAT_MASK_DIR}")
    return masks


def collect_category_test_images() -> dict[str, list[Path]]:
    """
    Return {category_name: [test_image_paths]} for each ECF category.
    """
    result = defaultdict(list)
    for category_dir in sorted(ECF_ROOT.iterdir()):
        if not category_dir.is_dir() or category_dir.name == "ground_truth":
            continue

        candidate_dirs = [
            category_dir / "test" / category_dir.name,
            category_dir / "test" / "defects",
        ]
        test_dir = category_dir / "test"
        if test_dir.exists():
            for sub in test_dir.iterdir():
                if sub.is_dir() and sub.name != "good" and sub not in candidate_dirs:
                    candidate_dirs.append(sub)

        for d in candidate_dirs:
            if d.exists():
                for img in d.glob("*"):
                    if img.suffix.lower() in {".bmp", ".jpg", ".jpeg", ".png"}:
                        result[category_dir.name].append(img)

    for cat, imgs in result.items():
        logger.info(f"  {cat:<25} {len(imgs)} test images")
    return dict(result)


def build_routing_plan(
    masks: dict[str, Path],
    category_images: dict[str, list[Path]],
) -> tuple[dict[str, list[tuple[Path, Path]]], list[str]]:
    """
    Build a plan mapping each mask to its destination path using exact and normalized stem matching.
    """
    exact_index = {}
    norm_index = {}

    for category, images in category_images.items():
        for img in images:
            exact_index[img.stem] = (category, img.stem)
            norm_index[normalize_stem(img.stem)] = (category, img.stem)

    routing = defaultdict(list)
    orphans = []

    for mask_stem, mask_path in masks.items():
        match = exact_index.get(mask_stem) or norm_index.get(normalize_stem(mask_stem))
        if match is None:
            orphans.append(mask_stem)
            continue

        category, img_stem = match
        # Save destination mask using the test image's stem so Anomalib matches them cleanly
        dst_dir = ECF_ROOT / category / "ground_truth" / category
        dst_path = dst_dir / f"{img_stem}.png"
        routing[category].append((mask_path, dst_path))

    return dict(routing), orphans


def apply_routing(
    routing: dict[str, list[tuple[Path, Path]]],
    dry_run: bool,
) -> dict:
    """Execute the routing plan. Returns summary stats."""
    stats = {"copied": 0, "already_present": 0, "categories": {}}

    for category, pairs in sorted(routing.items()):
        cat_copied = 0
        cat_present = 0

        for src, dst in pairs:
            if dst.exists():
                cat_present += 1
                continue

            if dry_run:
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            cat_copied += 1

        stats["copied"] += cat_copied
        stats["already_present"] += cat_present
        stats["categories"][category] = {
            "total_masks": len(pairs),
            "copied": cat_copied,
            "already_present": cat_present,
        }

        action = "would copy" if dry_run else "copied"
        logger.info(
            f"  {category:<25} {len(pairs):>4} masks | "
            f"{action}: {cat_copied} | already present: {cat_present}"
        )

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--execute", action="store_true", default=False)
    parser.add_argument(
        "--delete-flat",
        action="store_true",
        help="After copying, remove the flat ground_truth/defects/ directory.",
    )
    args = parser.parse_args()

    dry_run = not args.execute
    if dry_run:
        logger.info("DRY RUN mode. Pass --execute to apply.")

    masks = collect_flat_masks()
    if not masks:
        return

    logger.info("Scanning category test directories...")
    category_images = collect_category_test_images()
    if not category_images:
        logger.error("No category test images found. Aborting.")
        return

    logger.info("Building routing plan...")
    routing, orphans = build_routing_plan(masks, category_images)

    total_routed = sum(len(pairs) for pairs in routing.values())
    logger.info(
        f"Routing plan: {total_routed}/{len(masks)} masks matched, "
        f"{len(orphans)} orphans."
    )

    if orphans:
        logger.warning(
            f"Orphan masks (no matching test image): {len(orphans)}. "
            f"First 5: {orphans[:5]}"
        )

    logger.info("Applying routing plan...")
    stats = apply_routing(routing, dry_run)

    print("\n" + "=" * 70)
    print(f"ECF MASK RESTRUCTURE SUMMARY ({'DRY RUN' if dry_run else 'EXECUTED'})")
    print("=" * 70)
    print(f"  Total masks in flat directory:  {len(masks)}")
    print(f"  Matched and routed:             {total_routed}")
    print(f"  Orphans (unmatched):            {len(orphans)}")
    if not dry_run:
        print(f"  Copied this run:                {stats['copied']}")
        print(f"  Already present (skipped):      {stats['already_present']}")

    if not dry_run and args.delete_flat:
        logger.warning(f"Deleting flat directory: {FLAT_MASK_DIR}")
        shutil.rmtree(FLAT_MASK_DIR)
        gt_parent = FLAT_MASK_DIR.parent
        if gt_parent.exists() and not any(gt_parent.iterdir()):
            gt_parent.rmdir()
    print("=" * 70)


if __name__ == "__main__":
    main()