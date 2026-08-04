"""
src/data/step_06_splits.py

Generates train/val/test split manifests as JSON files in data/splits/.

Partitioning strategy: val_normals_only (Option 1)
  - Source train/good      -> 90% train, 10% val (both normal only)
  - Source test/good       -> 100% test (normal)
  - Source test/<defect>   -> 100% test (anomalous)

Rationale:
  - Val set is normal-only, used for anomaly threshold selection (3-sigma or
    F1-optimal), not for computing val AUROC during training
  - Test set is preserved exactly as dataset authors designed it, so results
    are directly comparable to published benchmark numbers
  - On SSGD (few defect examples per class) this avoids shrinking test/good
    which would degrade evaluation stability

For each category the manifest also contains a self-documenting 'stats' block
so the split shape is inspectable without iterating the file lists.

Usage:
    python src/data/step_06_splits.py                    # all datasets
    python src/data/step_06_splits.py --dataset mvtec_ad
    python src/data/step_06_splits.py --dataset ssgd --val-fraction 0.15
"""

import argparse
import json
import logging
import random
import csv
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
PROCESSED_ROOT = Path("data/processed")
SPLITS_ROOT = Path("data/splits")
SPLITS_ROOT.mkdir(exist_ok=True)

RANDOM_SEED = 42
VAL_FRACTION = 0.1
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

# Optional: enrichment source for SSGD entries — per-image defect types
# from the COCO conversion manifest
SSGD_MANIFEST_CSV = Path("logs/ssgd_mask_manifest.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def as_posix(path: Path) -> str:
    """Normalise path separators to forward slashes for cross-platform manifests."""
    return path.as_posix()


def find_mask_path(image_path: Path, category_dir: Path, defect_class: str) -> str | None:
    stem = image_path.stem
    
    # Convention 1: mask inside category (MVTec-style)
    # data/processed/mvtec_ad/metal_nut/ground_truth/scratch/000_mask.png
    dataset_level_gt = category_dir / "ground_truth"
    candidates = [
        dataset_level_gt / defect_class / f"{stem}.png",
        dataset_level_gt / defect_class / f"{stem}_mask.png",
        dataset_level_gt / "defects" / f"{stem}.png",
        dataset_level_gt / f"{stem}.png",
    ]
    
    # Convention 2: mask at dataset level (ECF-style)
    # data/processed/ecf/ground_truth/defects/Num1028.png
    dataset_root = category_dir.parent
    dataset_level_candidates = [
        dataset_root / "ground_truth" / "defects" / f"{stem}.png",
        dataset_root / "ground_truth" / defect_class / f"{stem}.png",
        dataset_root / "ground_truth" / f"{stem}.png",
    ]
    
    for c in candidates + dataset_level_candidates:
        if c.exists():
            return as_posix(c)
    return None


def load_ssgd_defect_types() -> dict[str, list[str]]:
    """
    Parse the SSGD mask manifest CSV to build a lookup:
    image_filename -> list of defect category names present.
    Empty dict if the manifest doesn't exist yet.
    """
    lookup = {}
    if not SSGD_MANIFEST_CSV.exists():
        return lookup

    with open(SSGD_MANIFEST_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            defect_types_str = row.get('defect_categories', '')
            if defect_types_str:
                lookup[row['image_filename']] = defect_types_str.split(';')
    logger.info(f"Loaded SSGD defect-type lookup for {len(lookup)} images")
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Split generation for a single category
# ─────────────────────────────────────────────────────────────────────────────
def collect_images(directory: Path) -> list[Path]:
    """Collect and sort image files under a directory (recursive)."""
    if not directory.exists():
        return []
    imgs = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(imgs)  # sorted() essential for deterministic shuffling


def build_entry(
    image_path: Path,
    label: str,
    is_anomalous: bool,
    mask_path: str | None = None,
    defect_types: list[str] | None = None,
) -> dict:
    """Construct a single manifest entry."""
    entry = {
        "image_path": as_posix(image_path),
        "label": label,
        "is_anomalous": is_anomalous,
        "mask_path": mask_path,
    }
    if defect_types:
        entry["defect_types"] = defect_types
    return entry


def generate_splits_for_category(
    dataset_name: str,
    category_dir: Path,
    val_fraction: float,
    seed: int,
    ssgd_defect_lookup: dict | None = None,
) -> dict:
    """
    Build a full split manifest for one category using Option 1 strategy.
    """
    rng = random.Random(seed)  # Isolated RNG per category — deterministic

    train_good_dir = category_dir / "train" / "good"
    test_dir = category_dir / "test"
    ground_truth_dir = category_dir / "ground_truth"

    splits = {"train": [], "val": [], "test": []}

    # ─── Train / Val split from train/good ──────────────────────────────────
    train_normals = collect_images(train_good_dir)
    rng.shuffle(train_normals)

    n_val = max(1, int(len(train_normals) * val_fraction)) if train_normals else 0
    val_images = train_normals[:n_val]
    train_images = train_normals[n_val:]

    for img in train_images:
        splits["train"].append(build_entry(img, "good", is_anomalous=False))

    for img in val_images:
        splits["val"].append(build_entry(img, "good", is_anomalous=False))

    # ─── Test split ─────────────────────────────────────────────────────────
    # test/good goes entirely to test
    test_good_dir = test_dir / "good"
    for img in collect_images(test_good_dir):
        splits["test"].append(build_entry(img, "good", is_anomalous=False))

    # For SSGD the test images are under test/defects rather than per-class folders
    # Handle both conventions
    if test_dir.exists():
        for defect_class_dir in sorted(test_dir.iterdir()):
            if not defect_class_dir.is_dir() or defect_class_dir.name == "good":
                continue

            defect_class = defect_class_dir.name
            for img in collect_images(defect_class_dir):
                # Check category ground_truth first; if missing, check dataset-level ground_truth
                gt_root = ground_truth_dir if ground_truth_dir.exists() else category_dir.parent / "ground_truth"
                
                mask_path = find_mask_path(img, category_dir, defect_class)
                # SSGD-specific enrichment: attach per-image defect types
                defect_types = None
                if ssgd_defect_lookup and img.name in ssgd_defect_lookup:
                    defect_types = ssgd_defect_lookup[img.name]

                splits["test"].append(build_entry(
                    img,
                    label=defect_class,
                    is_anomalous=True,
                    mask_path=mask_path,
                    defect_types=defect_types,
                ))

    # ─── Compute stats block ────────────────────────────────────────────────
    def compute_stats(entries: list[dict]) -> dict:
        normal = sum(1 for e in entries if not e["is_anomalous"])
        anomalous = sum(1 for e in entries if e["is_anomalous"])
        by_class = Counter(e["label"] for e in entries)
        return {
            "total": len(entries),
            "normal": normal,
            "anomalous": anomalous,
            "by_class": dict(sorted(by_class.items())),
        }

    manifest = {
        "dataset_name": dataset_name,
        "category": category_dir.name,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "split_strategy": "val_normals_only",
        "val_fraction": val_fraction,
        "stats": {
            "train": compute_stats(splits["train"]),
            "val":   compute_stats(splits["val"]),
            "test":  compute_stats(splits["test"]),
        },
        "splits": splits,
    }

    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level generation
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR_MAP = {
    "mvtec_ad": "mvtec_ad",
    "ecf": "ECF-Dataset",
    "ssgd": "SSGD",
}

EXCLUDE_DIRS = {"ground_truth", ".DS_Store", "__MACOSX", ".git", ".dvc"}

def generate_all_splits(dataset_name: str, val_fraction: float, seed: int):
    # Lookup exact folder name on disk
    folder_name = DATASET_DIR_MAP.get(dataset_name, dataset_name)
    dataset_dir = PROCESSED_ROOT / folder_name

    if not dataset_dir.exists():
        logger.warning(f"Dataset not found: {dataset_dir}")
        return

    ssgd_lookup = load_ssgd_defect_types() if dataset_name == "ssgd" else None

    all_categories = {}
    for category_dir in sorted(dataset_dir.iterdir()):
        if not category_dir.is_dir() or category_dir.name in EXCLUDE_DIRS:
            continue

        manifest = generate_splits_for_category(
            dataset_name=dataset_name,
            category_dir=category_dir,
            val_fraction=val_fraction,
            seed=seed,
            ssgd_defect_lookup=ssgd_lookup,
        )
        all_categories[category_dir.name] = manifest

        stats = manifest["stats"]
        logger.info(
            f"  {category_dir.name:<25} | "
            f"train {stats['train']['total']:>4} | "
            f"val {stats['val']['total']:>3} | "
            f"test {stats['test']['total']:>4} "
            f"({stats['test']['anomalous']} anomalous, {stats['test']['normal']} normal)"
        )

    output_path = SPLITS_ROOT / f"{dataset_name}_splits.json"
    output_path.write_text(json.dumps(all_categories, indent=2))
    logger.info(f"Manifest saved: {output_path}")

    # Also emit a top-level summary
    summary = {
        "dataset_name": dataset_name,
        "num_categories": len(all_categories),
        "totals": {
            "train": sum(m["stats"]["train"]["total"] for m in all_categories.values()),
            "val":   sum(m["stats"]["val"]["total"]   for m in all_categories.values()),
            "test":  sum(m["stats"]["test"]["total"]  for m in all_categories.values()),
        },
    }
    logger.info(f"Dataset totals — {summary['totals']}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", "mvtec_ad", "ecf", "ssgd"],
        help="Which dataset to process (default: all)"
    )
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    datasets = ["mvtec_ad", "ecf", "ssgd"] if args.dataset == "all" else [args.dataset]

    all_summaries = []
    for ds in datasets:
        logger.info(f"\n{'='*70}\nGenerating splits: {ds}\n{'='*70}")
        summary = generate_all_splits(ds, args.val_fraction, args.seed)
        if summary:
            all_summaries.append(summary)

    # Print grand totals
    print("\n" + "=" * 70)
    print("SPLIT MANIFEST GENERATION SUMMARY")
    print("=" * 70)
    for s in all_summaries:
        t = s["totals"]
        print(
            f"  {s['dataset_name']:<12} | {s['num_categories']:>2} categories | "
            f"train {t['train']:>5} | val {t['val']:>4} | test {t['test']:>5}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()