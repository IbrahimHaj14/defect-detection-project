"""
src/c1_detector/sweep_patchcore_ecf.py

Runs the full ECF (3CAD) benchmark sweep using PatchCore.
Logs all runs to MLflow under the 'c1-patchcore-ecf' experiment.

Notes specific to ECF:
    - Category names contain spaces and numeric prefixes ('1.scratch',
      '3.blade mark'). Anomalib's Folder handles these fine.
    - `pseudo_broken_solder` has zero anomalous test images — Anomalib
      cannot compute image AUROC without both classes present. Skipped.
    - `missing_plate` (3 normals) and `metal_foreign_body` (11 normals)
      will train but produce noisy results. Warned but not skipped.

Also writes a per-category summary CSV to
    outputs/tables/c1/ecf_patchcore_sweep.csv
directly from the training loop, so it does not depend on the MLflow
evaluation script running afterwards.

Usage:
    python -m src.c1_detector.sweep_patchcore_ecf
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

from src.c1_detector.train_patchcore import train_patchcore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATASET_NAME = "ecf"
MANIFEST_PATH = Path("data/splits/ecf_splits.json")

# All 13 ECF categories in the manifest
# ECF_CATEGORIES = [
#     "1.scratch",
#     "2.Indentation",
#     "3.blade mark",
#     "4.dirt",
#     "5.particle",
#     "b2b_flex_pcb",
#     "contamination",
#     "ic_flex_pcb",
#     "ir_base",
#     "metal_foreign_body",
#     "missing_plate",
#     "pseudo_broken_solder",
#     "solder_bead",
# ]

ECF_CATEGORIES = ["ir_base"]

# Categories that must be skipped (documented reasons)
SKIP_CATEGORIES = {
    "pseudo_broken_solder": "0 anomalous test images — AUROC not computable",
}

# Categories with very few training normals — proceed with caution
LOW_SAMPLE_WARNING_THRESHOLD = 20
RESULTS_CSV_PATH = Path("outputs/tables/c1/ecf_patchcore_sweep.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Manifest helpers
# ─────────────────────────────────────────────────────────────────────────────
def check_category_readiness(category: str, manifest: dict) -> tuple[bool, str]:
    """
    Verify the category has enough data to train and evaluate meaningfully.
    Returns (ok, reason).
    """
    if category in SKIP_CATEGORIES:
        return False, SKIP_CATEGORIES[category]

    if category not in manifest:
        return False, "Category missing from manifest"

    entry = manifest[category]
    train_count = entry["stats"]["train"]["total"]
    test_anom   = entry["stats"]["test"]["anomalous"]
    test_normal = entry["stats"]["test"]["normal"]

    if train_count == 0:
        return False, "No training samples"
    if test_anom == 0:
        return False, "0 anomalous test images"
    if test_normal == 0:
        return False, "0 normal test images"
    if train_count < LOW_SAMPLE_WARNING_THRESHOLD:
        # Warn but do not skip
        logger.warning(
            f"  Low training sample count for '{category}': {train_count} images. "
            f"Results will be noisy."
        )

    return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────
def run_ecf_sweep() -> None:
    total_start = time.time()

    # Load manifest for readiness checks
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    logger.info("=" * 70)
    logger.info(" STARTING PATCHCORE ECF (3CAD) 13-CATEGORY SWEEP")
    logger.info("=" * 70)

    successful_results: list[dict] = []
    skipped_categories: list[dict] = []
    failed_categories: list[dict] = []

    total_categories = len(ECF_CATEGORIES)

    for idx, category in enumerate(ECF_CATEGORIES, 1):
        logger.info(f"\n[{idx}/{total_categories}] >>> Category: '{category}' <<<")

        # Skip check
        ok, reason = check_category_readiness(category, manifest)
        if not ok:
            logger.warning(f"  SKIPPED: {category} — {reason}")
            skipped_categories.append({"category": category, "reason": reason})
            continue

        # Train and evaluate
        try:
            metrics = train_patchcore(
                dataset=DATASET_NAME,
                category=category,
                image_size=(256, 256),
                backbone="wide_resnet50_2",
                layers=["layer2", "layer3"],
                coreset_ratio=0.01,
                num_neighbors=9,
                train_batch_size=32,
                eval_batch_size=32,
                num_workers=4,
                seed=42,
            )
            successful_results.append({
                "category": category,
                "image_AUROC":  metrics.get("image_AUROC"),
                "image_F1Score": metrics.get("image_F1Score"),
                "pixel_AUROC":  metrics.get("pixel_AUROC"),
                "pixel_F1Score": metrics.get("pixel_F1Score"),
            })
            logger.info(
                f"  DONE [{idx}/{total_categories}]: "
                f"image_AUROC={metrics.get('image_AUROC', 'n/a'):.4f}"
            )

        except Exception as e:
            logger.error(f"  FAILED: {category} — {e}", exc_info=True)
            failed_categories.append({"category": category, "error": str(e)})
            continue

    total_time = (time.time() - total_start) / 60.0

    # ─── Write per-category results CSV ─────────────────────────────────────
    RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "image_AUROC", "image_F1Score",
                "pixel_AUROC", "pixel_F1Score",
            ],
        )
        writer.writeheader()
        for row in successful_results:
            writer.writerow(row)

        # Compute and append mean row (skipped categories excluded)
        if successful_results:
            mean_row = {"category": "MEAN"}
            for key in ["image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score"]:
                values = [
                    row[key] for row in successful_results
                    if row[key] is not None
                ]
                mean_row[key] = sum(values) / len(values) if values else None
            writer.writerow(mean_row)

    # ─── Summary log ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info(" ECF SWEEP COMPLETE")
    logger.info("=" * 70)
    logger.info(f" Total Wall Time:  {total_time:.2f} minutes")
    logger.info(f" Successful:       {len(successful_results)}/{total_categories}")
    logger.info(f" Skipped:          {len(skipped_categories)}/{total_categories}")
    logger.info(f" Failed:           {len(failed_categories)}/{total_categories}")
    logger.info(f" Results CSV:      {RESULTS_CSV_PATH}")

    if skipped_categories:
        logger.info("\n Skipped Categories:")
        for entry in skipped_categories:
            logger.info(f"   - {entry['category']}: {entry['reason']}")

    if failed_categories:
        logger.error("\n Failed Categories:")
        for entry in failed_categories:
            logger.error(f"   - {entry['category']}: {entry['error']}")

    if successful_results:
        image_aurocs = [r["image_AUROC"] for r in successful_results if r["image_AUROC"] is not None]
        if image_aurocs:
            mean_auroc = sum(image_aurocs) / len(image_aurocs)
            logger.info(f"\n Mean image_AUROC across {len(image_aurocs)} completed categories: {mean_auroc:.4f}")


if __name__ == "__main__":
    run_ecf_sweep()