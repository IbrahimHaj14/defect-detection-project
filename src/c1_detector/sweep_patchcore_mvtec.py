"""
src/c1_detector/sweep_patchcore_mvtec.py

Runs the full 15-category MVTec AD benchmark sweep using PatchCore.
Logs all runs to MLflow under the 'c1-patchcore-mvtec_ad' experiment.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

# Import the train_patchcore function directly from train_patchcore.py
from src.c1_detector.train_patchcore import train_patchcore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Full 15 MVTec AD categories (10 objects, 5 textures)
MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def run_mvtec_sweep() -> None:
    """Execute PatchCore training and evaluation across all 15 MVTec AD categories."""
    total_start = time.time()
    successful_categories = []
    failed_categories = []

    logger.info("=" * 70)
    logger.info(" STARTING PATCHCORE MVTEC AD 15-CATEGORY SWEEP")
    logger.info("=" * 70)

    for idx, category in enumerate(MVTEC_CATEGORIES, 1):
        logger.info(f"\n[{idx}/15] >>> Processing Category: {category.upper()} <<<")
        
        try:
            # Calls your exact train_patchcore function with default optimal parameters
            train_patchcore(
                dataset="mvtec_ad",
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
            successful_categories.append(category)
            logger.info(f"Successfully completed [{idx}/15]: {category}")

        except Exception as e:
            logger.error(f"FAILED processing category '{category}': {e}", exc_info=True)
            failed_categories.append(category)
            continue

    total_time = (time.time() - total_start) / 60.0

    logger.info("\n" + "=" * 70)
    logger.info(" SWEEP COMPLETE")
    logger.info("=" * 70)
    logger.info(f" Total Wall Time: {total_time:.2f} minutes")
    logger.info(f" Successful ({len(successful_categories)}/15): {successful_categories}")
    if failed_categories:
        logger.error(f" Failed ({len(failed_categories)}/15): {failed_categories}")


if __name__ == "__main__":
    run_mvtec_sweep()