"""
src/c1_detector/sweep_dinomaly_all.py

Runs the full Dinomaly sweeps: MVTec AD (15 categories) then ECF (12 categories,
skipping pseudo_broken_solder). Total expected wall time ~15-18 hours on Blackwell.
Safe to leave running overnight — each category's results are logged to MLflow
and written to CSV as they complete, so partial progress is preserved even if
the machine dies mid-sweep.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

from src.c1_detector.train_dinomaly import train_dinomaly

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

ECF_CATEGORIES = [
    "1.scratch", "2.Indentation", "3.blade mark", "4.dirt", "5.particle",
    "b2b_flex_pcb", "contamination", "ic_flex_pcb", "ir_base",
    "metal_foreign_body", "missing_plate", "solder_bead",
    # pseudo_broken_solder skipped: no anomalous test images
]

ECF_SKIP = {"pseudo_broken_solder": "0 anomalous test images — AUROC not computable"}


def run_sweep(dataset: str, categories: list[str], skip_reasons: dict = None) -> None:
    """Run Dinomaly across all categories, writing incremental CSV as we go."""
    skip_reasons = skip_reasons or {}
    csv_path = Path(f"outputs/tables/c1/{dataset}_dinomaly_sweep.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV header immediately so we have a valid file even if the sweep dies
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "category", "image_AUROC", "image_F1Score",
            "pixel_AUROC", "pixel_F1Score", "train_time_minutes"
        ])
        writer.writeheader()

    logger.info("=" * 70)
    logger.info(f" STARTING DINOMALY SWEEP — {dataset.upper()} ({len(categories)} categories)")
    logger.info("=" * 70)

    sweep_start = time.time()
    results = []

    for idx, category in enumerate(categories, 1):
        logger.info(f"\n[{idx}/{len(categories)}] >>> {dataset}/{category} <<<")

        if category in skip_reasons:
            logger.warning(f"  SKIPPED: {skip_reasons[category]}")
            continue

        try:
            cat_start = time.time()
            metrics = train_dinomaly(
                dataset=dataset,
                category=category,
                image_size=(448, 448),
                crop_size=392,
                encoder_name="dinov2reg_vit_base_14",
                max_epochs=100,
                train_batch_size=16,
                eval_batch_size=16,
                num_workers=4,
                seed=42,
            )
            cat_minutes = (time.time() - cat_start) / 60.0

            row = {
                "category": category,
                "image_AUROC": metrics.get("image_AUROC"),
                "image_F1Score": metrics.get("image_F1Score"),
                "pixel_AUROC": metrics.get("pixel_AUROC"),
                "pixel_F1Score": metrics.get("pixel_F1Score"),
                "train_time_minutes": round(cat_minutes, 2),
            }
            results.append(row)

            # Append this row to the CSV immediately (survives crashes)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writerow(row)

            logger.info(
                f"  DONE [{idx}/{len(categories)}]: "
                f"image_AUROC={metrics.get('image_AUROC', 'n/a'):.4f} "
                f"({cat_minutes:.1f} min)"
            )

        except Exception as e:
            logger.error(f"  FAILED: {category} — {e}", exc_info=True)
            continue

    total_hours = (time.time() - sweep_start) / 3600.0
    logger.info("\n" + "=" * 70)
    logger.info(f" {dataset.upper()} SWEEP COMPLETE — {total_hours:.2f} hours")
    logger.info(f" Results CSV: {csv_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    total_start = time.time()

    # MVTec first
    run_sweep("mvtec_ad", MVTEC_CATEGORIES)

    # Then ECF
    run_sweep("ecf", ECF_CATEGORIES, skip_reasons=ECF_SKIP)

    total_hours = (time.time() - total_start) / 3600.0
    logger.info(f"\n\n>>> ALL SWEEPS COMPLETE — Total wall time: {total_hours:.2f} hours <<<")