"""
src/c1_detector/train_patchcore.py

Train a PatchCore model on a single category using the manifest-driven datamodule.

This script is the "hello world" for C1 training. Running it on mvtec_ad/bottle
should produce an image-level AUROC of approximately 99.5 percent, matching the
published PatchCore baseline. If it does, the entire data + training pipeline
is validated end-to-end and we can scale to full sweeps.

Hyperparameters follow the executive summary:
    - Backbone: WideResNet-50
    - Layers extracted: 2 and 3 (mid-level features)
    - Coreset ratio: 1 percent (matches paper sweet spot)
    - Nearest neighbours k: 9
    - Input resolution: 256 x 256

Usage:
    python -m src.c1_detector.train_patchcore \
        --dataset mvtec_ad \
        --category bottle

    python -m src.c1_detector.train_patchcore \
        --dataset mvtec_ad \
        --category bottle \
        --image-size 256 \
        --coreset-ratio 0.01 \
        --num-neighbors 9
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

import torch
import mlflow
import mlflow.pytorch

from anomalib.models import Patchcore
from anomalib.engine import Engine

# Adjust the import path.
# If run via `python -m src.c1_detector.train_patchcore`, use relative import:
#   from .datamodule import build_datamodule_from_manifest, summarise_datamodule
# If run as a plain script, use the direct import below:
from .datamodule import build_datamodule_from_manifest, summarise_datamodule

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MANIFEST_PATHS = {
    "mvtec_ad": "data/splits/mvtec_ad_splits.json",
    "ecf":      "data/splits/ecf_splits.json",
    "ssgd":     "data/splits/ssgd_splits.json",
}

CHECKPOINT_ROOT = Path("outputs/checkpoints/c1/patchcore")
MLFLOW_TRACKING_URI = "file:./outputs/logs/mlflow"


# ─────────────────────────────────────────────────────────────────────────────
# Main train routine
# ─────────────────────────────────────────────────────────────────────────────
def train_patchcore(
    dataset: str,
    category: str,
    image_size: tuple[int, int] = (256, 256),
    backbone: str = "wide_resnet50_2",
    layers: list[str] = ["layer2", "layer3"],
    coreset_ratio: float = 0.01,
    num_neighbors: int = 9,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    device: str = "auto",
) -> dict:
    """
    Train PatchCore on a single category and log to MLflow.
    Returns a dict of final metrics.
    """
    # ── Verify GPU ──────────────────────────────────────────────────────────
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_vram:.1f} GB VRAM)")
    else:
        logger.warning("Training on CPU — this will be very slow.")

    # ── Set up MLflow ────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_name = f"c1-patchcore-{dataset}"
    mlflow.set_experiment(experiment_name)

    run_name = f"{category}-r{coreset_ratio}-{image_size[0]}px"
    logger.info(f"MLflow experiment: {experiment_name}")
    logger.info(f"MLflow run name:   {run_name}")

    with mlflow.start_run(run_name=run_name) as run:
        # ── Log parameters ──────────────────────────────────────────────────
        params = {
            "model": "patchcore",
            "dataset": dataset,
            "category": category,
            "backbone": backbone,
            "layers": ",".join(layers),
            "image_size_h": image_size[0],
            "image_size_w": image_size[1],
            "coreset_ratio": coreset_ratio,
            "num_neighbors": num_neighbors,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "seed": seed,
            "device": device,
        }
        mlflow.log_params(params)
        logger.info(f"Params: {params}")

        # ── Build datamodule ────────────────────────────────────────────────
        manifest_path = MANIFEST_PATHS[dataset]
        dm = build_datamodule_from_manifest(
            manifest_path=manifest_path,
            category=category,
            image_size=image_size,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            seed=seed,
        )
        dm.setup()
        summarise_datamodule(dm)

        # ── Build model ─────────────────────────────────────────────────────
        model = Patchcore(
            backbone=backbone,
            layers=layers,
            pre_trained=True,
            coreset_sampling_ratio=coreset_ratio,
            num_neighbors=num_neighbors,
        )
        logger.info(f"Model: PatchCore ({backbone}, layers={layers})")

        # ── Set up Anomalib Engine ──────────────────────────────────────────
        # For PatchCore, "training" is a single-pass memory bank construction.
        # We only need max_epochs=1 for the feature extraction pass.
        checkpoint_dir = CHECKPOINT_ROOT / dataset / category
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        engine = Engine(
            max_epochs=1,
            default_root_dir=str(checkpoint_dir),
            accelerator=device,
            devices=1,
        )

        # ── Train (memory-bank construction) ────────────────────────────────
        logger.info("Starting PatchCore training (memory-bank construction)...")
        train_start = time.time()
        engine.fit(datamodule=dm, model=model)
        train_time = time.time() - train_start
        logger.info(f"Training completed in {train_time:.1f} seconds.")
        mlflow.log_metric("train_time_seconds", train_time)

        # Log peak VRAM
        if device == "cuda":
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f"Peak VRAM: {peak_vram_gb:.2f} GB")
            mlflow.log_metric("peak_vram_gb", peak_vram_gb)

        # ── Evaluate on test set ────────────────────────────────────────────
        logger.info("Evaluating on test set...")
        test_start = time.time()
        test_results = engine.test(datamodule=dm, model=model)
        test_time = time.time() - test_start
        logger.info(f"Testing completed in {test_time:.1f} seconds.")
        mlflow.log_metric("test_time_seconds", test_time)

        # ── Extract and log metrics ─────────────────────────────────────────
        # Anomalib returns a list of dicts (one per dataloader).
        metrics_dict = test_results[0] if test_results else {}

        logger.info("=" * 60)
        logger.info(f"RESULTS: {dataset}/{category}")
        logger.info("=" * 60)
        for metric_name, metric_value in sorted(metrics_dict.items()):
            if isinstance(metric_value, (int, float)):
                logger.info(f"  {metric_name:<40} {metric_value:.4f}")
                # Clean metric name for MLflow (no slashes/spaces)
                clean_name = metric_name.replace("/", "_").replace(" ", "_")
                mlflow.log_metric(clean_name, float(metric_value))
        logger.info("=" * 60)

        # ── Save the trained model ──────────────────────────────────────────
        ckpt_path = checkpoint_dir / "final.ckpt"
        try:
            engine.trainer.save_checkpoint(str(ckpt_path))
            mlflow.log_artifact(str(ckpt_path))
            logger.info(f"Checkpoint saved: {ckpt_path}")
        except Exception as e:
            logger.warning(f"Could not save checkpoint: {e}")

        # ── Step 1: Export TorchScript (model.pt) for Deployment ────────────
        try:
            canonical_export_dir = CHECKPOINT_ROOT / dataset / category
            canonical_export_dir.mkdir(parents=True, exist_ok=True)

            # Search Anomalib's root directory for exported TorchScript model
            anomalib_weights = list(checkpoint_dir.rglob("model.pt"))
            if anomalib_weights:
                src = anomalib_weights[0]
                dest = canonical_export_dir / "model.pt"
                if src != dest:
                    shutil.copy2(src, dest)
                logger.info(f"Model weights successfully exported to: {dest}")
                mlflow.log_param("model_weights_path", str(dest))
            else:
                logger.warning("Torch model.pt not found in Anomalib output path.")
        except Exception as e:
            logger.warning(f"Could not export canonical model.pt: {e}")

        # ───────────────────────────────────────────────────────────────────

        mlflow.log_metric("total_wall_time_seconds", train_time + test_time)
        logger.info(f"MLflow run ID: {run.info.run_id}")
        logger.info(f"MLflow UI: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")

        return metrics_dict


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train PatchCore on one category using the manifest-driven pipeline."
    )
    parser.add_argument("--dataset", type=str, default="mvtec_ad",
                        choices=["mvtec_ad", "ecf", "ssgd"])
    parser.add_argument("--category", type=str, default="bottle",
                        help="Category within the dataset (e.g. 'bottle' for mvtec_ad).")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Square input resolution.")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument("--coreset-ratio", type=float, default=0.01)
    parser.add_argument("--num-neighbors", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_patchcore(
        dataset=args.dataset,
        category=args.category,
        image_size=(args.image_size, args.image_size),
        backbone=args.backbone,
        coreset_ratio=args.coreset_ratio,
        num_neighbors=args.num_neighbors,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()