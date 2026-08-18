"""
src/c1_detector/train_dinomaly.py

Train a Dinomaly model on a single category using the manifest-driven datamodule.

Dinomaly (Guo et al., CVPR 2025) is a reconstruction-based anomaly detector
that uses a frozen DINOv2 Vision Transformer encoder and a trainable decoder.
Unlike PatchCore, this is a real neural network with gradient training —
expect wall times of 20-40 minutes per category vs 1-2 minutes for PatchCore.

Hyperparameters follow the executive summary and Anomalib defaults:
    - Encoder: DINOv2 ViT-Base with register tokens (frozen)
    - Target layers: [2, 3, 4, 5, 6, 7, 8, 9] (mid-level blocks)
    - Image size: 448x448 (Anomalib default, multiple of 14 for DINOv2)
    - Crop size: 392 (center crop, standard DINOv2 preprocessing)
    - Optimizer: StableAdamW (handled internally by Anomalib)
    - LR schedule: warm cosine (handled internally)
    - Max epochs: 100 (Dinomaly typically converges by epoch 80-120)

Usage:
    python -m src.c1_detector.train_dinomaly \
        --dataset mvtec_ad \
        --category bottle

    python -m src.c1_detector.train_dinomaly \
        --dataset mvtec_ad \
        --category bottle \
        --max-epochs 100 \
        --batch-size 16
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

from anomalib.models import Dinomaly
from anomalib.engine import Engine

from src.c1_detector.datamodule import (
    build_datamodule_from_manifest,
    summarise_datamodule,
)

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

CHECKPOINT_ROOT = Path("outputs/checkpoints/c1/dinomaly")
MLFLOW_TRACKING_URI = "file:./outputs/logs/mlflow"


# ─────────────────────────────────────────────────────────────────────────────
# Main train routine
# ─────────────────────────────────────────────────────────────────────────────
def train_dinomaly(
    dataset: str,
    category: str,
    image_size: tuple[int, int] = (448, 448),
    crop_size: int = 392,
    encoder_name: str = "dinov2reg_vit_base_14",
    max_epochs: int = 100,
    train_batch_size: int = 16,
    eval_batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
    device: str = "auto",
    early_stopping_patience: int = 20,
) -> dict:
    """
    Train Dinomaly on a single category and log to MLflow.
    Returns a dict of final metrics.
    """
    # ── Verify GPU ──────────────────────────────────────────────────────────
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_vram:.1f} GB VRAM)")
        # Reset peak memory tracking so we get a clean reading per run
        torch.cuda.reset_peak_memory_stats()
    else:
        logger.warning("Training on CPU — Dinomaly will be VERY slow.")

    # ── Set up MLflow ────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_name = f"c1-dinomaly-{dataset}"
    mlflow.set_experiment(experiment_name)

    run_name = f"{category}-e{max_epochs}-{image_size[0]}px"
    logger.info(f"MLflow experiment: {experiment_name}")
    logger.info(f"MLflow run name:   {run_name}")

    with mlflow.start_run(run_name=run_name) as run:
        # ── Log parameters ──────────────────────────────────────────────────
        params = {
            "model": "dinomaly",
            "dataset": dataset,
            "category": category,
            "encoder_name": encoder_name,
            "image_size_h": image_size[0],
            "image_size_w": image_size[1],
            "crop_size": crop_size,
            "max_epochs": max_epochs,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "early_stopping_patience": early_stopping_patience,
            "seed": seed,
            "device": device,
        }
        mlflow.log_params(params)
        logger.info(f"Params: {params}")

        # ── Build datamodule ────────────────────────────────────────────────
        # NOTE: Dinomaly does its own image resizing/normalisation via its
        # configure_pre_processor(). We still set image_size on the datamodule
        # for consistent loading, but Dinomaly will apply DINOv2-appropriate
        # preprocessing internally.
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
        # Dinomaly signature (Anomalib >= 2.1):
        #   Dinomaly(encoder_name=..., target_layers=..., fuse_layer_encoder=...,
        #            fuse_layer_decoder=..., remove_class_token=..., ...)
        # We accept the defaults for target_layers and fuse groupings — the
        # paper's recommended values are baked in for base-size models.
        model = Dinomaly(
            encoder_name=encoder_name,
        )
        logger.info(f"Model: Dinomaly (encoder={encoder_name})")

        # ── Set up Anomalib Engine ──────────────────────────────────────────
        checkpoint_dir = CHECKPOINT_ROOT / dataset / category
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        engine = Engine(
            max_epochs=max_epochs,
            default_root_dir=str(checkpoint_dir),
            accelerator=device,
            devices=1,
            # No image logging during training — saves time, we visualise later
            logger=False,
        )

        # ── Train ────────────────────────────────────────────────────────────
        logger.info(f"Starting Dinomaly training for {max_epochs} epochs...")
        train_start = time.time()
        engine.fit(datamodule=dm, model=model)
        train_time = time.time() - train_start
        logger.info(f"Training completed in {train_time / 60:.2f} minutes.")
        mlflow.log_metric("train_time_seconds", train_time)
        mlflow.log_metric("train_time_minutes", train_time / 60)

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
        metrics_dict = test_results[0] if test_results else {}

        logger.info("=" * 60)
        logger.info(f"RESULTS: {dataset}/{category}")
        logger.info("=" * 60)
        for metric_name, metric_value in sorted(metrics_dict.items()):
            if isinstance(metric_value, (int, float)):
                logger.info(f"  {metric_name:<40} {metric_value:.4f}")
                clean_name = metric_name.replace("/", "_").replace(" ", "_")
                mlflow.log_metric(clean_name, float(metric_value))
        logger.info("=" * 60)

        # ── Save the trained model ──────────────────────────────────────────
        # Anomalib saves under: default_root_dir/Dinomaly/<name>/weights/torch/model.pt
        # We copy it to a canonical location for downstream use (frontend, C3).
        canonical_export_dir = CHECKPOINT_ROOT / dataset / category
        canonical_export_dir.mkdir(parents=True, exist_ok=True)
        canonical_export_path = canonical_export_dir / "model.pt"

        # Find the model.pt Anomalib saved
        anomalib_torch_models = list(checkpoint_dir.rglob("weights/torch/model.pt"))
        if anomalib_torch_models:
            src = anomalib_torch_models[0]
            shutil.copy2(src, canonical_export_path)
            logger.info(f"Model weights exported to: {canonical_export_path}")
            mlflow.log_param("model_weights_path", str(canonical_export_path))
        else:
            logger.warning(
                f"Could not find torch model.pt under {checkpoint_dir}. "
                "Check Anomalib default_root_dir structure."
            )

        mlflow.log_metric("total_wall_time_seconds", train_time + test_time)
        logger.info(f"MLflow run ID: {run.info.run_id}")

        return metrics_dict


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train Dinomaly on one category using the manifest-driven pipeline."
    )
    parser.add_argument("--dataset", type=str, default="mvtec_ad",
                        choices=["mvtec_ad", "ecf", "ssgd"])
    parser.add_argument("--category", type=str, default="bottle",
                        help="Category within the dataset (e.g. 'bottle' for mvtec_ad).")
    parser.add_argument("--image-size", type=int, default=448,
                        help="Square input resolution (default: 448, multiple of 14).")
    parser.add_argument("--crop-size", type=int, default=392,
                        help="Center crop size (default: 392).")
    parser.add_argument("--encoder-name", type=str, default="dinov2reg_vit_base_14",
                        help="DINOv2 encoder variant.")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dinomaly(
        dataset=args.dataset,
        category=args.category,
        image_size=(args.image_size, args.image_size),
        crop_size=args.crop_size,
        encoder_name=args.encoder_name,
        max_epochs=args.max_epochs,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()