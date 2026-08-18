import logging
from pathlib import Path
from src.c1_detector.datamodule import build_datamodule_from_manifest

logging.basicConfig(level=logging.WARNING)

manifest_path = Path("data/splits/ssgd_splits.json")
categories = ["lb101", "lb201"]

print("=== Starting SSGD Datamodule Verification ===")
for cat in categories:
    try:
        dm = build_datamodule_from_manifest(
            manifest_path=manifest_path,
            category=cat,
            image_size=(256, 256),
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
        )
        dm.setup()

        # Load first test batch to verify tensor pairing
        test_loader = dm.test_dataloader()
        batch = next(iter(test_loader))

        img_shape = list(batch.image.shape)
        mask_shape = list(batch.gt_mask.shape) if hasattr(batch, "gt_mask") and batch.gt_mask is not None else "None"

        print(f"✓ {cat}: setup passed!")
        print(f"   Train: {len(dm.train_data)} | Val: {len(dm.val_data)} | Test: {len(dm.test_data)}")
        print(f"   Test Batch Image Shape: {img_shape}")
        print(f"   Test Batch Mask Shape:  {mask_shape}")
    except Exception as e:
        print(f"✗ {cat}: FAILED with error -> {e}")

print("=============================================")
