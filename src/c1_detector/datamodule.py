"""
src/c1_detector/datamodule.py

Bridges our split manifests (data/splits/*.json) to Anomalib's Folder datamodule.

Design rationale:
    - Anomalib v2's Folder class handles all Lightning integration cleanly.
      We do not reinvent it.
    - Our split manifests remain the source of truth for reproducibility.
      This module reads the manifest and constructs an Anomalib Folder
      datamodule pointed at the exact files the manifest specifies.
    - Training scripts import from here — no direct Anomalib imports elsewhere.

Usage:
    from src.c1_detector.datamodule import build_datamodule_from_manifest

    dm = build_datamodule_from_manifest(
        manifest_path="data/splits/mvtec_ad_splits.json",
        category="bottle",
        image_size=(256, 256),
        train_batch_size=32,
        eval_batch_size=32,
    )
    dm.setup()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from anomalib.data import Folder

logger = logging.getLogger(__name__)


def _resolve_category_root(manifest_entry: dict, category: str, dataset_name: str) -> Path:
    """
    Infer the category root directory from the manifest paths.

    Manifest entries look like:
      data/processed/mvtec_ad/bottle/train/good/000.png

    Category root:
      data/processed/mvtec_ad/bottle/

    We derive this from any train entry (they always exist) by walking
    up from image_path to strip 'train/good/<filename>'.
    """
    train_entries = manifest_entry.get("splits", {}).get("train", [])
    if not train_entries:
        raise ValueError(
            f"Cannot infer category root for {dataset_name}/{category}: "
            "no train entries in manifest."
        )

    sample_path = Path(train_entries[0]["image_path"])
    # Walk up: filename -> good -> train -> category root
    category_root = sample_path.parent.parent.parent
    if not category_root.exists():
        raise FileNotFoundError(f"Inferred category root does not exist: {category_root}")

    logger.info(f"  Category root resolved to: {category_root}")
    return category_root


def _detect_mask_layout(
    category_root: Path,
    manifest_entry: dict,
) -> tuple[str | None, bool]:
    """
    Determine the ground_truth directory layout.

    Returns:
        (mask_dir_relative_to_root, is_dataset_level)

    Two conventions supported:
        1. MVTec / SSGD-style (category-level):
           <category_root>/ground_truth/<defect_class>/<image_stem>.png
        2. ECF-style (dataset-level):
           <dataset_root>/ground_truth/defects/<image_stem>.png

    Detection strategy: check where the first non-null mask_path in the
    manifest actually lives.
    """
    for entry in manifest_entry.get("splits", {}).get("test", []):
        mask_path = entry.get("mask_path")
        if mask_path:
            mask_path_obj = Path(mask_path)
            if not mask_path_obj.exists():
                continue

            # Check if the mask is inside the category root
            try:
                relative = mask_path_obj.relative_to(category_root)
                # e.g. relative == PosixPath('ground_truth/scratch/000.png')
                return str(relative.parts[0]), False  # 'ground_truth', category-level
            except ValueError:
                # Mask is outside the category root -> dataset-level layout
                dataset_root = category_root.parent
                try:
                    relative = mask_path_obj.relative_to(dataset_root)
                    return str(relative.parent), True  # e.g. 'ground_truth/defects', dataset-level
                except ValueError:
                    logger.warning(
                        f"Mask path {mask_path_obj} is not under category or dataset root."
                    )
                    return None, False
    return None, False


def _collect_test_defect_dirs(category_root: Path) -> list[str]:
    """
    Return the list of defect subdirectories under test/, excluding 'good'.
    e.g. ['bent', 'broken_large', 'broken_small', 'contamination', 'crack']
    """
    test_dir = category_root / "test"
    if not test_dir.exists():
        return []
    return sorted(
        d.name for d in test_dir.iterdir()
        if d.is_dir() and d.name != "good"
    )


def build_datamodule_from_manifest(
    manifest_path: str | Path,
    category: str,
    image_size: tuple[int, int] = (256, 256),
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> Folder:
    """
    Construct an Anomalib Folder datamodule for a specific category from a manifest.

    Args:
        manifest_path: Path to a *_splits.json file (e.g. data/splits/mvtec_ad_splits.json).
        category: Category name within the manifest (e.g. 'bottle', 'metal_nut', 'lb101').
        image_size: (H, W) resize target for both images and masks.
        train_batch_size: Batch size for train/val dataloaders.
        eval_batch_size: Batch size for test dataloader.
        num_workers: DataLoader worker count. Reduce to 0 on Windows if you hit issues.
        seed: Random seed for any internal Anomalib splitting (should be inactive
              since we provide explicit test/val dirs).

    Returns:
        Configured but not-yet-set-up Anomalib Folder datamodule. Call .setup() before use.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        all_categories = json.load(f)

    if category not in all_categories:
        available = list(all_categories.keys())
        raise KeyError(
            f"Category '{category}' not found in {manifest_path.name}. "
            f"Available: {available}"
        )

    manifest_entry = all_categories[category]
    dataset_name = manifest_entry["dataset_name"]

    logger.info(f"Building datamodule: {dataset_name} / {category}")
    logger.info(f"  Manifest: {manifest_path}")
    stats = manifest_entry["stats"]
    logger.info(
        f"  Stats — train {stats['train']['total']} | val {stats['val']['total']} | "
        f"test {stats['test']['total']} "
        f"({stats['test']['anomalous']} anomalous, {stats['test']['normal']} normal)"
    )

    # Resolve directory paths
    category_root = _resolve_category_root(manifest_entry, category, dataset_name)
    defect_dirs = _collect_test_defect_dirs(category_root)
    mask_dir, is_dataset_level = _detect_mask_layout(category_root, manifest_entry)

    logger.info(f"  Defect dirs: {defect_dirs}")
    logger.info(f"  Mask layout: {'dataset-level' if is_dataset_level else 'category-level'}")
    logger.info(f"  Mask dir: {mask_dir}")

    # For dataset-level masks (ECF), Anomalib's Folder expects mask_dir relative
    # to root. We set root = dataset root in that case.
    if is_dataset_level:
        root = category_root.parent
        # Anomalib's Folder needs paths relative to root
        normal_dir = str((category_root / "train" / "good").relative_to(root))
        normal_test_dir = str((category_root / "test" / "good").relative_to(root))
        abnormal_dir = [
            str((category_root / "test" / d).relative_to(root))
            for d in defect_dirs
        ]
    else:
        root = category_root
        normal_dir = "train/good"
        normal_test_dir = "test/good"
        abnormal_dir = [f"test/{d}" for d in defect_dirs]

    datamodule = Folder(
        name=f"{dataset_name}_{category}",
        root=str(root),
        normal_dir=normal_dir,
        normal_test_dir=normal_test_dir,
        abnormal_dir=abnormal_dir,
        mask_dir=mask_dir,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    return datamodule


def summarise_datamodule(dm: Folder) -> None:
    """Post-setup sanity print. Call after dm.setup()."""
    logger.info(f"Datamodule '{dm.name}' ready:")
    logger.info(f"  Train samples: {len(dm.train_data)}")
    logger.info(f"  Val samples:   {len(dm.val_data)}")
    logger.info(f"  Test samples:  {len(dm.test_data)}")


if __name__ == "__main__":
    # Quick self-test: build and set up a datamodule for MVTec AD bottle
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dm = build_datamodule_from_manifest(
        manifest_path="data/splits/mvtec_ad_splits.json",
        category="bottle",
        image_size=(256, 256),
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,  # 0 for Windows compatibility during smoke test
    )
    dm.setup()
    summarise_datamodule(dm)

    # Fetch one batch from each split to confirm shapes
    for split_name, loader_fn in [
        ("train", dm.train_dataloader),
        ("val",   dm.val_dataloader),
        ("test",  dm.test_dataloader),
    ]:
        batch = next(iter(loader_fn()))
        img_shape = batch.image.shape if hasattr(batch, "image") else batch["image"].shape
        logger.info(f"  [{split_name.upper()}] batch image shape: {tuple(img_shape)}")