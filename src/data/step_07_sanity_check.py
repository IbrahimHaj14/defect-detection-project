"""
src/data/step_07_sanity_check.py

Pre-flight sanity check script to verify dataset loading from split manifests.

Checks:
  1. One batch loads cleanly for train/val/test (shape, dtype)
  2. Full test set iteration counts normal vs anomalous images
  3. Mask max value across test set confirms defective masks are non-zero
  4. Runs for one representative category per dataset
"""

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms.v2 as T

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4

# One representative category per dataset to verify
CHECKS = [
    {"splits_path": "data/splits/mvtec_ad_splits.json", "category": "metal_nut"},
    {"splits_path": "data/splits/ecf_splits.json",      "category": "1.scratch"},
    {"splits_path": "data/splits/ssgd_splits.json",     "category": "lb101"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class ManifestDataset(Dataset):
    """PyTorch Dataset that loads samples directly from a JSON split manifest."""

    def __init__(self, entries: list, image_size: tuple = (256, 256)):
        self.entries = entries
        self.img_transform = T.Compose([
            T.ToImage(),
            T.Resize(image_size),
            T.ToDtype(torch.float32, scale=True),
        ])
        self.mask_transform = T.Compose([
            T.ToImage(),
            T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        img_tensor = self.img_transform(image)

        mask_tensor = torch.zeros((1, *IMAGE_SIZE), dtype=torch.float32)
        if item["mask_path"] is not None:
            mask = Image.open(item["mask_path"]).convert("L")
            mask_tensor = self.mask_transform(mask)

        return {
            "image":        img_tensor,
            "mask":         mask_tensor,
            "is_anomalous": torch.tensor(item["is_anomalous"], dtype=torch.bool),
            "label":        item["label"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────
def check_first_batch(split_name: str, entries: list) -> bool:
    """Load one batch and log shape / dtype / anomalous flags."""
    if not entries:
        logger.warning(f"  [{split_name.upper()}] Empty split — skipping")
        return True

    ds = ManifestDataset(entries, IMAGE_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    batch = next(iter(loader))

    img  = batch["image"]
    mask = batch["mask"]
    anom = batch["is_anomalous"]

    logger.info(
        f"  [{split_name.upper()}] Batch OK — "
        f"image {tuple(img.shape)} {img.dtype} | "
        f"mask {tuple(mask.shape)} (min={mask.min():.1f}, max={mask.max():.1f}) | "
        f"anomalous {anom.tolist()}"
    )
    return True


def check_full_test(entries: list) -> dict:
    """
    Iterate every test batch and count normal vs anomalous images.
    Also tracks the maximum mask pixel value across all batches.

    Expected for a healthy dataset:
      - total_anomalous > 0
      - total_normal > 0
      - max_mask_value > 0  (defective masks have non-zero pixels)
    """
    ds = ManifestDataset(entries, IMAGE_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    total_normal    = 0
    total_anomalous = 0
    max_mask_value  = 0.0

    for batch in loader:
        flags = batch["is_anomalous"]
        total_normal    += (~flags).sum().item()
        total_anomalous +=   flags.sum().item()
        max_mask_value   = max(max_mask_value, batch["mask"].max().item())

    return {
        "total_normal":    total_normal,
        "total_anomalous": total_anomalous,
        "max_mask_value":  max_mask_value,
    }


def evaluate_test_results(dataset_name: str, category: str, results: dict) -> bool:
    """Log test results and flag any problems. Returns True if healthy."""
    n   = results["total_normal"]
    a   = results["total_anomalous"]
    mmv = results["max_mask_value"]

    passed = True

    if a == 0:
        logger.error(
            f"  [TEST] FAIL — 0 anomalous images loaded. "
            f"Check manifest test entries for {dataset_name}/{category}."
        )
        passed = False

    if n == 0:
        logger.error(
            f"  [TEST] FAIL — 0 normal images loaded. "
            f"AUROC cannot be computed without both classes."
        )
        passed = False

    if a > 0 and mmv == 0.0:
        logger.error(
            f"  [TEST] FAIL — anomalous images present but all masks are zero. "
            f"Mask conversion may have failed."
        )
        passed = False

    if passed:
        logger.info(
            f"  [TEST] Full iteration OK — "
            f"{n} normal | {a} anomalous | "
            f"max mask value {mmv:.4f}"
        )

    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────
def run_sanity_check(splits_path: str, category: str) -> bool:
    path = Path(splits_path)
    if not path.exists():
        logger.error(f"Manifest not found: {path}")
        return False

    logger.info(f"Loading: {path} [{category}]")
    with open(path, "r") as f:
        all_categories = json.load(f)

    if category not in all_categories:
        logger.error(f"Category '{category}' not found in {path.name}")
        logger.info(f"Available: {list(all_categories.keys())}")
        return False

    manifest = all_categories[category]
    dataset_name = manifest["dataset_name"]

    # Print stats summary from manifest
    stats = manifest["stats"]
    logger.info(
        f"  Stats — "
        f"train {stats['train']['total']} | "
        f"val {stats['val']['total']} | "
        f"test {stats['test']['total']} "
        f"({stats['test']['anomalous']} anomalous, {stats['test']['normal']} normal)"
    )

    all_passed = True

    # Check 1: one batch per split
    for split_name in ["train", "val", "test"]:
        entries = manifest["splits"][split_name]
        ok = check_first_batch(split_name, entries)
        all_passed = all_passed and ok

    # Check 2: full test iteration — the critical one
    logger.info("  Iterating full test set...")
    test_entries = manifest["splits"]["test"]
    results = check_full_test(test_entries)
    ok = evaluate_test_results(dataset_name, category, results)
    all_passed = all_passed and ok

    return all_passed


def main():
    print("\n" + "=" * 70)
    print("DATA PIPELINE SANITY CHECK")
    print("=" * 70)

    overall_results = {}

    for check in CHECKS:
        label = f"{Path(check['splits_path']).stem} / {check['category']}"
        print(f"\n--- {label} ---")
        passed = run_sanity_check(check["splits_path"], check["category"])
        overall_results[label] = "PASS" if passed else "FAIL"

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_ok = True
    for label, result in overall_results.items():
        symbol = "OK" if result == "PASS" else "!!"
        print(f"  [{symbol}] {label:<42} {result}")
        if result != "PASS":
            all_ok = False

    print()
    if all_ok:
        print("  All checks passed. Data pipeline is ready for C1 training.")
    else:
        print("  One or more checks failed. Resolve before training.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()