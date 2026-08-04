"""
src/data/step_03b_ssgd_splits.py

Organizes datasets missing a default test/good partition (SSGD & ECF):
  - Normal images: 80% -> train/good/, 20% -> test/good/
  - Defective images: 100% -> test/defects/ (or test/<defect_class>/)
"""

import csv
import random
import shutil
from pathlib import Path

MANIFEST_CSV = Path("logs/ssgd_mask_manifest.csv")
SSGD_ROOT = Path("data/processed/SSGD")
ECF_ROOT = Path("data/processed/ECF-Dataset")
SEED = 42


def organize_dataset_normals(
    dataset_root: Path,
    normal_files: dict[str, list[Path]] = None,
    split_ratio: float = 0.8,
):
    """
    Splits normal images 80% to train/good/ and 20% to test/good/.
    Moves defective images to test/.
    """
    rng = random.Random(SEED)

    for category_dir in sorted(dataset_root.iterdir()):
        if not category_dir.is_dir() or category_dir.name == "ground_truth":
            continue

        category = category_dir.name
        train_good = category_dir / "train" / "good"
        test_good = category_dir / "test" / "good"

        train_good.mkdir(parents=True, exist_ok=True)
        test_good.mkdir(parents=True, exist_ok=True)

        # Collect all normal images currently in category
        if normal_files and category in normal_files:
            normals = normal_files[category]
        else:
            # Fallback for ECF: normal images sitting in train/good or root
            normals = list(train_good.glob("*")) + list(
                (category_dir / "good").glob("*")
            )
            normals = [p for p in normals if p.is_file()]

        if not normals:
            continue

        rng.shuffle(normals)
        n_train = int(len(normals) * split_ratio)
        train_normals = normals[:n_train]
        test_normals = normals[n_train:]

        # Move to respective locations
        for p in train_normals:
            dest = train_good / p.name
            if p != dest and p.exists():
                shutil.move(str(p), str(dest))

        for p in test_normals:
            dest = test_good / p.name
            if p != dest and p.exists():
                shutil.move(str(p), str(dest))

        print(
            f"[{dataset_root.name} / {category}] Split normals -> {len(train_normals)} train/good | {len(test_normals)} test/good"
        )


def fix_ssgd_and_ecf():
    # --- 1. Process SSGD using manifest CSV ---
    if MANIFEST_CSV.exists():
        ssgd_normals = {"lb101": [], "lb201": []}
        with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                is_defective = (
                    row.get("is_defective", "False").strip().lower() == "true"
                )
                filename = row["image_filename"]
                part = row["part"]
                if not is_defective:
                    # Locate file on disk
                    found = list((SSGD_ROOT / part).rglob(filename))
                    if found:
                        ssgd_normals[part].append(found[0])

        organize_dataset_normals(
            SSGD_ROOT, normal_files=ssgd_normals, split_ratio=0.8
        )

    # --- 2. Process ECF ---
    if ECF_ROOT.exists():
        organize_dataset_normals(ECF_ROOT, split_ratio=0.8)


if __name__ == "__main__":
    fix_ssgd_and_ecf()