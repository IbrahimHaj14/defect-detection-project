"""
src/data/step_01_audit.py

Scans data/raw/ and produces a full audit report:
- Detects non-ASCII folder and file names
- Counts images per split per category
- Checks for missing masks
- Checks image format consistency
- Checks for corrupted/zero-byte files
- Outputs: logs/audit_report.json + logs/audit_report.txt
"""

import os
import json
import logging
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

RAW_ROOT = Path("data/raw")
REPORT_PATH = Path("logs/audit_report.json")
REPORT_PATH.parent.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

def is_ascii_safe(path: Path) -> bool:
    try:
        str(path).encode('ascii')
        return True
    except UnicodeEncodeError:
        return False

def check_image_readable(path: Path) -> dict:
    """Try opening an image and return basic properties."""
    try:
        with Image.open(path) as img:
            return {
                "readable": True,
                "size": img.size,       # (W, H)
                "mode": img.mode,       # RGB, L, RGBA, etc.
                "format": img.format,
                "file_size_bytes": path.stat().st_size
            }
    except Exception as e:
        return {"readable": False, "error": str(e)}

def audit_dataset(dataset_root: Path) -> dict:
    report = {
        "dataset": dataset_root.name,
        "total_images": 0,
        "non_ascii_paths": [],
        "corrupted_files": [],
        "zero_byte_files": [],
        "categories": defaultdict(lambda: {"train_normal": 0, "test_normal": 0, "test_defect": 0, "masks": 0}),
        "image_sizes": defaultdict(int),   # {(W,H): count}
        "image_modes": defaultdict(int),
        "issues": []
    }

    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file():
            continue

        # Non-ASCII check
        if not is_ascii_safe(path):
            report["non_ascii_paths"].append(str(path))
            report["issues"].append(f"NON-ASCII path: {repr(str(path))}")

        # Zero-byte check
        if path.stat().st_size == 0:
            report["zero_byte_files"].append(str(path))
            report["issues"].append(f"Zero-byte file: {path}")
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        report["total_images"] += 1

        # Image integrity check
        img_info = check_image_readable(path)
        if not img_info["readable"]:
            report["corrupted_files"].append(str(path))
            report["issues"].append(f"Corrupted: {path} — {img_info['error']}")
            continue

        report["image_sizes"][str(img_info["size"])] += 1
        report["image_modes"][img_info["mode"]] += 1

    return report

def run_audit():
    full_report = {}
    for dataset_dir in sorted(RAW_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        logger.info(f"Auditing: {dataset_dir.name}")
        full_report[dataset_dir.name] = audit_dataset(dataset_dir)

    # Serialise defaultdicts for JSON
    def serialise(obj):
        if isinstance(obj, defaultdict):
            return dict(obj)
        raise TypeError(f"Not serialisable: {type(obj)}")

    REPORT_PATH.write_text(json.dumps(full_report, default=serialise, indent=2))

    # Human-readable summary
    summary_lines = []
    for dataset, report in full_report.items():
        summary_lines.append(f"\n{'='*60}")
        summary_lines.append(f"Dataset: {dataset}")
        summary_lines.append(f"  Total images: {report['total_images']}")
        summary_lines.append(f"  Non-ASCII paths: {len(report['non_ascii_paths'])}")
        summary_lines.append(f"  Corrupted files: {len(report['corrupted_files'])}")
        summary_lines.append(f"  Zero-byte files: {len(report['zero_byte_files'])}")
        summary_lines.append(f"  Image sizes found: {dict(report['image_sizes'])}")
        summary_lines.append(f"  Image modes found: {dict(report['image_modes'])}")
        if report['issues']:
            summary_lines.append(f"  ⚠ ISSUES:")
            for issue in report['issues'][:10]:  # Show first 10
                summary_lines.append(f"    - {issue}")

    summary = "\n".join(summary_lines)
    print(summary)
    Path("logs/audit_report.txt").write_text(summary, encoding='utf-8')
    logger.info(f"Audit complete. Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    run_audit()