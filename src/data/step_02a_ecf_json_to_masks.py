"""
src/data/step_02a_ecf_json_to_masks.py

Converts ECF per-image JSON annotations into binary PNG ground-truth masks.
Preserves relative subdirectory paths inside ground_truth/ to eliminate mask overwrites.

Usage:
    python src/data/step_02a_ecf_json_to_masks.py --force
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_ecf_json_files(raw_dir: Path) -> list[Path]:
    """Locate all annotation JSON files in the raw ECF directory."""
    json_files = sorted(raw_dir.rglob("*.json"))
    if not json_files:
        logger.error(f"No JSON files found under {raw_dir}")
    return json_files


def get_category_and_mask_path(
    json_path: Path,
    raw_dir: Path,
    processed_dir: Path,
) -> tuple[str, Path]:
    """
    Derives category and preserved relative mask path from raw directory structure:
    - If raw root folder is 'Serious defect', category is rel_parts[1], subpath starts at rel_parts[2:].
    - Otherwise, category is rel_parts[0], subpath starts at rel_parts[1:].
    """
    rel_parts = json_path.relative_to(raw_dir).parts

    if rel_parts[0] == "Serious defect" and len(rel_parts) > 1:
        category = rel_parts[1]
        sub_parts = rel_parts[2:]
    else:
        category = rel_parts[0]
        sub_parts = rel_parts[1:]

    sub_path = Path(*sub_parts)
    stem = sub_path.stem
    if stem.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        stem = Path(stem).stem

    rel_mask_path = sub_path.parent / f"{stem}.png"

    mask_dir = processed_dir / category / "ground_truth" / rel_mask_path.parent
    mask_dir.mkdir(parents=True, exist_ok=True)

    full_mask_path = mask_dir / rel_mask_path.name
    return category, full_mask_path


def extract_shapes(data: dict | list) -> list[dict]:
    """Extract polygons/shapes from various JSON schema formats."""
    if isinstance(data, list):
        return [{"points": data, "shape_type": "polygon", "label": "defect"}]

    if not isinstance(data, dict):
        return []

    shapes = []

    if "shapes" in data and isinstance(data["shapes"], list):
        for s in data["shapes"]:
            shapes.append({
                "points": s.get("points", []),
                "shape_type": s.get("shape_type", "polygon"),
                "label": s.get("label", "defect"),
            })

    elif "objects" in data and isinstance(data["objects"], list):
        for obj in data["objects"]:
            pts = obj.get("polygon", obj.get("points", obj.get("segmentation", [])))
            shapes.append({
                "points": pts,
                "shape_type": obj.get("shape_type", "polygon"),
                "label": obj.get("label", obj.get("class", "defect")),
            })

    elif "regions" in data:
        regions = data["regions"]
        if isinstance(regions, dict):
            regions = list(regions.values())
        for reg in regions:
            shape_attrs = reg.get("shape_attributes", {})
            shape_type = shape_attrs.get("name", "polygon")
            if shape_type == "polygon":
                all_x = shape_attrs.get("all_points_x", [])
                all_y = shape_attrs.get("all_points_y", [])
                pts = list(zip(all_x, all_y))
            else:
                pts = reg.get("points", [])
            shapes.append({
                "points": pts,
                "shape_type": shape_type,
                "label": reg.get("region_attributes", {}).get("label", "defect"),
            })

    return shapes


def shape_to_mask(shape: dict, height: int, width: int) -> np.ndarray:
    """Rasterise a single shape/polygon entry into a binary mask of shape (height, width)."""
    mask = Image.new("L", (width, height), 0)
    raw_points = shape.get("points", [])
    shape_type = shape.get("shape_type", "polygon")

    if not raw_points:
        return np.array(mask, dtype=np.uint8)

    poly_points = []
    if isinstance(raw_points[0], (list, tuple)):
        poly_points = [(float(p[0]), float(p[1])) for p in raw_points if len(p) >= 2]
    elif isinstance(raw_points[0], (int, float)) and len(raw_points) >= 6:
        poly_points = [(float(raw_points[i]), float(raw_points[i + 1])) for i in range(0, len(raw_points) - 1, 2)]

    if not poly_points:
        return np.array(mask, dtype=np.uint8)

    draw = ImageDraw.Draw(mask)

    if shape_type in ("polygon", "linestrip"):
        if len(poly_points) >= 3:
            draw.polygon(poly_points, outline=1, fill=1)
        elif len(poly_points) == 2:
            draw.line(poly_points, fill=1, width=3)
    elif shape_type == "rectangle":
        if len(poly_points) >= 2:
            x0, y0 = poly_points[0]
            x1, y1 = poly_points[1]
            xmin, xmax = min(x0, x1), max(x0, x1)
            ymin, ymax = min(y0, y1), max(y0, y1)
            draw.rectangle([xmin, ymin, xmax, ymax], outline=1, fill=1)
    elif len(poly_points) >= 3:
        draw.polygon(poly_points, outline=1, fill=1)

    return np.array(mask, dtype=np.uint8)


def process_ecf_json(
    json_path: Path,
    raw_dir: Path,
    processed_dir: Path,
    manifest_rows: list[dict],
    force: bool = False,
) -> dict:
    """Parse a single ECF JSON annotation file and export its binary mask."""
    category, mask_path = get_category_and_mask_path(json_path, raw_dir, processed_dir)
    stem = mask_path.stem

    summary_entry = {
        "json_path": str(json_path),
        "mask_path": str(mask_path),
        "category": category,
        "stem": stem,
        "written": False,
        "skipped_exists": False,
        "error": False,
    }

    if mask_path.exists() and not force:
        summary_entry["skipped_exists"] = True
        return summary_entry

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading {json_path}: {e}")
        summary_entry["error"] = True
        return summary_entry

    shapes = extract_shapes(data)

    height = None
    width = None
    if isinstance(data, dict):
        height = data.get("imageHeight") or data.get("height") or data.get("image_height")
        width = data.get("imageWidth") or data.get("width") or data.get("image_width")

    if height is None or width is None:
        matching_images = [
            p for p in json_path.parent.glob(f"{stem}.*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ]
        if matching_images:
            with Image.open(matching_images[0]) as img:
                width, height = img.size
        else:
            height, width = 256, 256

    combined_mask = np.zeros((height, width), dtype=np.uint8)
    categories_present = set()

    for shape in shapes:
        label = shape.get("label", "defect")
        categories_present.add(label)
        s_mask = shape_to_mask(shape, height, width)
        combined_mask = np.maximum(combined_mask, s_mask)

    final_mask = (combined_mask * 255).astype(np.uint8)
    Image.fromarray(final_mask, mode="L").save(mask_path)
    summary_entry["written"] = True

    manifest_rows.append({
        "category": category,
        "json_filename": json_path.name,
        "mask_filename": mask_path.name,
        "mask_relative_path": str(mask_path.relative_to(processed_dir)),
        "num_defect_regions": len(shapes),
        "defect_categories": ";".join(sorted(categories_present)),
        "height": height,
        "width": width,
    })

    return summary_entry


def main():
    parser = argparse.ArgumentParser(description="Convert ECF JSON annotations to PNG masks.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/ECF-Dataset"),
        help="Directory containing raw ECF JSON files",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/ECF-Dataset"),
        help="Root directory for processed ECF dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing mask PNGs",
    )
    args = parser.parse_args()

    json_files = find_ecf_json_files(args.raw_dir)
    if not json_files:
        return

    logger.info(f"Processing {len(json_files)} ECF JSON files into category subdirectories...")

    manifest_rows = []
    summaries = []

    for jf in json_files:
        res = process_ecf_json(
            jf,
            args.raw_dir,
            args.processed_dir,
            manifest_rows,
            force=args.force,
        )
        summaries.append(res)

    manifest_path = Path("logs/ecf_mask_manifest.csv")
    manifest_path.parent.mkdir(exist_ok=True)
    if manifest_rows:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
        logger.info(f"Manifest written: {manifest_path} ({len(manifest_rows)} rows)")

    masks_written = sum(1 for s in summaries if s["written"])
    skipped_exists = sum(1 for s in summaries if s["skipped_exists"])
    errors = sum(1 for s in summaries if s["error"])

    print("\n" + "=" * 70)
    print("ECF JSON → PNG MASK CONVERSION SUMMARY")
    print("=" * 70)
    print(f"  Total JSON files found:     {len(json_files)}")
    print(f"  Masks written to disk:     {masks_written}")
    if skipped_exists:
        print(f"  Skipped (already exist):   {skipped_exists}")
    if errors:
        print(f"  Errors encountered:        {errors}")
    print(f"  Output Base Directory:     {args.processed_dir}")
    print(f"  Manifest CSV:              {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()