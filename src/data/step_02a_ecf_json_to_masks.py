"""
src/data/step_02a_ecf_json_to_masks.py

Converts ECF per-image JSON annotations into binary PNG ground-truth masks.

Behaviour:
- One binary PNG mask (0 = background, 255 = defect) per annotated JSON
- Matches the stem name of the corresponding defective image
- Output mask path: data/processed/ECF-Dataset/ground_truth/defects/<stem>.png
- Emits a manifest CSV at logs/ecf_mask_manifest.csv for data auditing

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def find_ecf_json_files(raw_dir: Path) -> list[Path]:
    """Locate all annotation JSON files in the raw ECF directory."""
    json_files = sorted(raw_dir.rglob('*.json'))
    if not json_files:
        logger.error(f"No JSON files found under {raw_dir}")
    return json_files


def shape_to_mask(shape: dict, height: int, width: int) -> np.ndarray:
    """
    Rasterise a single shape/polygon entry into a binary mask of shape (height, width).
    """
    mask = Image.new('L', (width, height), 0)
    points = shape.get('points', [])
    shape_type = shape.get('shape_type', 'polygon')

    if not points:
        return np.array(mask, dtype=np.uint8)

    # Handle Polygon / Linestrip
    if shape_type in ('polygon', 'linestrip'):
        if len(points) >= 3:
            poly_points = [(p[0], p[1]) for p in points]
            ImageDraw.Draw(mask).polygon(poly_points, outline=1, fill=1)

    # Handle Rectangle / Bounding Box (Safe Min/Max Sorting)
    elif shape_type == 'rectangle':
        if len(points) == 2:
            x0, y0 = points[0]
            x1, y1 = points[1]
            xmin, xmax = min(x0, x1), max(x0, x1)
            ymin, ymax = min(y0, y1), max(y0, y1)
            ImageDraw.Draw(mask).rectangle([xmin, ymin, xmax, ymax], outline=1, fill=1)

    # Generic Fallback for flat coordinate lists
    elif isinstance(points[0], (int, float)) and len(points) >= 6:
        poly_points = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
        ImageDraw.Draw(mask).polygon(poly_points, outline=1, fill=1)

    return np.array(mask, dtype=np.uint8)


def process_ecf_json(
    json_path: Path,
    raw_dir: Path,
    mask_output_dir: Path,
    manifest_rows: list[dict],
    force: bool = False,
) -> dict:
    """
    Parse a single ECF JSON annotation file and export its binary mask.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stem = json_path.stem
    mask_filename = f"{stem}.png"
    mask_path = mask_output_dir / mask_filename

    summary_entry = {
        'json_path': str(json_path),
        'stem': stem,
        'written': False,
        'skipped_exists': False,
        'error': False,
    }

    if mask_path.exists() and not force:
        summary_entry['skipped_exists'] = True
        return summary_entry

    shapes = data.get('shapes', [])
    
    # If standard LabelMe 'shapes' is empty, look for custom lists
    if not shapes and isinstance(data, list):
        shapes = [{'points': data, 'shape_type': 'polygon'}]

    # Determine dimensions (first try JSON metadata, then look up image)
    height = data.get('imageHeight')
    width = data.get('imageWidth')

    if height is None or width is None:
        # Find matching image in raw_dir to extract dimensions
        matching_images = list(raw_dir.rglob(f"{stem}.*"))
        image_files = [img for img in matching_images if img.suffix.lower() in ('.jpg', '.jpeg', '.bmp', '.png')]
        
        if image_files:
            with Image.open(image_files[0]) as img:
                width, height = img.size
        else:
            # Fallback default if image not found
            height, width = 1000, 1000

    combined_mask = np.zeros((height, width), dtype=np.uint8)
    categories_present = set()

    for shape in shapes:
        label = shape.get('label', 'defect')
        categories_present.add(label)
        s_mask = shape_to_mask(shape, height, width)
        combined_mask = np.maximum(combined_mask, s_mask)

    # Scale mask binary values to standard 0 / 255
    final_mask = (combined_mask * 255).astype(np.uint8)

    # Save PNG mask
    Image.fromarray(final_mask, mode='L').save(mask_path)
    summary_entry['written'] = True

    manifest_rows.append({
        'json_filename': json_path.name,
        'mask_filename': mask_filename,
        'num_defect_regions': len(shapes),
        'defect_categories': ';'.join(sorted(categories_present)),
        'height': height,
        'width': width,
    })

    return summary_entry


def main():
    parser = argparse.ArgumentParser(description="Convert ECF JSON annotations to PNG masks.")
    parser.add_argument(
        '--raw-dir',
        type=Path,
        default=Path('data/raw/ECF-Dataset'),
        help='Directory containing raw ECF JSON files'
    )
    parser.add_argument(
        '--processed-dir',
        type=Path,
        default=Path('data/processed/ECF-Dataset'),
        help='Root directory for processed ECF dataset'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing mask PNGs'
    )
    args = parser.parse_args()

    mask_output_dir = args.processed_dir / 'ground_truth' / 'defects'
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    json_files = find_ecf_json_files(args.raw_dir)
    if not json_files:
        return

    logger.info(f"Processing {len(json_files)} ECF JSON files into {mask_output_dir}...")

    manifest_rows = []
    summaries = []

    for jf in json_files:
        res = process_ecf_json(jf, args.raw_dir, mask_output_dir, manifest_rows, force=args.force)
        summaries.append(res)

    # Write Manifest CSV
    manifest_path = Path('logs/ecf_mask_manifest.csv')
    manifest_path.parent.mkdir(exist_ok=True)
    if manifest_rows:
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
        logger.info(f"Manifest written: {manifest_path} ({len(manifest_rows)} rows)")

    masks_written = sum(1 for s in summaries if s['written'])
    skipped_exists = sum(1 for s in summaries if s['skipped_exists'])

    print('\n' + '=' * 70)
    print('ECF JSON → PNG MASK CONVERSION SUMMARY')
    print('=' * 70)
    print(f"  Total JSON files found:     {len(json_files)}")
    print(f"  Masks written to disk:     {masks_written}")
    if skipped_exists:
        print(f"  Skipped (already exist):   {skipped_exists}")
    print(f"  Output Mask Directory:     {mask_output_dir}")
    print(f"  Manifest CSV:              {manifest_path}")
    print('=' * 70)


if __name__ == '__main__':
    main()