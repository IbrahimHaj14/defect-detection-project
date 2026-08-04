"""
src/data/step_02b_ssgd_coco_to_masks.py

Converts SSGD COCO-format JSON annotations into per-image binary PNG masks.

The SSGD dataset ships pixel-level ground truth as COCO JSON files rather than
individual PNG masks. Each JSON file contains 'annotations' entries with
polygon segmentation coordinates for many images, indexed by 'image_id'.
Anomalib requires per-image PNG masks, so we rasterise the polygons here.

Behaviour:
- One binary PNG mask (0 = normal, 255 = defective) per defective image
- Filename matches source image filename (e.g. lb101_00001.png)
- Also produces an all-zero mask for images that appear in 'images' with no annotations
- All defect categories collapsed into one binary mask (the model produces
  a single anomaly score per pixel; per-category masks are only needed for
  multi-class supervised detection which is not the primary paradigm here)
- Also emits a manifest CSV listing image → defect classes present, for later analysis

Usage:
    python src/data/step_02b_ssgd_coco_to_masks.py \\
        --annotations-dir data/raw/SSGD \\
        --images-root data/processed/ssgd \\
        --output-root data/processed/ssgd

Idempotent: safe to re-run; existing masks are overwritten only if --force is set.
"""

import argparse
import json
import logging
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def find_coco_files(annotations_dir: Path) -> list[Path]:
    """Locate master COCO JSON files, ignoring 5-fold CV split files (train1, val1, etc.)."""
    all_json = sorted(annotations_dir.rglob('*.json'))
    # Keep only master files (e.g. annotations_lb101.json, annotations_lb201.json)
    master_files = [
        f for f in all_json 
        if not any(fold in f.stem.lower() for fold in ['train', 'val'])
    ]
    if not master_files:
        logger.error(f"No master JSON files found under {annotations_dir}")
    return master_files


def polygon_to_mask(
    polygon_coords: list[float],
    height: int,
    width: int,
) -> np.ndarray:
    """
    Rasterise a single polygon (COCO format: flat list [x1,y1,x2,y2,...])
    into a binary numpy mask of shape (height, width).
    """
    mask = Image.new('L', (width, height), 0)
    if len(polygon_coords) < 6:
        # Fewer than 3 points — not a valid polygon
        return np.array(mask, dtype=np.uint8)
    # Convert flat list to [(x1,y1), (x2,y2), ...]
    points = [
        (polygon_coords[i], polygon_coords[i + 1])
        for i in range(0, len(polygon_coords), 2)
    ]
    ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)


def build_mask_for_image(
    annotations: list[dict],
    height: int,
    width: int,
) -> tuple[np.ndarray, set[int]]:
    """
    Combine all annotations for a single image into one binary mask.
    Returns (mask, set of category_ids present).
    """
    combined = np.zeros((height, width), dtype=np.uint8)
    categories_present = set()

    for ann in annotations:
        if ann.get('ignore', 0) == 1:
            continue
        seg = ann.get('segmentation', [])
        # COCO segmentation can be:
        #   - a flat list [x1,y1,x2,y2,...]  (polygon)
        #   - a list of polygons [[x1,y1,...], [x1,y1,...]]  (multiple polygons)
        #   - RLE dict (not expected for SSGD)
        if isinstance(seg, list) and seg and isinstance(seg[0], (int, float)):
            # Flat polygon
            polygons = [seg]
        elif isinstance(seg, list) and seg and isinstance(seg[0], list):
            # List of polygons
            polygons = seg
        else:
            # Fall back to bounding box if no segmentation
            bbox = ann.get('bbox')
            if bbox:
                x, y, w, h = bbox
                polygons = [[x, y, x + w, y, x + w, y + h, x, y + h]]
            else:
                continue

        for polygon in polygons:
            polygon_mask = polygon_to_mask(polygon, height, width)
            combined = np.maximum(combined, polygon_mask)

        categories_present.add(ann.get('category_id'))

    # Scale to 0/255 for standard PNG mask format
    return (combined * 255).astype(np.uint8), categories_present


def process_coco_file(
    coco_path: Path,
    images_root: Path,
    output_root: Path,
    manifest_rows: list[dict],
    force: bool = False,
) -> dict:
    """
    Process a single COCO JSON file. Returns a summary dict.
    """
    logger.info(f"Loading COCO file: {coco_path}")
    with open(coco_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco.get('images', [])}
    categories = {c['id']: c['name'] for c in coco.get('categories', [])}
    annotations_by_image = defaultdict(list)
    for ann in coco.get('annotations', []):
        annotations_by_image[ann['image_id']].append(ann)

    logger.info(
        f"  {len(images)} images, {len(coco.get('annotations', []))} annotations, "
        f"{len(categories)} categories"
    )

    # Determine the part (lb101 / lb201) from filename or annotations
    # SSGD convention: file names like 'annotations_lb101.json' → part 'lb101'
    part = None
    for token in coco_path.stem.split('_'):
        if token.startswith('lb'):
            part = token
            break
    if part is None:
        # Fall back: use JSON filename stem
        part = coco_path.stem

    # Output goes to data/processed/ssgd/<part>/ground_truth/defects/
    mask_output_dir = output_root / part / 'ground_truth' / 'defects'
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'coco_file': str(coco_path),
        'part': part,
        'total_images_in_coco': len(images),
        'defective_images': 0,
        'normal_images': 0,
        'masks_written': 0,
        'skipped_missing_source': 0,
    }

    for image_id, img_info in images.items():
        filename = img_info['file_name']
        # COCO 'height' and 'width' fields — required to size the mask correctly
        height = img_info.get('height')
        width = img_info.get('width')

        if height is None or width is None:
            # Read from the actual image file
            source_image_path = images_root / part / 'test' / 'defects' / filename
            if not source_image_path.exists():
                summary['skipped_missing_source'] += 1
                continue
            with Image.open(source_image_path) as img:
                width, height = img.size

        anns = annotations_by_image.get(image_id, [])
        mask_filename = Path(filename).stem + '.png'
        mask_path = mask_output_dir / mask_filename

        if mask_path.exists() and not force:
            continue

        if anns:
            mask_array, categories_present = build_mask_for_image(anns, height, width)
            category_names = sorted(
                categories.get(cid, f'cat_{cid}') for cid in categories_present
            )
            summary['defective_images'] += 1
        else:
            # No annotations → all-zero mask (image is normal)
            mask_array = np.zeros((height, width), dtype=np.uint8)
            category_names = []
            summary['normal_images'] += 1

        Image.fromarray(mask_array, mode='L').save(mask_path)
        summary['masks_written'] += 1

        manifest_rows.append({
            'part': part,
            'image_filename': filename,
            'mask_filename': mask_filename,
            'is_defective': bool(anns),
            'defect_categories': ';'.join(category_names),
            'num_defect_regions': len(anns),
        })

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--annotations-dir',
        type=Path,
        default=Path('data/raw/SSGD'),
        help='Directory containing SSGD COCO JSON files'
    )
    parser.add_argument(
        '--images-root',
        type=Path,
        default=Path('data/processed/SSGD'),
        help='Root of processed SSGD images (contains lb101/, lb201/)'
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path('data/processed/SSGD'),
        help='Root for output masks (will write to <part>/ground_truth/defects/)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing masks'
    )
    args = parser.parse_args()

    coco_files = find_coco_files(args.annotations_dir)
    if not coco_files:
        return

    manifest_rows = []
    all_summaries = []

    for coco_file in coco_files:
        summary = process_coco_file(
            coco_file,
            images_root=args.images_root,
            output_root=args.output_root,
            manifest_rows=manifest_rows,
            force=args.force,
        )
        all_summaries.append(summary)

    # Write manifest CSV
    manifest_path = Path('logs/ssgd_mask_manifest.csv')
    manifest_path.parent.mkdir(exist_ok=True)
    if manifest_rows:
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
        logger.info(f"Manifest written: {manifest_path} ({len(manifest_rows)} rows)")

    # Print summary
    print('\n' + '=' * 70)
    print('SSGD COCO → PNG MASK CONVERSION SUMMARY')
    print('=' * 70)
    for s in all_summaries:
        print(f"\n  Part: {s['part']}")
        print(f"    Total images in COCO:     {s['total_images_in_coco']}")
        print(f"    Defective (with anns):    {s['defective_images']}")
        print(f"    Normal (no anns):         {s['normal_images']}")
        print(f"    Masks written to disk:    {s['masks_written']}")
        if s['skipped_missing_source']:
            print(f"    ⚠ Skipped (source missing): {s['skipped_missing_source']}")

    total_masks = sum(s['masks_written'] for s in all_summaries)
    total_defective = sum(s['defective_images'] for s in all_summaries)
    print(f"\n  TOTAL masks written:  {total_masks}")
    print(f"  TOTAL defective:      {total_defective}")
    print(f"\n  Manifest CSV: {manifest_path}")
    print('=' * 70)


if __name__ == '__main__':
    main()