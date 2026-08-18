import pathlib
import shutil

root = pathlib.Path('data/processed/SSGD')

for cat in sorted(root.iterdir()):
    if not cat.is_dir():
        continue
    
    gt_dir = cat / 'ground_truth'
    test_dir = cat / 'test'
    if not gt_dir.exists() or not test_dir.exists():
        continue

    # 1. Map all active test stems (excluding 'good')
    test_stems = {p.stem for p in test_dir.rglob('*.*') if p.is_file() and 'good' not in p.parts}
    
    # 2. Match GT masks
    all_masks = [p for p in gt_dir.rglob('*.*') if p.is_file()]
    selected = {}
    for m in all_masks:
        stem = m.stem.replace('_mask', '').replace('mask_', '')
        if stem in test_stems and stem not in selected:
            selected[stem] = m

    # 3. Copy matched masks directly to gt_dir root as <stem>.png
    for stem, src in selected.items():
        dest = gt_dir / f"{stem}.png"
        if src != dest:
            shutil.copy2(src, dest)

    # 4. Remove subdirectories and unselected/orphaned files in gt_dir
    for item in list(gt_dir.iterdir()):
        if item.is_dir():
            shutil.rmtree(item)
        elif item.is_file() and item.stem not in selected:
            item.unlink()

    final_masks = len(list(gt_dir.glob('*.*')))
    print(f"{cat.name}: {final_masks} masks flattened and aligned with {len(test_stems)} test stems.")
