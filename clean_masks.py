import pathlib
import shutil

root = pathlib.Path('data/processed/ECF-Dataset')

for cat in sorted(root.iterdir()):
    if not cat.is_dir():
        continue
    gt_dir = cat / 'ground_truth'
    test_dir = cat / 'test'
    if not gt_dir.exists() or not test_dir.exists():
        continue

    test_stems = {p.stem for p in test_dir.rglob('*.*') if p.is_file()}
    all_masks = [p for p in gt_dir.rglob('*.*') if p.is_file()]

    selected = {}
    for m in all_masks:
        if m.stem in test_stems and m.stem not in selected:
            selected[m.stem] = m

    for stem, src in selected.items():
        dest = gt_dir / f"{stem}{src.suffix}"
        if src != dest:
            shutil.copy2(src, dest)

    for item in list(gt_dir.iterdir()):
        if item.is_dir():
            shutil.rmtree(item)
        elif item.is_file() and item.stem not in selected:
            item.unlink()

    count = len(list(gt_dir.glob('*.*')))
    print(f"{cat.name}: {count} masks cleanly structured in root ground_truth.")
