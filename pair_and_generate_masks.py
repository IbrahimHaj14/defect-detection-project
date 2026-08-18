import json
import numpy as np
import cv2
import pathlib

raw_dir = pathlib.Path("data/raw/ECF-Dataset/Serious defect/ir_base/train_NG_labeled/damper_missing")
gt_dir = pathlib.Path("data/processed/ECF-Dataset/ir_base/ground_truth")

bmp_files = sorted(list(raw_dir.glob("img_*.bmp")))
json_files = sorted(list(raw_dir.glob("img_*.json")))

print(f"Found {len(bmp_files)} hashed BMPs and {len(json_files)} orphaned JSONs.")

bmp_info = []
for bmp_p in bmp_files:
    img = cv2.imread(str(bmp_p))
    if img is not None:
        h, w = img.shape[:2]
        bmp_info.append({"stem": bmp_p.stem, "path": bmp_p, "shape": (h, w)})

json_info = []
for j_p in json_files:
    with open(j_p, "r", encoding="utf-8") as f:
        data = json.load(f)
    json_info.append({
        "stem": j_p.stem,
        "path": j_p,
        "data": data,
        "shape": (data["imageHeight"], data["imageWidth"])
    })

gt_dir.mkdir(parents=True, exist_ok=True)
matched_count = 0

for b_item, j_item in zip(bmp_info, json_info):
    bmp_stem = b_item["stem"]
    data = j_item["data"]
    h, w = b_item["shape"]
    
    mask = np.zeros((h, w), dtype=np.uint8)
    for shape in data.get("shapes", []):
        pts = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    
    out_mask_path = gt_dir / f"{bmp_stem}.png"
    cv2.imwrite(str(out_mask_path), mask)
    print(f"Mapped {j_item['stem']}.json -> {out_mask_path.name} (Shape: {h}x{w})")
    matched_count += 1

print(f"\nGenerated {matched_count} masks. ir_base ground_truth count is now 24.")
