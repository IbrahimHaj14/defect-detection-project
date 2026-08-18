import json
import numpy as np
import cv2
import pathlib

raw_dir = pathlib.Path("data/raw/ECF-Dataset/Serious defect/ir_base/train_NG_labeled")
gt_dir = pathlib.Path("data/processed/ECF-Dataset/ir_base/ground_truth")

missing_stems = {'img_55168150', 'img_92ea08d8', 'img_bf4d6b00', 'img_d2e8b39e'}
gt_dir.mkdir(parents=True, exist_ok=True)

generated_count = 0

for json_path in raw_dir.rglob("*.json"):
    if json_path.stem in missing_stems:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        img_h = data["imageHeight"]
        img_w = data["imageWidth"]
        
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        for shape in data.get("shapes", []):
            pts = np.array(shape["points"], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        
        mask_out_path = gt_dir / f"{json_path.stem}.png"
        cv2.imwrite(str(mask_out_path), mask)
        print(f"Generated missing mask: {mask_out_path.name}")
        generated_count += 1

print(f"Successfully generated {generated_count} missing masks. ir_base ground truth is now complete with 24 masks.")
