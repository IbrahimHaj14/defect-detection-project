import os
from pathlib import Path

exts = ('.png', '.jpg', '.jpeg', '.bmp', '.json')

def count_files(directory, is_raw=False):
    counts = {"ECF": 0, "MVTec": 0, "SSGD": 0}
    mvtec_cats = {'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                  'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
                  'transistor', 'wood', 'zipper'}
    
    for root, _, files in os.walk(directory):
        if "__MACOSX" in root or "defect_samples" in root:
            continue
        for f in files:
            if f.lower().endswith(exts):
                path_parts = Path(root).parts
                if "ECF-Dataset" in path_parts:
                    counts["ECF"] += 1
                elif "SSGD" in path_parts:
                    counts["SSGD"] += 1
                elif any(cat in path_parts for cat in mvtec_cats):
                    counts["MVTec"] += 1
    return counts

raw_counts = count_files("data/raw", is_raw=True)
proc_counts = count_files("data/processed")

print(f"\n{'DATASET':<12} | {'CLEAN RAW':<10} | {'PROCESSED':<10} | {'DIFFERENCE':<10}")
print("-" * 50)
for key in ["ECF", "MVTec", "SSGD"]:
    diff = raw_counts[key] - proc_counts[key]
    print(f"{key:<12} | {raw_counts[key]:<10} | {proc_counts[key]:<10} | {diff:<10}")