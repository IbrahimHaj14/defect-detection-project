import os
import shutil
from pathlib import Path
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- THE TRANSLATION MAP ---
EXPLICIT_MAP = {
    # Folders (including the variations found in logs)
    "1.NG训练集-已标注": "train_NG_labeled",
    "2.NG测试集-已标注": "test_NG_labeled",
    "1.NG集-已标注": "train_NG_labeled",
    "2.OK集": "test_OK",
    "3.OK测试集": "test_OK",
    "3.缺陷示意": "defect_samples",
    "3.缺陷示例": "defect_samples",
    "4.缺陷示意": "defect_samples",
    
    # Specific Defect Types
    "1.漏焊+断焊": "pseudo_broken_solder",
    "2.金属异物": "metal_foreign_body",
    "3.IRBASE": "ir_base",
    "4.B2B软板器材": "b2b_flex_pcb",
    "5.IC软板器材": "ic_flex_pcb",
    "6.板材缺失": "missing_plate",
    "7.露锡珠少胶": "solder_bead",
    "8.大面积脏污": "contamination",
    "大面积缺失": "large_area_missing",
    "阻尼器缺失": "damper_missing",
    "画线": "drawn_line",
    
    # Algorithm related
    "算法框": "algo_box",
    "程序3 相机1 算法框30": "proc3_cam1_box30"
}

# Add these to your EXPLICIT_MAP in step_02_rename.py
EXPLICIT_MAP.update({
    f"算法框{i}": f"algo_box_{i}" for i in range(1, 13)
})

def sanitize_ecf_dataset(root_path):
    root = Path(root_path)
    if not root.exists():
        logging.error(f"Path not found: {root_path}")
        return

    # Walk through the directory (bottom-up is crucial when renaming)
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        current_path = Path(dirpath)
        
        # 1. Rename Folders
        for d in dirnames:
            if d in EXPLICIT_MAP:
                old_dir = current_path / d
                new_dir = current_path / EXPLICIT_MAP[d]
                logging.info(f"Renaming Folder: {d} -> {EXPLICIT_MAP[d]}")
                os.rename(old_dir, new_dir)
            elif any(ord(char) > 127 for char in d):
                logging.warning(f"UNMAPPED CHINESE FOLDER FOUND: {d}")

        # 2. Rename Files (Active Sanitization)
        for f in filenames:
            if any(ord(char) > 127 for char in f):
                ext = Path(f).suffix
                # Generate a simple clean name using a hash of the original name
                # This ensures the new name is ASCII and unique
                new_f = f"img_{hash(f) & 0xffffffff:x}{ext}" 
                
                try:
                    os.rename(current_path / f, current_path / new_f)
                    logging.info(f"Renaming File: {f} -> {new_f}")
                except Exception as e:
                    logging.error(f"Failed to rename file {f}: {e}")

if __name__ == "__main__":
    # We target the extracted ECF folder
    target = "data/raw/ECF-Dataset"
    logging.info(f"Starting sanitization of {target}...")
    sanitize_ecf_dataset(target)
    logging.info("Sanitization complete.")