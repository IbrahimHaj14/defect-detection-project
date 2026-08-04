import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def restructure_mvtec(raw_root: Path, proc_root: Path):
    """Copies all flat MVTec AD categories straight into data/processed/mvtec_ad/."""
    mvtec_categories = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
        'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
        'transistor', 'wood', 'zipper'
    ]
    logging.info("--- Restructuring MVTec AD Categories ---")
    copied = 0
    target_base = proc_root / "mvtec_ad"
    
    for cat in mvtec_categories:
        cat_raw = raw_root / cat
        if not cat_raw.exists():
            continue
        cat_target = target_base / cat
        for root, _, files in os.walk(cat_raw):
            rel_path = Path(root).relative_to(cat_raw)
            for f in files:
                if f.startswith('.'):
                    continue
                src = Path(root) / f
                dst = cat_target / rel_path / f
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
    logging.info(f"MVTec AD complete. Files copied: {copied}")

def restructure_ecf(raw_root: Path, proc_root: Path):
    """Restructures ECF-Dataset into canonical Anomalib layout without dropping nested files."""
    logging.info("--- Restructuring ECF-Dataset ---")
    ecf_raw = raw_root / "ECF-Dataset"
    ecf_proc = proc_root / "ECF-Dataset"
    
    if not ecf_raw.exists():
        logging.error(f"ECF path missing: {ecf_raw}")
        return

    copied = 0
    for root, _, files in os.walk(ecf_raw):
        current_dir = Path(root)
        
        # Skip junk directories
        if "defect_samples" in current_dir.parts or "__MACOSX" in current_dir.parts:
            continue
            
        rel_parts = current_dir.relative_to(ecf_raw).parts
        if not rel_parts:
            continue

        # Extract category name (e.g., "Serious defect/b2b_flex_pcb" -> "b2b_flex_pcb")
        if rel_parts[0] == "Serious defect":
            if len(rel_parts) < 2:
                continue
            category_name = rel_parts[1]
        else:
            category_name = rel_parts[0]

        path_str_upper = str(current_dir).upper()

        for f in files:
            file_ext = f.lower()
            if file_ext.startswith('.') or file_ext in ('.db', '.txt'):
                continue

            src = current_dir / f

            # Route 1: Normal images (OK) -> train/good
            if "OK" in path_str_upper:
                if file_ext.endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                    dst = ecf_proc / category_name / "train" / "good" / f
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    copied += 1

            # Route 2: Defect images / masks (NG) -> test & ground_truth
            elif "NG" in path_str_upper:
                if file_ext.endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                    dst = ecf_proc / category_name / "test" / category_name / f
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    copied += 1
                elif file_ext.endswith('.json'):
                    dst = ecf_proc / category_name / "ground_truth" / category_name / f
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    copied += 1

    logging.info(f"ECF-Dataset complete. Files copied: {copied}")

def restructure_ssgd(raw_root: Path, proc_root: Path):
    """Restructures SSGD into processed format."""
    logging.info("--- Restructuring SSGD Dataset ---")
    ssgd_raw = raw_root / "SSGD"
    ssgd_proc = proc_root / "SSGD"
    
    if not ssgd_raw.exists():
        logging.error(f"SSGD path missing: {ssgd_raw}")
        return

    copied = 0
    for root, _, files in os.walk(ssgd_raw):
        current_dir = Path(root)
        if "__MACOSX" in current_dir.parts:
            continue

        rel_parts = current_dir.relative_to(ssgd_raw).parts
        if not rel_parts:
            continue

        board_type = rel_parts[0] # lb101 or lb201 or annotations_lb101/201

        for f in files:
            file_ext = f.lower()
            if file_ext.startswith('.'):
                continue
                
            src = current_dir / f
            
            if "annotations_" in board_type:
                clean_board = board_type.replace("annotations_", "")
                dst = ssgd_proc / clean_board / "ground_truth" / "defects" / f
            else:
                dst = ssgd_proc / board_type / "test" / "defects" / f

            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    logging.info(f"SSGD complete. Files copied: {copied}")

def main():
    raw_root = Path("data/raw")
    proc_root = Path("data/processed")

    if proc_root.exists():
        logging.info("Wiping old data/processed/ directory for a fresh execution...")
        shutil.rmtree(proc_root)

    restructure_mvtec(raw_root, proc_root)
    restructure_ecf(raw_root, proc_root)
    restructure_ssgd(raw_root, proc_root)
    logging.info("=== STEP 3 RESTRUCTURE COMPLETE FOR ALL DATASETS ===")

if __name__ == "__main__":
    main()