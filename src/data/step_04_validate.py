import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_processed_data(processed_dir: Path):
    logging.info("--- Starting Step 4: Processed Data Validation Gate ---")
    
    if not processed_dir.exists():
        logging.error(f"Processed directory not found: {processed_dir}")
        return False

    summary = {}
    corrupt_files = []
    
    # Iterate through each dataset in data/processed (ecf, mvtec_ad, ssgd)
    for dataset_path in processed_dir.iterdir():
        if not dataset_path.is_dir():
            continue
            
        dataset_name = dataset_path.name
        summary[dataset_name] = {"train_good": 0, "test_images": 0, "ground_truth": 0, "categories": 0}
        
        # Walk through sub-categories
        for cat_path in dataset_path.iterdir():
            if not cat_path.is_dir():
                continue
            
            summary[dataset_name]["categories"] += 1
            
            for root, _, files in os.walk(cat_path):
                current_path = Path(root)
                for f in files:
                    file_path = current_path / f
                    
                    # 1. Zero-byte file check
                    if file_path.stat().st_size == 0:
                        corrupt_files.append(str(file_path))
                        continue
                    
                    # 2. Count distributions
                    if "train" in current_path.parts and "good" in current_path.parts:
                        summary[dataset_name]["train_good"] += 1
                    elif "test" in current_path.parts:
                        summary[dataset_name]["test_images"] += 1
                    elif "ground_truth" in current_path.parts:
                        summary[dataset_name]["ground_truth"] += 1

    # Print Summary Table
    logging.info("\n" + "="*55)
    logging.info(f"{'DATASET':<15} | {'CATEGORIES':<10} | {'TRAIN (OK)':<10} | {'TEST':<8} | {'MASKS':<8}")
    logging.info("="*55)
    for ds_name, stats in summary.items():
        logging.info(f"{ds_name:<15} | {stats['categories']:<10} | {stats['train_good']:<10} | {stats['test_images']:<8} | {stats['ground_truth']:<8}")
    logging.info("="*55)

    if corrupt_files:
        logging.warning(f"Validation WARNING: Found {len(corrupt_files)} corrupted/empty files!")
        for cf in corrupt_files[:5]:
            logging.warning(f"  - Corrupt: {cf}")
        return False
    else:
        logging.info("SUCCESS: 0 corrupt files found. All paths are valid and ready for training!")
        return True

if __name__ == "__main__":
    PROCESSED_ROOT = Path("data/processed")
    validate_processed_data(PROCESSED_ROOT)