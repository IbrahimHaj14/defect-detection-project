"""
scripts/export_patchcore_figures.py

Generates 3-panel sample visualizations (Input Image | Ground Truth Mask | Predicted Anomaly Map)
for defective samples across all 15 MVTec AD and 12 ECF categories.
Saves figure outputs to outputs/figures/c1/patchcore_examples/ at 300 DPI for Chapter 1.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.c1_detector.dataset import AnomalyDataset
from src.c1_detector.patchcore import PatchCoreDetector

OUTPUT_DIR = Path("outputs/figures/c1/patchcore_examples")

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "zipper", "wood"
]

ECF_CATEGORIES = [
    "1.scratch", "2.Indentation", "3.blade mark", "4.dirt", "5.particle",
    "b2b_flex_pcb", "contamination", "ic_flex_pcb", "ir_base",
    "metal_foreign_body", "missing_plate", "solder_bead"
]


def export_category_figure(dataset_name: str, category: str, manifest_path: str):
    """Loads dataset split, runs inference on first defective image, and exports 3-panel figure."""
    try:
        # Load dataset
        dataset = AnomalyDataset(
            manifest_path=manifest_path,
            category=category,
            split="test",
            image_size=(256, 256)
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Find first anomalous sample
        sample = None
        for batch in loader:
            if batch["label"].item() == 1:  # Anomalous
                sample = batch
                break

        if sample is None:
            print(f"Skipping {dataset_name}/{category}: No defective test sample found.")
            return

        # Train/infer model on category
        detector = PatchCoreDetector(
            backbone_name="wide_resnet50_2",
            layers=["layer2", "layer3"],
            coreset_ratio=0.01,
            num_neighbors=9
        )
        
        # Fit on train set
        train_dataset = AnomalyDataset(
            manifest_path=manifest_path,
            category=category,
            split="train",
            image_size=(256, 256)
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        detector.fit(train_loader)

        # Predict anomaly map
        images = sample["image"]
        masks = sample["mask"]
        _, anomaly_maps = detector.predict(images)

        # Prepare numpy visualizations
        img_np = images[0].permute(1, 2, 0).cpu().numpy()
        # Un-normalize image for display
        img_np = np.clip(img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)

        gt_mask = masks[0].squeeze().cpu().numpy()
        pred_map = anomaly_maps[0].squeeze().cpu().numpy()

        # Plot 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(img_np)
        axes[0].set_title("Input Image", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth Mask", fontsize=11)
        axes[1].axis("off")

        axes[2].imshow(img_np)
        axes[2].imshow(pred_map, cmap="jet", alpha=0.5)
        axes[2].set_title("Predicted Heatmap", fontsize=11)
        axes[2].axis("off")

        plt.suptitle(f"{dataset_name} — {category}", fontsize=13, y=0.98)
        plt.tight_layout()

        # Save figure
        safe_cat_name = category.replace(".", "_").replace(" ", "_")
        save_path = OUTPUT_DIR / f"{dataset_name.lower()}_{safe_cat_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    except Exception as e:
        print(f"Error processing {dataset_name}/{category}: {e}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating MVTec AD heatmap figures...")
    for cat in MVTEC_CATEGORIES:
        export_category_figure("MVTec", cat, "data/splits/mvtec_splits.json")

    print("\nGenerating ECF heatmap figures...")
    for cat in ECF_CATEGORIES:
        export_category_figure("ECF", cat, "data/splits/ecf_splits.json")

    print(f"\nAll heatmap figures exported to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()