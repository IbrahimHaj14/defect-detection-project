import mlflow
import pandas as pd
from pathlib import Path

MLFLOW_TRACKING_URI = "file:./outputs/logs/mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Fetch experiment runs
exp = mlflow.get_experiment_by_name("c1-patchcore-mvtec_ad")
if not exp:
    raise ValueError("Experiment 'c1-patchcore-mvtec_ad' not found in MLflow.")

runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

if runs.empty:
    raise ValueError("No runs found for experiment 'c1-patchcore-mvtec_ad'.")

# Print available metric columns to assist debugging if needed
metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
print(f"Detected MLflow metric columns:\n{metric_cols}\n")

# Helper function to find matching column dynamically
def find_col(candidates):
    for c in candidates:
        if c in runs.columns:
            return c
    return None

img_auroc_col = find_col(["metrics.dataset_image_AUROC", "metrics.image_AUROC", "metrics.test_image_AUROC", "metrics.ImageAUROC", "metrics.pixel_AUROC"])
pix_auroc_col = find_col(["metrics.dataset_pixel_AUROC", "metrics.pixel_AUROC", "metrics.test_pixel_AUROC", "metrics.PixelAUROC"])
time_col = find_col(["metrics.test_time_seconds", "metrics.total_wall_time_seconds"])

category_col = "params.category"

selected_cols = [col for col in [category_col, img_auroc_col, pix_auroc_col, time_col] if col is not None]
df = runs[selected_cols].copy()

# Rename columns cleanly
rename_dict = {}
if category_col in df: rename_dict[category_col] = "Category"
if img_auroc_col in df: rename_dict[img_auroc_col] = "Image_AUROC"
if pix_auroc_col in df: rename_dict[pix_auroc_col] = "Pixel_AUROC"
if time_col in df: rename_dict[time_col] = "Test_Time_s"

df = df.rename(columns=rename_dict)
df = df.sort_values("Category").reset_index(drop=True)

# Calculate MEAN row for numeric columns
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
mean_vals = {"Category": "MEAN"}
for col in numeric_cols:
    mean_vals[col] = df[col].mean()

mean_row = pd.DataFrame([mean_vals])
df = pd.concat([df, mean_row], ignore_index=True)

# Export CSV
out_dir = Path("outputs/tables/c1")
out_dir.mkdir(parents=True, exist_ok=True)
csv_path = out_dir / "mvtec_patchcore_results.csv"
df.to_csv(csv_path, index=False)

print("=" * 60)
print(" MVTec AD PatchCore Benchmark Evaluation (Table 1)")
print("=" * 60)
print(df.to_string(index=False))
print("=" * 60)
print(f"Saved results table to: {csv_path}\n")