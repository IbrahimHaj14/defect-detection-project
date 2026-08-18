"""
scripts/format_patchcore_summary.py

Merges MVTec AD and ECF PatchCore results into a clean, dissertation-ready CSV table.
Formatted to 4 decimal places, sorted alphabetically, with clear dataset section MEANs.
"""

import csv
from pathlib import Path

MVTEC_CSV = Path("outputs/tables/c1/mvtec_patchcore_results.csv")
ECF_CSV = Path("outputs/tables/c1/ecf_patchcore_sweep.csv")
OUTPUT_CSV = Path("outputs/tables/c1/patchcore_summary.csv")


def process_dataset_rows(csv_path: Path, dataset_name: str) -> tuple[list[dict], dict]:
    """Reads CSV, filters existing MEAN rows, sorts categories, formats numbers to 4 decimal places.

    Behavior improvements:
    - Trim category names on read.
    - Robustly parse floats (ignore missing / invalid values when computing means).
    - Always return a `mean_dict` with either 4-decimal strings or "N/A" so the
      caller can reliably write a MEAN row for each dataset section.
    """
    metrics = ["image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score"]

    mean_dict = {"dataset": dataset_name, "category": "MEAN"}

    if not csv_path.exists():
        print(f"Warning: {csv_path} does not exist.")
        for m in metrics:
            mean_dict[m] = "N/A"
        return [], mean_dict

    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        # Build a mapping from normalized header -> original header for flexible matching
        headers = reader.fieldnames or []
        norm_to_orig = {}
        for h in headers:
            if h is None:
                continue
            nh = str(h).lower().replace(" ", "").replace("-", "")
            norm_to_orig[nh] = h

        # Helper to find an original header from candidate normalized names
        def find_header(candidates):
            for c in candidates:
                if c in norm_to_orig:
                    return norm_to_orig[c]
            return None

        # Map canonical metric keys to actual CSV column names when available
        col_map = {}
        col_map["image_AUROC"] = find_header(["image_auroc", "imageauroc", "image_auc"])
        col_map["image_F1Score"] = find_header(["image_f1score", "imagef1score", "image_f1", "imagef1"]) 
        col_map["pixel_AUROC"] = find_header(["pixel_auroc", "pixelauroc", "pixel_auc"])
        col_map["pixel_F1Score"] = find_header(["pixel_f1score", "pixelf1score", "pixel_f1", "pixelf1"]) 

        for row in reader:
            # Accept either 'category' or 'Category' etc.
            cat_key = None
            for candidate in ("category", "Category", "Category "):
                if candidate in row:
                    cat_key = candidate
                    break
            cat = row.get(cat_key or "category", "")
            if cat is None:
                continue
            cat = str(cat).strip()
            if not cat or cat.upper() == "MEAN":
                continue

            # Build a normalized row containing canonical metric keys
            clean = {"category": cat}
            for canonical in ["image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score"]:
                src = col_map.get(canonical)
                if src and src in row:
                    clean[canonical] = row.get(src, "")
                else:
                    clean[canonical] = ""

            rows.append(clean)

    # Sort alphabetically by trimmed category (case-insensitive)
    rows = sorted(rows, key=lambda x: x["category"].lower())

    # Calculate exact section mean (ignore missing/invalid values)
    for m in metrics:
        vals = []
        for r in rows:
            v = r.get(m, "")
            if v is None:
                continue
            s = str(v).strip()
            if s == "":
                continue
            try:
                vals.append(float(s))
            except ValueError:
                continue
        if vals:
            mean_dict[m] = f"{sum(vals) / len(vals):.4f}"
        else:
            mean_dict[m] = "N/A"

    # Format each row's numeric metrics to 4 decimal places (or "N/A")
    formatted_rows = []
    for r in rows:
        formatted = {"dataset": dataset_name, "category": r["category"]}
        for m in metrics:
            val = r.get(m, "")
            if val is None:
                formatted[m] = "N/A"
                continue
            s = str(val).strip()
            if s == "":
                formatted[m] = "N/A"
                continue
            try:
                formatted[m] = f"{float(s):.4f}"
            except ValueError:
                formatted[m] = "N/A"
        formatted_rows.append(formatted)

    return formatted_rows, mean_dict


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    mvtec_rows, mvtec_mean = process_dataset_rows(MVTEC_CSV, "MVTec AD")
    ecf_rows, ecf_mean = process_dataset_rows(ECF_CSV, "ECF")

    fieldnames = [
        "dataset",
        "category",
        "image_AUROC",
        "image_F1Score",
        "pixel_AUROC",
        "pixel_F1Score",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Write MVTec section
        for r in mvtec_rows:
            writer.writerow(r)
        if mvtec_mean:
            writer.writerow(mvtec_mean)

        # Blank divider row for clear visual separation
        writer.writerow({k: "" for k in fieldnames})

        # Write ECF section
        for r in ecf_rows:
            writer.writerow(r)
        if ecf_mean:
            writer.writerow(ecf_mean)

    print(f"Dissertation table successfully saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()