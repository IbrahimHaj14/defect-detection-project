"""Phase 4 acceptance test: four-class train/generate/fidelity sweep."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c2_synthesis.generate.sweep_generate_all import (  # noqa: E402
    candidate_count_for_target,
    generate_sweep,
)
from src.c2_synthesis.metrics.report import FIELDNAMES  # noqa: E402
from src.c2_synthesis.train.sweep_train_all import train_sweep  # noqa: E402

CLASS_SUBSETS = {
    "mvtec_ad": ("bottle", "cable"),
    "ecf": ("1.scratch", "2.Indentation"),
}
DATASETS = tuple(CLASS_SUBSETS)
PAIR_BUDGET = 2
TRAINING_STEPS = 2
SMOKE_STEPS = 1
TARGET_ACCEPTED = 2
CANDIDATE_COUNT = 3
CHECKPOINT_ROOT = "outputs/checkpoints/c2_phase4_verify"
SYNTHETIC_ROOT = "outputs/synthetic_phase4_verify"
RUNTIME_CONFIG_ROOT = "outputs/logs/c2/phase4_verify_configs"
REPORT_PATH = REPO_ROOT / "outputs/tables/c2/fidelity.csv"


def _assert_triples(output_dir: Path, expected_uids: tuple[str, ...]) -> None:
    assert len(expected_uids) == TARGET_ACCEPTED
    for uid in expected_uids:
        image_path = output_dir / "images" / f"{uid}.png"
        mask_path = output_dir / "masks" / f"{uid}.png"
        meta_path = output_dir / "meta" / f"{uid}.json"
        assert image_path.is_file() and image_path.stat().st_size > 0
        assert mask_path.is_file() and mask_path.stat().st_size > 0
        assert meta_path.is_file() and meta_path.stat().st_size > 0
        with meta_path.open("r", encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)
        assert metadata["uid"] == uid
        assert metadata["lfs_passed"] is True
        assert math.isfinite(float(metadata["lpips_score"]))


def _assert_report() -> None:
    assert REPORT_PATH.is_file() and REPORT_PATH.stat().st_size > 0
    with REPORT_PATH.open("r", encoding="utf-8", newline="") as report_file:
        reader = csv.DictReader(report_file)
        assert tuple(reader.fieldnames or ()) == FIELDNAMES
        rows = {
            (row["dataset"], row["class"]): row
            for row in reader
            if row["dataset"] in CLASS_SUBSETS
            and row["class"] in CLASS_SUBSETS[row["dataset"]]
        }
    expected = {
        (dataset, class_name)
        for dataset, classes in CLASS_SUBSETS.items()
        for class_name in classes
    }
    assert set(rows) == expected, f"Missing Phase 4 fidelity rows: {expected - set(rows)}"
    for key, row in rows.items():
        assert int(row["n_real"]) >= 2, key
        assert int(row["n_synth"]) == TARGET_ACCEPTED, key
        assert int(row["crop_exclusion_count"]) >= 0, key
        for column in ("FID", "KID", "LPIPS_diversity", "lfs_acceptance"):
            assert math.isfinite(float(row[column])), f"Non-finite {column} for {key}"


def main() -> None:
    started_at = time.perf_counter()
    assert candidate_count_for_target(TARGET_ACCEPTED, 25.0) == CANDIDATE_COUNT

    training = train_sweep(
        datasets=DATASETS,
        class_subsets=CLASS_SUBSETS,
        pair_budget=PAIR_BUDGET,
        training_steps=TRAINING_STEPS,
        smoke_steps=SMOKE_STEPS,
        checkpoint_root=CHECKPOINT_ROOT,
        runtime_config_root=RUNTIME_CONFIG_ROOT,
        reuse_existing=False,
        log_to_mlflow=False,
    )
    assert len(training) == 4
    assert sum(not record.reused for record in training) == 4
    for record in training:
        assert (record.checkpoint_dir / "lora.safetensors").is_file()
        assert (record.checkpoint_dir / "token.pt").is_file()
        assert record.config_path.is_file()
        assert math.isfinite(record.elapsed_seconds) and record.elapsed_seconds > 0.0
        assert math.isfinite(record.peak_vram_gib) and record.peak_vram_gib > 0.0

    generation = generate_sweep(
        datasets=DATASETS,
        class_subsets=CLASS_SUBSETS,
        pair_budget=PAIR_BUDGET,
        training_steps=TRAINING_STEPS,
        smoke_steps=SMOKE_STEPS,
        checkpoint_root=CHECKPOINT_ROOT,
        synthetic_root=SYNTHETIC_ROOT,
        runtime_config_root=RUNTIME_CONFIG_ROOT,
        report_path=REPORT_PATH,
        target_accepted=TARGET_ACCEPTED,
        candidate_count=CANDIDATE_COUNT,
        acceptance_rate_bounds=(0.0, 1.0),
        log_metrics_to_mlflow=True,
    )
    assert len(generation) == 4
    for record in generation:
        _assert_triples(record.output_dir, record.persisted_uids)
        row = record.fidelity
        assert math.isfinite(row.fid) and math.isfinite(row.kid)
        assert math.isfinite(row.lpips_diversity)
        assert row.n_synth == TARGET_ACCEPTED

    _assert_report()
    elapsed = time.perf_counter() - started_at
    print("Phase 4 verification passed")
    print("Classes: mvtec_ad/{bottle,cable}, ecf/{1.scratch,2.Indentation}")
    print(f"Training: {TRAINING_STEPS} steps/class after {SMOKE_STEPS}-step smoke")
    print(f"Synthetic triples: {TARGET_ACCEPTED} accepted/class")
    print("FID/KID: finite for all four classes")
    print(f"Fidelity report: {REPORT_PATH.relative_to(REPO_ROOT).as_posix()}")
    print(f"Elapsed time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
