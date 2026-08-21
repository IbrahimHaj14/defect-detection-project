"""Phase 3 acceptance test for composited generation and LFS filtering."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c2_synthesis.data.mask_bank import (  # noqa: E402
    generate_masks,
    real_mask_transplant,
)
from src.c2_synthesis.generate.generate_defects import (  # noqa: E402
    assert_pixel_composite_invariant,
    generate_and_filter,
    save_samples_grid,
)
from src.c2_synthesis.generate.low_fidelity_selection import filter_batch  # noqa: E402
from src.c2_synthesis.train.train_lora_defect import load_training_config  # noqa: E402

CANDIDATE_COUNT = 30
LFS_PERCENTILE = 25.0
MIN_ACCEPTANCE_RATE = 0.30
MAX_ACCEPTANCE_RATE = 0.90
GRID_COLUMNS = 6
GRID_THUMBNAIL_SIZE = 192
META_KEYS = {
    "uid",
    "dataset",
    "class",
    "source_clean_image",
    "source_mask",
    "generation_mode",
    "crop_offset",
    "crop_bbox",
    "lora_checkpoint",
    "token",
    "seed",
    "num_inference_steps",
    "guidance_scale",
    "lpips_score",
    "lfs_passed",
}
VERIFY_OUTPUT_ROOT = REPO_ROOT / "outputs/synthetic_phase3_verify"


def _verify_mask_bank() -> None:
    clean = Image.new("RGB", (64, 48), color=(128, 128, 128))
    source = np.zeros((32, 32), dtype=np.uint8)
    source[8:13, 10:17] = 1
    transplanted = real_mask_transplant(clean, source, (20, 21))
    assert transplanted.shape == (48, 64)
    assert transplanted.dtype == np.uint8
    assert set(np.unique(transplanted)).issubset({0, 1})
    assert int(transplanted.sum()) == int(source.sum())
    assert transplanted[21:26, 20:27].all()
    try:
        generate_masks([clean], count=1, seed=42)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Optional mask-generator stub must fail explicitly in Phase 3")


def _verify_lfs_percentile_contract() -> None:
    samples = [
        {
            "generated_image": np.full((8, 8, 3), value, dtype=np.uint8),
            "clean_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "mask": np.ones((8, 8), dtype=np.uint8),
        }
        for value in (1, 2, 3, 4)
    ]

    def deterministic_score(generated: object, _clean: object, _mask: object) -> float:
        return float(np.asarray(generated).mean())

    result = filter_batch(samples, percentile=25.0, scorer=deterministic_score)
    assert math.isclose(result.threshold, 1.75)
    assert math.isclose(result.acceptance_rate, 0.75)
    assert len(result.accepted) == 3 and len(result.rejected) == 1


def _validate_metadata(metadata: dict[str, object], expected_uid: str) -> None:
    assert set(metadata) == META_KEYS, f"Metadata schema mismatch for {expected_uid}"
    assert metadata["uid"] == expected_uid
    assert metadata["dataset"] == "mvtec_ad"
    assert metadata["class"] == "bottle"
    assert metadata["generation_mode"] == "whole"
    assert metadata["crop_offset"] == [0, 0]
    assert metadata["crop_bbox"] == [0, 0, 512, 512]
    assert isinstance(metadata["source_clean_image"], str)
    assert isinstance(metadata["source_mask"], str)
    assert Path(str(metadata["source_clean_image"])).as_posix() == metadata["source_clean_image"]
    assert Path(str(metadata["source_mask"])).as_posix() == metadata["source_mask"]
    assert metadata["token"] == "<mvtec-bottle-defect>"
    assert isinstance(metadata["seed"], int)
    assert metadata["num_inference_steps"] == 50
    assert math.isclose(float(metadata["guidance_scale"]), 7.5)
    assert math.isfinite(float(metadata["lpips_score"]))
    assert float(metadata["lpips_score"]) >= 0.0
    assert metadata["lfs_passed"] is True


def _assert_mlflow_acceptance_metric(expected_rate: float) -> str:
    mlflow.set_tracking_uri("file:./outputs/logs/mlflow")
    experiment = mlflow.get_experiment_by_name("c2-generate-mvtec_ad")
    assert experiment is not None, "Generation MLflow experiment was not created"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'bottle-30candidates'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    assert runs, "Generation MLflow run was not recorded"
    metrics = runs[0].data.metrics
    assert "lfs_acceptance_rate" in metrics
    assert math.isclose(metrics["lfs_acceptance_rate"], expected_rate, abs_tol=1.0e-12)
    assert metrics["candidates"] == CANDIDATE_COUNT
    return runs[0].info.run_id


def main() -> None:
    started_at = time.perf_counter()
    train_config = load_training_config()
    assert train_config["dataset_key"] == "mvtec_ad"
    assert train_config["class_name"] == "bottle"
    checkpoint_dir = (
        REPO_ROOT
        / "outputs"
        / "checkpoints"
        / "c2"
        / "mvtec_ad"
        / "bottle"
    )
    assert (checkpoint_dir / "lora.safetensors").is_file()
    assert (checkpoint_dir / "token.pt").is_file()

    _verify_mask_bank()
    _verify_lfs_percentile_contract()
    result = generate_and_filter(
        candidate_count=CANDIDATE_COUNT,
        lfs_percentile=LFS_PERCENTILE,
        acceptance_rate_bounds=(MIN_ACCEPTANCE_RATE, MAX_ACCEPTANCE_RATE),
        output_root=VERIFY_OUTPUT_ROOT,
    )
    assert len(result.candidates) == CANDIDATE_COUNT
    assert len(result.lfs.decisions) == CANDIDATE_COUNT
    assert 0.0 < result.lfs.acceptance_rate <= 1.0
    assert MIN_ACCEPTANCE_RATE <= result.lfs.acceptance_rate <= MAX_ACCEPTANCE_RATE
    assert len(result.persisted_uids) == len(result.lfs.accepted)

    decisions = {decision.sample.uid: decision for decision in result.lfs.decisions}
    for candidate in result.candidates:
        assert candidate.generated_image.mode == "RGB"
        assert candidate.generated_image.size == (512, 512)
        assert candidate.mask.shape == (512, 512)
        assert candidate.mask.any()
        assert_pixel_composite_invariant(
            candidate.generated_image,
            candidate.clean_image,
            candidate.mask,
        )

        image_path = result.output_dir / "images" / f"{candidate.uid}.png"
        mask_path = result.output_dir / "masks" / f"{candidate.uid}.png"
        meta_path = result.output_dir / "meta" / f"{candidate.uid}.json"
        if decisions[candidate.uid].accepted:
            assert image_path.is_file() and mask_path.is_file() and meta_path.is_file()
            with Image.open(image_path) as generated_file:
                persisted_image = generated_file.convert("RGB").copy()
            with Image.open(mask_path) as mask_file:
                persisted_mask = np.asarray(mask_file.convert("L"), dtype=np.uint8)
            assert set(np.unique(persisted_mask)).issubset({0, 255})
            assert np.array_equal(
                np.asarray(persisted_image, dtype=np.uint8),
                np.asarray(candidate.generated_image, dtype=np.uint8),
            )
            assert_pixel_composite_invariant(
                persisted_image,
                candidate.clean_image,
                persisted_mask,
            )
            with meta_path.open("r", encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)
            _validate_metadata(metadata, candidate.uid)
            assert math.isclose(
                float(metadata["lpips_score"]),
                decisions[candidate.uid].score,
                abs_tol=1.0e-12,
            )
        else:
            assert not image_path.exists() and not mask_path.exists() and not meta_path.exists()

    temporary_files = list(result.output_dir.rglob("*.tmp.*"))
    assert not temporary_files, f"Atomic-write temporary files remain: {temporary_files}"

    grid_path = save_samples_grid(
        result,
        REPO_ROOT / "outputs/figures/c2/phase3_verify_samples_bottle.png",
        columns=GRID_COLUMNS,
        thumbnail_size=GRID_THUMBNAIL_SIZE,
    )
    assert grid_path.is_file() and grid_path.stat().st_size > 0
    with Image.open(grid_path) as grid:
        assert grid.size == (
            GRID_COLUMNS * GRID_THUMBNAIL_SIZE,
            5 * (GRID_THUMBNAIL_SIZE + max(18, GRID_THUMBNAIL_SIZE // 9)),
        )
        dpi = grid.info.get("dpi")
        assert dpi is not None and all(abs(float(value) - 300.0) < 1.0 for value in dpi)

    run_id = _assert_mlflow_acceptance_metric(result.lfs.acceptance_rate)
    elapsed_seconds = time.perf_counter() - started_at
    print("Phase 3 verification passed")
    print(f"Candidates generated: {len(result.candidates)}")
    print(f"Accepted triples: {len(result.persisted_uids)}")
    print(f"Rejected candidates: {len(result.lfs.rejected)}")
    print(f"LFS percentile: {result.lfs.percentile:.2f}")
    print(f"LFS threshold: {result.lfs.threshold:.6f}")
    print(f"LFS acceptance rate: {result.lfs.acceptance_rate * 100.0:.2f}%")
    print("Pixel-composite invariant: PASS for all 30 candidates and persisted outputs")
    print(f"MLflow run ID: {run_id}")
    print(f"Peak VRAM: {result.peak_vram_gib:.3f} GiB")
    print(f"Synthetic output: {result.output_dir.relative_to(REPO_ROOT).as_posix()}")
    print(f"Samples grid: {grid_path.relative_to(REPO_ROOT).as_posix()}")
    print(f"Elapsed time: {elapsed_seconds:.2f} s")


if __name__ == "__main__":
    main()
