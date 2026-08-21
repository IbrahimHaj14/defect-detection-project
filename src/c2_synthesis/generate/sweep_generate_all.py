"""Phase 4 generation, LFS, fidelity-metric, and report sweep."""

from __future__ import annotations

import argparse
import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import mlflow
import torch

from src.c2_synthesis.data.pair_builder import (
    build_pairs,
    get_eligible_classes,
    load_dataset_config,
)
from src.c2_synthesis.data.patch_extractor import extract_defect_crop
from src.c2_synthesis.generate.generate_defects import generate_and_filter
from src.c2_synthesis.metrics.fid_kid import compute_fid_kid
from src.c2_synthesis.metrics.lpips_diversity import mean_pairwise_lpips
from src.c2_synthesis.metrics.report import FidelityRow, write_fidelity_report
from src.c2_synthesis.train.sweep_train_all import (
    DEFAULT_SWEEP_CONFIG_PATH,
    load_sweep_config,
    select_production_pairs,
    write_runtime_training_config,
)
from src.c2_synthesis.train.train_lora_defect import (
    DEFAULT_CONFIG_PATH as DEFAULT_TRAIN_CONFIG_PATH,
    load_training_config,
)
from src.c2_synthesis.utils.image_io import load_image_rgb, load_mask_binary
from src.c2_synthesis.utils.logging_utils import get_logger
from src.c2_synthesis.utils.mlflow_utils import start_c2_run

logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SweepGenerationRecord:
    """One class's synthetic-set and fidelity result."""

    dataset: str
    class_name: str
    output_dir: Path
    persisted_uids: tuple[str, ...]
    fidelity: FidelityRow
    generation_seconds: float
    metric_seconds: float
    peak_vram_gib: float


def _repo_path(path_value: str | Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    return path if path.is_absolute() else _REPO_ROOT / path


def candidate_count_for_target(target: int, percentile: float) -> int:
    """Return the smallest batch expected to retain ``target`` samples."""

    if target <= 0:
        raise ValueError("target must be positive")
    if not 0.0 <= percentile < 100.0:
        raise ValueError("percentile must lie in [0,100)")
    keep_fraction = 1.0 - percentile / 100.0
    return max(target, math.ceil(target / keep_fraction))


def _real_metric_inputs(
    dataset: str,
    class_name: str,
) -> tuple[list[str], int]:
    """Return real images within C2 scope plus the crop-exclusion count."""

    all_pairs, _clean_paths = build_pairs(dataset, class_name, "all")
    dataset_config = load_dataset_config(dataset)
    if str(dataset_config["generation_mode"]) != "patch":
        return [image_path for image_path, _ in all_pairs], 0

    crop_size = int(dataset_config["crop_size"])
    usable_paths: list[str] = []
    for image_path, mask_path in all_pairs:
        crop = extract_defect_crop(
            load_image_rgb(_repo_path(image_path)),
            load_mask_binary(_repo_path(mask_path)),
            size=crop_size,
        )
        if crop is not None:
            usable_paths.append(image_path)
    exclusion_count = len(all_pairs) - len(usable_paths)
    logger.info(
        "Crop eligibility for %s/%s: total=%d usable=%d excluded=%d",
        dataset,
        class_name,
        len(all_pairs),
        len(usable_paths),
        exclusion_count,
    )
    return usable_paths, exclusion_count


def _synthetic_paths(output_dir: Path, uids: Sequence[str]) -> list[Path]:
    paths = [output_dir / "images" / f"{uid}.png" for uid in uids]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 4 synthetic image: {missing[0]}")
    return paths


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_sweep(
    config_path: str | Path = DEFAULT_SWEEP_CONFIG_PATH,
    *,
    datasets: Sequence[str] | None = None,
    class_subsets: Mapping[str, Sequence[str]] | None = None,
    pair_budget: int | None = None,
    training_steps: int | None = None,
    smoke_steps: int | None = None,
    checkpoint_root: str | Path | None = None,
    synthetic_root: str | Path | None = None,
    runtime_config_root: str | Path | None = None,
    report_path: str | Path | None = None,
    target_accepted: int | None = None,
    candidate_count: int | None = None,
    acceptance_rate_bounds: tuple[float, float] | None = None,
    log_metrics_to_mlflow: bool = True,
) -> tuple[SweepGenerationRecord, ...]:
    """Generate accepted triples and append finite metrics for each class."""

    sweep_config = load_sweep_config(config_path)
    selected_datasets = list(sweep_config["datasets"] if datasets is None else datasets)
    effective_budget = int(sweep_config["pair_budget"] if pair_budget is None else pair_budget)
    effective_steps = int(
        sweep_config["training_steps"] if training_steps is None else training_steps
    )
    effective_smoke = int(sweep_config["smoke_steps"] if smoke_steps is None else smoke_steps)
    effective_checkpoint_root = (
        sweep_config["checkpoint_root"] if checkpoint_root is None else checkpoint_root
    )
    effective_synthetic_root = (
        sweep_config["synthetic_root"] if synthetic_root is None else synthetic_root
    )
    effective_report_path = _repo_path(
        sweep_config["report_path"] if report_path is None else report_path
    )
    configured_target = sweep_config["target_accepted_per_class"]
    effective_target = (
        None
        if target_accepted is None and configured_target is None
        else int(configured_target if target_accepted is None else target_accepted)
    )
    percentile = float(sweep_config["lfs_percentile"])
    effective_candidates = (
        int(sweep_config["candidate_count_per_class"])
        if candidate_count is None
        else int(candidate_count)
    )
    if effective_candidates <= 0:
        raise ValueError("candidate_count must be positive")
    if effective_target is not None and effective_candidates < effective_target:
        raise ValueError("candidate_count cannot be smaller than target_accepted")
    bounds = (
        (
            float(sweep_config["min_lfs_acceptance_rate"]),
            float(sweep_config["max_lfs_acceptance_rate"]),
        )
        if acceptance_rate_bounds is None
        else tuple(float(value) for value in acceptance_rate_bounds)
    )

    records: list[SweepGenerationRecord] = []
    base_training_config = load_training_config(DEFAULT_TRAIN_CONFIG_PATH)
    for dataset in selected_datasets:
        eligible = get_eligible_classes(dataset)
        requested = list(
            eligible
            if class_subsets is None or dataset not in class_subsets
            else class_subsets[dataset]
        )
        invalid = sorted(set(requested).difference(eligible))
        if invalid:
            raise ValueError(f"Ineligible Phase 4 classes for {dataset}: {invalid}")
        for class_name in requested:
            pairs = select_production_pairs(
                dataset,
                class_name,
                effective_budget,
                seed=int(sweep_config["seed"]),
                base_training_config=base_training_config,
            )
            actual_budget = len(pairs)
            runtime_path = write_runtime_training_config(
                dataset,
                class_name,
                sweep_config,
                pair_budget=actual_budget,
                training_steps=effective_steps,
                smoke_steps=effective_smoke,
                checkpoint_root=effective_checkpoint_root,
                runtime_config_root=runtime_config_root,
            )
            logger.info(
                "Generating Phase 4 set for %s/%s: %d candidates -> %s accepted",
                dataset,
                class_name,
                effective_candidates,
                "LFS-filtered" if effective_target is None else effective_target,
            )
            generation = generate_and_filter(
                candidate_count=effective_candidates,
                lfs_percentile=percentile,
                acceptance_rate_bounds=bounds,
                accepted_target=effective_target,
                train_config_path=runtime_path,
                base_config_path=sweep_config["base_config_path"],
                output_root=effective_synthetic_root,
                pairs_override=pairs,
            )
            expected_persisted = (
                len(generation.lfs.accepted)
                if effective_target is None
                else effective_target
            )
            if len(generation.persisted_uids) != expected_persisted:
                raise RuntimeError(
                    f"{dataset}/{class_name} persisted {len(generation.persisted_uids)} "
                    f"samples instead of {expected_persisted}"
                )

            metric_started = time.perf_counter()
            real_paths, crop_exclusion_count = _real_metric_inputs(dataset, class_name)
            synthetic_paths = _synthetic_paths(
                generation.output_dir,
                generation.persisted_uids,
            )
            fidelity = compute_fid_kid(
                real_paths,
                synthetic_paths,
                device=str(sweep_config["metric_device"]),
                batch_size=int(sweep_config["metric_batch_size"]),
                kid_num_subsets=int(sweep_config["kid_num_subsets"]),
                kid_max_subset_size=int(sweep_config["kid_max_subset_size"]),
                seed=int(sweep_config["seed"]),
            )
            diversity = mean_pairwise_lpips(
                synthetic_paths,
                device=str(sweep_config["metric_device"]),
                resolution=int(sweep_config["lpips_resolution"]),
                batch_size=int(sweep_config["lpips_batch_size"]),
            )
            metric_seconds = time.perf_counter() - metric_started
            row = FidelityRow(
                dataset=dataset,
                class_name=class_name,
                n_real=fidelity.n_real,
                n_synth=fidelity.n_synthetic,
                crop_exclusion_count=crop_exclusion_count,
                fid=fidelity.fid,
                kid=fidelity.kid,
                lpips_diversity=diversity,
                lfs_acceptance=generation.lfs.acceptance_rate,
            )
            write_fidelity_report([row], effective_report_path, preserve_existing=True)
            if log_metrics_to_mlflow:
                with start_c2_run(
                    f"c2-generate-{dataset}",
                    f"{class_name}-phase4-fidelity",
                    {
                        "dataset": dataset,
                        "class": class_name,
                        "n_real": fidelity.n_real,
                        "n_synth": fidelity.n_synthetic,
                        "crop_exclusion_count": crop_exclusion_count,
                        "report_path": effective_report_path,
                    },
                ):
                    mlflow.log_metrics(
                        {
                            "FID": fidelity.fid,
                            "KID": fidelity.kid,
                            "LPIPS_diversity": diversity,
                            "lfs_acceptance_rate": generation.lfs.acceptance_rate,
                            "generation_wall_time_seconds": generation.elapsed_seconds,
                            "metric_wall_time_seconds": metric_seconds,
                            "peak_vram_gib": generation.peak_vram_gib,
                        }
                    )
            record = SweepGenerationRecord(
                dataset=dataset,
                class_name=class_name,
                output_dir=generation.output_dir,
                persisted_uids=generation.persisted_uids,
                fidelity=row,
                generation_seconds=generation.elapsed_seconds,
                metric_seconds=metric_seconds,
                peak_vram_gib=generation.peak_vram_gib,
            )
            records.append(record)
            print(
                f"Phase 4 fidelity {dataset}/{class_name}: "
                f"FID={row.fid:.6f}, KID={row.kid:.6f}, "
                f"LPIPS={row.lpips_diversity:.6f}, "
                f"LFS={row.lfs_acceptance * 100.0:.2f}%, "
                f"generation={generation.elapsed_seconds:.2f}s, "
                f"metrics={metric_seconds:.2f}s, "
                f"peak_vram={generation.peak_vram_gib:.3f} GiB"
            )
            del generation
            _release_cuda()
    return tuple(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SWEEP_CONFIG_PATH)
    parser.add_argument("--datasets", nargs="+")
    args = parser.parse_args()
    records = generate_sweep(args.config, datasets=args.datasets)
    print(f"Phase 4 generation sweep complete: {len(records)} classes")


if __name__ == "__main__":
    main()
