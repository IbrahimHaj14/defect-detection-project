"""Phase 4 class-wise LoRA training sweep."""

from __future__ import annotations

import argparse
import gc
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import mlflow
import torch
import yaml

from src.c2_synthesis.data.pair_builder import (
    build_pairs,
    get_eligible_classes,
    load_dataset_config,
)
from src.c2_synthesis.data.patch_extractor import extract_defect_crop
from src.c2_synthesis.train.train_lora_defect import (
    DEFAULT_CONFIG_PATH as DEFAULT_TRAIN_CONFIG_PATH,
    load_training_config,
    make_object_rectangle_mask,
    train_lora_defect,
)
from src.c2_synthesis.utils.image_io import load_image_rgb, load_mask_binary
from src.c2_synthesis.utils.logging_utils import get_logger
from src.c2_synthesis.utils.mlflow_utils import start_c2_run

logger = get_logger(__name__)

_C2_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SWEEP_CONFIG_PATH = _C2_ROOT / "configs" / "phase4_sweep.yaml"


@dataclass(frozen=True)
class SweepTrainingRecord:
    """One class's Phase 4 training outcome."""

    dataset: str
    class_name: str
    config_path: Path
    checkpoint_dir: Path
    elapsed_seconds: float
    peak_vram_gib: float
    reused: bool


def _repo_path(path_value: str | Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    return path if path.is_absolute() else _REPO_ROOT / path


def _class_slug(class_name: str, *, drop_numeric_prefix: bool = False) -> str:
    value = class_name
    if drop_numeric_prefix:
        value = re.sub(r"^\d+[._ -]*", "", value)
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError(f"Cannot construct a token slug from class {class_name!r}")
    return slug


def load_sweep_config(
    path: str | Path = DEFAULT_SWEEP_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate the config-driven Phase 4 sweep contract."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    required = {
        "datasets",
        "pair_budget",
        "training_steps",
        "smoke_steps",
        "candidate_count_per_class",
        "target_accepted_per_class",
        "lfs_percentile",
        "min_lfs_acceptance_rate",
        "max_lfs_acceptance_rate",
        "seed",
        "checkpoint_root",
        "synthetic_root",
        "report_path",
        "runtime_config_root",
        "base_config_path",
        "reuse_existing_checkpoints",
        "enable_masked_ti",
        "calibration_training_minutes_per_class",
        "calibration_generation_minutes_per_class",
        "max_projected_sweep_hours",
        "expected_class_count",
        "mvtec_token_template",
        "ecf_token_template",
        "defect_prompt_template",
        "mvtec_object_prompt_template",
        "ecf_object_prompt_template",
        "metric_device",
        "metric_batch_size",
        "kid_num_subsets",
        "kid_max_subset_size",
        "lpips_resolution",
        "lpips_batch_size",
    }
    if not isinstance(config, dict):
        raise ValueError(f"Phase 4 config must be a mapping: {config_path}")
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Phase 4 config is missing keys: {sorted(missing)}")
    if not config["datasets"]:
        raise ValueError("Phase 4 requires at least one dataset")
    for key in (
        "pair_budget",
        "training_steps",
        "smoke_steps",
        "candidate_count_per_class",
        "metric_batch_size",
        "kid_num_subsets",
        "kid_max_subset_size",
        "lpips_resolution",
        "lpips_batch_size",
    ):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    target = config["target_accepted_per_class"]
    if target is not None:
        if int(target) <= 0:
            raise ValueError("target_accepted_per_class must be null or positive")
        if int(target) > int(config["candidate_count_per_class"]):
            raise ValueError(
                "target_accepted_per_class cannot exceed candidate_count_per_class"
            )
    if not isinstance(config["enable_masked_ti"], bool):
        raise ValueError("enable_masked_ti must be a boolean")
    if int(config["training_steps"]) < int(config["smoke_steps"]):
        raise ValueError("training_steps must be at least smoke_steps")
    minimum = float(config["min_lfs_acceptance_rate"])
    maximum = float(config["max_lfs_acceptance_rate"])
    if not 0.0 <= minimum <= maximum <= 1.0:
        raise ValueError("Phase 4 LFS acceptance bounds must lie in [0,1]")
    if not 0.0 <= float(config["lfs_percentile"]) < 100.0:
        raise ValueError("lfs_percentile must lie in [0,100)")
    return config


def select_production_pairs(
    dataset: str,
    class_name: str,
    pair_budget: int,
    *,
    seed: int,
    base_training_config: Mapping[str, Any] | None = None,
) -> list[tuple[str, str]]:
    """Select the first deterministic pairs satisfying all training contracts.

    Whole-frame MVTec pairs require no additional filtering. ECF pairs are
    dropped if the approved 256 crop cannot be formed or if no object rectangle
    can meet the configured 1.2x--4x bbox-area bounds. Every drop is logged and
    the scan continues until the configured production budget is filled.
    """

    if pair_budget <= 0:
        raise ValueError("pair_budget must be positive")
    training_config = dict(
        load_training_config(DEFAULT_TRAIN_CONFIG_PATH)
        if base_training_config is None
        else base_training_config
    )
    dataset_config = load_dataset_config(dataset)
    all_pairs, _clean_paths = build_pairs(dataset, class_name, "all")
    if str(dataset_config["generation_mode"]) == "whole":
        selected = all_pairs[:pair_budget]
    else:
        crop_size = int(dataset_config["crop_size"])
        selected: list[tuple[str, str]] = []
        eligibility_rng = random.Random(int(seed))
        for image_path, mask_path in all_pairs:
            crop = extract_defect_crop(
                load_image_rgb(_repo_path(image_path)),
                load_mask_binary(_repo_path(mask_path)),
                size=crop_size,
            )
            if crop is None:
                logger.warning(
                    "Phase 4 pair drop for %s/%s: crop is not eligible (%s)",
                    dataset,
                    class_name,
                    mask_path,
                )
                continue
            try:
                make_object_rectangle_mask(
                    crop.mask,
                    min_bbox_area_scale=float(
                        training_config["object_mask_min_bbox_area_scale"]
                    ),
                    max_bbox_area_scale=float(
                        training_config["object_mask_max_bbox_area_scale"]
                    ),
                    rng=eligibility_rng,
                )
            except ValueError as error:
                logger.warning(
                    "Phase 4 pair drop for %s/%s: object mask is ineligible "
                    "(%s; %s)",
                    dataset,
                    class_name,
                    mask_path,
                    error,
                )
                continue
            selected.append((image_path, mask_path))
            if len(selected) == pair_budget:
                break
    if not selected:
        raise RuntimeError(f"{dataset}/{class_name} has no eligible Phase 4 pairs")
    if len(selected) < pair_budget:
        logger.warning(
            "Phase 4 %s/%s has only %d eligible pairs; using all available "
            "instead of the requested budget %d",
            dataset,
            class_name,
            len(selected),
            pair_budget,
        )
    return selected


def write_runtime_training_config(
    dataset: str,
    class_name: str,
    sweep_config: Mapping[str, Any],
    *,
    pair_budget: int,
    training_steps: int,
    smoke_steps: int,
    checkpoint_root: str | Path,
    runtime_config_root: str | Path | None = None,
) -> Path:
    """Materialise one complete class config without changing source defaults."""

    dataset_config = load_dataset_config(dataset)
    runtime = load_training_config(DEFAULT_TRAIN_CONFIG_PATH)
    drop_prefix = dataset == "ecf"
    slug = _class_slug(class_name, drop_numeric_prefix=drop_prefix)
    token_template_key = "ecf_token_template" if dataset == "ecf" else "mvtec_token_template"
    object_template_key = (
        "ecf_object_prompt_template" if dataset == "ecf" else "mvtec_object_prompt_template"
    )
    runtime.update(
        {
            "dataset_key": dataset,
            "class_name": class_name,
            "pair_budget": int(pair_budget),
            "resolution": (
                int(dataset_config["crop_size"])
                if str(dataset_config["generation_mode"]) == "patch"
                else int(runtime["resolution"])
            ),
            "learned_token": str(sweep_config[token_template_key]).format(
                class_slug=slug
            ),
            "defect_prompt_template": str(sweep_config["defect_prompt_template"]),
            "object_prompt_template": str(sweep_config[object_template_key]),
            "max_steps": int(training_steps),
            "smoke_steps": int(smoke_steps),
            # NPI masked textual inversion (arXiv:2604.22850). This flag is
            # copied per class so the production choice remains ablatable.
            "enable_masked_ti": bool(sweep_config["enable_masked_ti"]),
            "seed": int(sweep_config["seed"]),
            "checkpoint_root": Path(checkpoint_root).as_posix(),
        }
    )
    root = _repo_path(
        sweep_config["runtime_config_root"]
        if runtime_config_root is None
        else runtime_config_root
    )
    output_path = root / dataset / f"{_class_slug(class_name)}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as config_file:
        yaml.safe_dump(runtime, config_file, sort_keys=False)
    load_training_config(output_path)
    return output_path


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_sweep(
    config_path: str | Path = DEFAULT_SWEEP_CONFIG_PATH,
    *,
    datasets: Sequence[str] | None = None,
    class_subsets: Mapping[str, Sequence[str]] | None = None,
    pair_budget: int | None = None,
    training_steps: int | None = None,
    smoke_steps: int | None = None,
    checkpoint_root: str | Path | None = None,
    runtime_config_root: str | Path | None = None,
    reuse_existing: bool | None = None,
    log_to_mlflow: bool = True,
) -> tuple[SweepTrainingRecord, ...]:
    """Train every requested eligible class after one mandatory timed smoke."""

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
    effective_reuse = bool(
        sweep_config["reuse_existing_checkpoints"]
        if reuse_existing is None
        else reuse_existing
    )
    if effective_budget <= 0 or effective_smoke <= 0 or effective_steps < effective_smoke:
        raise ValueError("Sweep pair/step overrides are invalid")

    base_training_config = load_training_config(DEFAULT_TRAIN_CONFIG_PATH)
    records: list[SweepTrainingRecord] = []
    smoke_complete = False
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
            class_started = time.perf_counter()
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
            checkpoint_dir = (
                _repo_path(effective_checkpoint_root) / dataset / class_name
            )
            lora_path = checkpoint_dir / "lora.safetensors"
            token_path = checkpoint_dir / "token.pt"
            if effective_reuse and lora_path.is_file() and token_path.is_file():
                logger.info("Reusing Phase 4 checkpoint for %s/%s", dataset, class_name)
                records.append(
                    SweepTrainingRecord(
                        dataset=dataset,
                        class_name=class_name,
                        config_path=runtime_path,
                        checkpoint_dir=checkpoint_dir,
                        elapsed_seconds=0.0,
                        peak_vram_gib=0.0,
                        reused=True,
                    )
                )
                continue

            if not smoke_complete:
                logger.info(
                    "Running mandatory %d-step Phase 4 smoke on %s/%s",
                    effective_smoke,
                    dataset,
                    class_name,
                )
                train_lora_defect(
                    runtime_path,
                    steps=effective_smoke,
                    enforce_smoke_gate=True,
                    log_to_mlflow=log_to_mlflow,
                    pairs_override=pairs,
                )
                smoke_complete = True
                _release_cuda()

            logger.info(
                "Training Phase 4 adapter for %s/%s (%d steps, %d pairs)",
                dataset,
                class_name,
                effective_steps,
                actual_budget,
            )
            result = train_lora_defect(
                runtime_path,
                steps=effective_steps,
                enforce_smoke_gate=False,
                log_to_mlflow=log_to_mlflow,
                pairs_override=pairs,
            )
            elapsed = time.perf_counter() - class_started
            record = SweepTrainingRecord(
                dataset=dataset,
                class_name=class_name,
                config_path=runtime_path,
                checkpoint_dir=checkpoint_dir,
                elapsed_seconds=elapsed,
                peak_vram_gib=result.peak_vram_gib,
                reused=False,
            )
            records.append(record)
            if log_to_mlflow:
                with start_c2_run(
                    f"c2-train-{dataset}",
                    f"{class_name}-phase4-summary",
                    {
                        "dataset": dataset,
                        "class": class_name,
                        "pair_budget": actual_budget,
                        "steps": effective_steps,
                        "checkpoint_dir": checkpoint_dir,
                    },
                ):
                    mlflow.log_metrics(
                        {
                            "class_wall_time_seconds": elapsed,
                            "peak_vram_gib": result.peak_vram_gib,
                            "steps_per_second": result.steps_per_second,
                        }
                    )
            print(
                f"Phase 4 training {dataset}/{class_name}: "
                f"wall={elapsed:.2f}s, peak_vram={result.peak_vram_gib:.3f} GiB"
            )
            _release_cuda()
    return tuple(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SWEEP_CONFIG_PATH)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    records = train_sweep(
        args.config,
        datasets=args.datasets,
        reuse_existing=not args.no_reuse,
        log_to_mlflow=not args.no_mlflow,
    )
    print(f"Phase 4 training sweep complete: {len(records)} classes")


if __name__ == "__main__":
    main()
