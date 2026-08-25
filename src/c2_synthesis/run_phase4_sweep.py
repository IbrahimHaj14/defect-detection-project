"""Run the approved Big Three Phase 4 train-and-generate sweep only."""

from __future__ import annotations

import argparse
import hashlib
import math
import statistics
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from src.c2_synthesis.data.pair_builder import get_eligible_classes
from src.c2_synthesis.generate.sweep_generate_all import (
    SweepGenerationRecord,
    generate_sweep,
)
from src.c2_synthesis.train.sweep_train_all import (
    DEFAULT_SWEEP_CONFIG_PATH,
    SweepTrainingRecord,
    load_sweep_config,
    train_sweep,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_path(path_value: str | Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    return path if path.is_absolute() else _REPO_ROOT / path


def projected_sweep_minutes(
    config: dict[str, object],
    *,
    training_class_count: int,
    generation_class_count: int,
) -> float:
    """Project resume training plus all-class generation from measured timings."""

    training_minutes = float(config["resume_training_minutes_per_class"])
    generation_minutes = float(config["calibration_generation_minutes_per_class"])
    return (
        training_minutes * training_class_count
        + generation_minutes * generation_class_count
    )


def _class_scope(config: dict[str, object]) -> list[tuple[str, str]]:
    return [
        (str(dataset), class_name)
        for dataset in config["datasets"]
        for class_name in get_eligible_classes(str(dataset))
    ]


def _checkpoint_paths(
    config: dict[str, object], dataset: str, class_name: str
) -> tuple[Path, Path]:
    root = _repo_path(str(config["checkpoint_root"])) / dataset / class_name
    return root / "lora.safetensors", root / "token.pt"


def _checkpoint_fingerprint(paths: Iterable[Path]) -> dict[Path, tuple[int, int, str]]:
    fingerprint: dict[Path, tuple[int, int, str]] = {}
    for path in paths:
        stat = path.stat()
        fingerprint[path] = (
            stat.st_size,
            stat.st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return fingerprint


def _assert_fingerprint_unchanged(
    before: dict[Path, tuple[int, int, str]],
) -> None:
    after = _checkpoint_fingerprint(before)
    changed = [path for path in before if before[path] != after[path]]
    if changed:
        raise RuntimeError(
            "Resume modified protected completed checkpoints: "
            + ", ".join(path.as_posix() for path in changed)
        )


def _write_phase4_summary(
    config: dict[str, object],
    training: tuple[SweepTrainingRecord, ...],
    generation: tuple[SweepGenerationRecord, ...],
) -> Path:
    """Write the requested per-class Phase 4 quality and fallback summary."""

    training_by_class = {
        (record.dataset, record.class_name): record for record in training
    }
    generation_keys = {(record.dataset, record.class_name) for record in generation}
    if generation_keys != set(training_by_class):
        raise RuntimeError("Training and generation class sets differ in Phase 4 summary")

    kid_values = [record.fidelity.kid for record in generation]
    if not all(math.isfinite(value) for value in kid_values):
        raise RuntimeError("Cannot summarise non-finite Phase 4 KID values")
    q1, _median, q3 = statistics.quantiles(
        kid_values, n=4, method="inclusive"
    )
    multiplier = float(config["kid_outlier_iqr_multiplier"])
    iqr = q3 - q1
    kid_lower = q1 - multiplier * iqr
    kid_upper = q3 + multiplier * iqr
    lfs_min = float(config["min_lfs_acceptance_rate"])
    lfs_max = float(config["max_lfs_acceptance_rate"])
    low_count_threshold = math.ceil(
        lfs_min * int(config["candidate_count_per_class"])
    )

    lines = [
        "Phase 4 Big Three Summary",
        f"generated_at={datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"classes={len(generation)} candidates_per_class={config['candidate_count_per_class']}",
        f"lfs_expected_band={lfs_min:.2f}-{lfs_max:.2f}",
        f"low_accepted_threshold={low_count_threshold}",
        (
            "kid_outlier_fence="
            f"[{kid_lower:.9f},{kid_upper:.9f}] "
            f"(Tukey {multiplier:.2f}x IQR)"
        ),
        "",
        (
            "dataset | class | pairs_used | object_loss_fallback_count | "
            "lfs_acceptance_rate | accepted_sample_count | FID | KID | "
            "LPIPS_diversity | flags"
        ),
    ]
    for record in generation:
        key = (record.dataset, record.class_name)
        train_record = training_by_class[key]
        row = record.fidelity
        accepted_count = len(record.persisted_uids)
        flags: list[str] = []
        if not lfs_min <= row.lfs_acceptance <= lfs_max:
            flags.append("LFS_OUTSIDE_EXPECTED_BAND")
        if accepted_count < low_count_threshold:
            flags.append("LOW_ACCEPTED_COUNT")
        if row.kid < kid_lower or row.kid > kid_upper:
            flags.append("KID_OUTLIER")
        lines.append(
            " | ".join(
                [
                    record.dataset,
                    record.class_name,
                    str(train_record.pairs_used),
                    str(train_record.object_mask_fallback_count),
                    f"{row.lfs_acceptance:.6f}",
                    str(accepted_count),
                    f"{row.fid:.9f}",
                    f"{row.kid:.9f}",
                    f"{row.lpips_diversity:.9f}",
                    ",".join(flags) if flags else "OK",
                ]
            )
        )

    output_path = _repo_path(str(config["summary_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


def _dvc_finalize(config: dict[str, object]) -> None:
    if not bool(config["enable_dvc_finalize"]):
        return
    targets = [str(value) for value in config["dvc_targets"]]
    subprocess.run(
        [sys.executable, "-m", "dvc", "add", *targets],
        cwd=_REPO_ROOT,
        check=True,
    )
    print(f"PHASE4_DVC_COMPLETE targets={','.join(targets)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SWEEP_CONFIG_PATH)
    args = parser.parse_args()

    config = load_sweep_config(args.config)
    scope = _class_scope(config)
    class_count = len(scope)
    expected = int(config["expected_class_count"])
    if class_count != expected:
        raise RuntimeError(
            f"Phase 4 scope changed: discovered {class_count} eligible classes, "
            f"expected {expected}"
        )

    reuse_enabled = bool(config["reuse_existing_checkpoints"])
    completed: list[tuple[str, str]] = []
    missing: list[tuple[str, str]] = []
    protected_paths: list[Path] = []
    for dataset, class_name in scope:
        lora_path, token_path = _checkpoint_paths(config, dataset, class_name)
        if reuse_enabled and lora_path.is_file() and token_path.is_file():
            completed.append((dataset, class_name))
            protected_paths.extend((lora_path, token_path))
        else:
            missing.append((dataset, class_name))
    print(
        "PHASE4_RESUME_SKIP="
        + ",".join(f"{dataset}/{class_name}" for dataset, class_name in completed),
        flush=True,
    )
    print(
        "PHASE4_RESUME_TRAIN="
        + ",".join(f"{dataset}/{class_name}" for dataset, class_name in missing),
        flush=True,
    )
    if len(completed) != 13 or len(missing) != 13:
        raise RuntimeError(
            f"Unexpected resume state: completed={len(completed)}, missing={len(missing)}"
        )
    protected_fingerprint = _checkpoint_fingerprint(protected_paths)

    projection_minutes = projected_sweep_minutes(
        config,
        training_class_count=len(missing),
        generation_class_count=class_count,
    )
    projection_hours = projection_minutes / 60.0
    limit_hours = float(config["max_projected_sweep_hours"])
    now = datetime.now().astimezone()
    print(
        f"PHASE4_PROJECTION train_classes={len(missing)} "
        f"generate_classes={class_count} minutes={projection_minutes:.1f} "
        f"hours={projection_hours:.3f} limit_hours={limit_hours:.1f}",
        flush=True,
    )
    if projection_hours > limit_hours:
        raise RuntimeError(
            f"Projected Phase 4 wall time {projection_hours:.3f}h exceeds "
            f"the {limit_hours:.3f}h launch limit"
        )
    print(
        "PHASE4_ESTIMATED_COMPLETION="
        f"{(now + timedelta(minutes=projection_minutes)).isoformat(timespec='seconds')}",
        flush=True,
    )

    training = train_sweep(args.config)
    _assert_fingerprint_unchanged(protected_fingerprint)
    print(f"PHASE4_TRAINING_COMPLETE classes={len(training)}", flush=True)
    for record in training:
        print(
            "PHASE4_OBJECT_FALLBACK "
            f"dataset={record.dataset} class={record.class_name} "
            f"count={record.object_mask_fallback_count}",
            flush=True,
        )
    generation = generate_sweep(args.config)
    _assert_fingerprint_unchanged(protected_fingerprint)
    summary_path = _write_phase4_summary(config, training, generation)
    print(f"PHASE4_SUMMARY={summary_path.as_posix()}", flush=True)
    _dvc_finalize(config)
    print(f"PHASE4_COMPLETE classes={len(generation)}", flush=True)
    # Deliberately stop here. Phase 5 requires explicit user authorisation.


if __name__ == "__main__":
    main()
