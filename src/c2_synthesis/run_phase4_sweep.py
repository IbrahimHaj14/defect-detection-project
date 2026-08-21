"""Run the approved Big Three Phase 4 train-and-generate sweep only."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from src.c2_synthesis.data.pair_builder import get_eligible_classes
from src.c2_synthesis.generate.sweep_generate_all import generate_sweep
from src.c2_synthesis.train.sweep_train_all import (
    DEFAULT_SWEEP_CONFIG_PATH,
    load_sweep_config,
    train_sweep,
)


def projected_sweep_minutes(config: dict[str, object], class_count: int) -> float:
    """Project the approved sweep from measured calibration timings."""

    per_class = float(config["calibration_training_minutes_per_class"]) + float(
        config["calibration_generation_minutes_per_class"]
    )
    return per_class * class_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SWEEP_CONFIG_PATH)
    args = parser.parse_args()

    config = load_sweep_config(args.config)
    datasets = list(config["datasets"])
    class_count = sum(len(get_eligible_classes(str(dataset))) for dataset in datasets)
    expected = int(config["expected_class_count"])
    if class_count != expected:
        raise RuntimeError(
            f"Phase 4 scope changed: discovered {class_count} eligible classes, "
            f"expected {expected}"
        )

    projection_minutes = projected_sweep_minutes(config, class_count)
    projection_hours = projection_minutes / 60.0
    limit_hours = float(config["max_projected_sweep_hours"])
    now = datetime.now().astimezone()
    print(
        f"PHASE4_PROJECTION classes={class_count} minutes={projection_minutes:.1f} "
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
    print(f"PHASE4_TRAINING_COMPLETE classes={len(training)}", flush=True)
    generation = generate_sweep(args.config)
    print(f"PHASE4_COMPLETE classes={len(generation)}", flush=True)
    # Deliberately stop here. Phase 5 requires explicit user authorisation.


if __name__ == "__main__":
    main()
