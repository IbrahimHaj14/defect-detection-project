"""Acceptance gate for C3 Phase 3 QLoRA fine-tuning."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

from src.c3_explanation.model.train_report_qlora import (
    load_phase3_config,
    run_cpu_preflight,
    run_training_stage,
    validate_checkpoint_artifacts,
)
from src.c3_explanation.utils.json_io import validate_against_schema

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        help="Validate final canonical artifacts after the live acceptance gate passed.",
    )
    return parser.parse_args()


def _verify_canonical(config: dict) -> None:
    for dataset in ("mvtec_ad", "ecf"):
        checkpoint = REPO_ROOT / config["data"]["datasets"][dataset]["checkpoint_dir"]
        if not checkpoint.is_dir():
            raise AssertionError(f"Canonical checkpoint is missing: {checkpoint}")
        validate_checkpoint_artifacts(checkpoint, config)
        print(f"CANONICAL_CHECKPOINT_VALID dataset={dataset} path={checkpoint}")


def main() -> int:
    args = _parse_args()
    config = load_phase3_config()
    assert int(config["training"]["num_workers"]) == 0
    assert int(config["training"]["batch_size"]) == 1
    assert int(config["training"]["gradient_accumulation_steps"]) == 8
    if args.canonical_only:
        _verify_canonical(config)
        print("C3_PHASE_3_CANONICAL_GATE_PASS")
        return 0

    preflight = run_cpu_preflight(config)
    assert preflight["mvtec_ad"].examples == 1008
    assert preflight["ecf"].examples == 2009
    assert preflight["mvtec_ad"].development_train_examples == 905
    assert preflight["mvtec_ad"].development_validation_examples == 103
    assert preflight["ecf"].development_train_examples == 1809
    assert preflight["ecf"].development_validation_examples == 200

    result = run_training_stage("mvtec_ad", "acceptance", config)
    losses = result.losses
    assert len(losses) >= 20
    assert all(math.isfinite(loss) for loss in losses)
    first_five = statistics.fmean(losses[:5])
    final_five = statistics.fmean(losses[-5:])
    assert final_five < first_five
    assert result.checkpoint_path is not None
    assert (REPO_ROOT / result.checkpoint_path / "adapter_model.safetensors").is_file()
    assert result.generated_report is not None
    validate_against_schema(
        result.generated_report,
        REPO_ROOT / config["data"]["report_schema_path"],
    )
    print(
        "C3_PHASE_3_ACCEPTANCE_GATE_PASS "
        f"first_5_mean={first_five:.8f} final_5_mean={final_five:.8f} "
        f"updates_per_second={result.optimizer_updates_per_second:.6f} "
        f"projected_full_hours={result.projected_full_run_hours:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
