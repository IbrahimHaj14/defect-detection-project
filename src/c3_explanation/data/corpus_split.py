"""Deterministic, stratified splitting for the real-defect C3 corpus."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_count_for_stratum(size: int, test_fraction: float) -> int:
    """Return the approved half-up test count, with singleton protection."""

    if size < 0:
        raise ValueError("Stratum size cannot be negative")
    if size <= 1:
        return 0
    fraction = Decimal(str(test_fraction))
    if not Decimal("0") < fraction < Decimal("1"):
        raise ValueError("test_fraction must lie strictly between 0 and 1")
    unbounded = int(
        (Decimal(size) * fraction).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
    return min(max(unbounded, 1), size - 1)


def _resolved(path_value: str | Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _is_c2_synthetic_path(path_value: str | Path) -> bool:
    resolved = _resolved(path_value)
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        return False
    parts = relative.parts
    return (
        len(parts) >= 2
        and parts[0].lower() == "outputs"
        and parts[1].lower().startswith("synthetic")
    )


def assert_no_c2_synthetic_paths(samples: Sequence[Mapping[str, Any]]) -> None:
    """Reject any member whose source image or mask resolves under C2 output."""

    for sample in samples:
        for field in ("image_path", "mask_path"):
            if _is_c2_synthetic_path(str(sample[field])):
                raise AssertionError(
                    f"C2 synthetic path is forbidden in the C3 corpus: {sample[field]}"
                )


def deterministic_stratified_split(
    samples: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    train_fraction: float,
    test_fraction: float,
) -> dict[str, list[dict[str, Any]]]:
    """Split real samples by ``(class, defect_type)`` using one seeded RNG.

    Members and strata are stably sorted before shuffling. Each non-singleton
    stratum uses round-half-up for its test count, clamped to ``[1, n-1]``;
    singleton strata remain in training.
    """

    train_decimal = Decimal(str(train_fraction))
    test_decimal = Decimal(str(test_fraction))
    if train_decimal + test_decimal != Decimal("1"):
        raise ValueError("train_fraction and test_fraction must sum exactly to 1")
    assert_no_c2_synthetic_paths(samples)

    uids = [str(sample["uid"]) for sample in samples]
    if len(uids) != len(set(uids)):
        raise ValueError("Corpus UIDs must be unique before splitting")

    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        copied = dict(sample)
        strata[(str(copied["class"]), str(copied["defect_type"]))].append(copied)

    rng = random.Random(int(seed))
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for stratum_key in sorted(strata):
        members = sorted(strata[stratum_key], key=lambda item: str(item["uid"]))
        rng.shuffle(members)
        test_count = test_count_for_stratum(len(members), float(test_decimal))
        test.extend(members[:test_count])
        train.extend(members[test_count:])

    train.sort(key=lambda item: str(item["uid"]))
    test.sort(key=lambda item: str(item["uid"]))
    assert_no_c2_synthetic_paths([*train, *test])
    return {"train": train, "test": test}
