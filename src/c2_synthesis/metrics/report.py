"""Atomic Phase 4 fidelity-table assembly."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

FIELDNAMES = (
    "dataset",
    "class",
    "n_real",
    "n_synth",
    "crop_exclusion_count",
    "FID",
    "KID",
    "LPIPS_diversity",
    "lfs_acceptance",
)


@dataclass(frozen=True)
class FidelityRow:
    """One per-dataset/class row in ``fidelity.csv``."""

    dataset: str
    class_name: str
    n_real: int
    n_synth: int
    crop_exclusion_count: int
    fid: float
    kid: float
    lpips_diversity: float
    lfs_acceptance: float

    def as_csv_row(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "class": self.class_name,
            "n_real": self.n_real,
            "n_synth": self.n_synth,
            "crop_exclusion_count": self.crop_exclusion_count,
            "FID": self.fid,
            "KID": self.kid,
            "LPIPS_diversity": self.lpips_diversity,
            "lfs_acceptance": self.lfs_acceptance,
        }

    def validate(self) -> None:
        if not self.dataset or not self.class_name:
            raise ValueError("Fidelity dataset and class must be non-empty")
        if self.n_real < 2 or self.n_synth < 2 or self.crop_exclusion_count < 0:
            raise ValueError("Fidelity counts are invalid")
        values = (self.fid, self.kid, self.lpips_diversity, self.lfs_acceptance)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Fidelity row contains non-finite metrics: {asdict(self)}")
        if not 0.0 <= self.lfs_acceptance <= 1.0:
            raise ValueError("lfs_acceptance must lie in [0,1]")


def read_fidelity_report(path: str | Path) -> list[dict[str, str]]:
    report_path = Path(path)
    if not report_path.is_file():
        return []
    with report_path.open("r", encoding="utf-8", newline="") as report_file:
        reader = csv.DictReader(report_file)
        if tuple(reader.fieldnames or ()) != FIELDNAMES:
            raise ValueError(f"Unexpected fidelity.csv columns: {reader.fieldnames}")
        return list(reader)


def write_fidelity_report(
    rows: Iterable[FidelityRow],
    path: str | Path,
    *,
    preserve_existing: bool = True,
) -> Path:
    """Upsert rows by ``(dataset,class)`` and atomically replace the CSV."""

    output_path = Path(path)
    combined: dict[tuple[str, str], dict[str, object]] = {}
    if preserve_existing:
        for existing in read_fidelity_report(output_path):
            combined[(existing["dataset"], existing["class"])] = existing
    for row in rows:
        row.validate()
        csv_row = row.as_csv_row()
        combined[(row.dataset, row.class_name)] = csv_row
    if not combined:
        raise ValueError("Cannot write an empty fidelity report")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as report_file:
            writer = csv.DictWriter(report_file, fieldnames=FIELDNAMES)
            writer.writeheader()
            for key in sorted(combined):
                writer.writerow(combined[key])
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path
