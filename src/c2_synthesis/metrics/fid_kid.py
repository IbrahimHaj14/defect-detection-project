"""Clean-FID Inception features for per-class FID and KID."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from cleanfid import fid


@dataclass(frozen=True)
class FidelityMetrics:
    """FID plus the small-sample-primary KID metric."""

    fid: float
    kid: float
    n_real: int
    n_synthetic: int


_FEATURE_MODELS: dict[str, torch.nn.Module] = {}


def _feature_model(device: torch.device) -> torch.nn.Module:
    key = str(device)
    if key not in _FEATURE_MODELS:
        _FEATURE_MODELS[key] = fid.build_feature_extractor(
            mode="clean",
            device=device,
            use_dataparallel=False,
        )
    return _FEATURE_MODELS[key]


def _features(
    paths: Sequence[str | Path],
    *,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    files = [str(Path(path)) for path in paths]
    if len(files) < 2:
        raise ValueError("FID/KID require at least two images in each set")
    missing = [path for path in files if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Metric image does not exist: {missing[0]}")
    features = fid.get_files_features(
        files,
        model=model,
        num_workers=0,
        batch_size=int(batch_size),
        device=device,
        mode="clean",
        verbose=False,
    )
    if features.ndim != 2 or not np.isfinite(features).all():
        raise FloatingPointError("Clean-FID returned invalid Inception features")
    return features


def compute_fid_kid(
    real_paths: Sequence[str | Path],
    synthetic_paths: Sequence[str | Path],
    *,
    device: str | torch.device,
    batch_size: int,
    kid_num_subsets: int,
    kid_max_subset_size: int,
    seed: int,
) -> FidelityMetrics:
    """Compute clean FID and polynomial-kernel KID from file lists.

    Clean-FID's internal DataLoader is explicitly invoked with
    ``num_workers=0`` for the mandatory Windows protection. KID uses the
    smaller class count as its subset size and is the primary fidelity metric
    for the small real-defect sets.
    """

    if batch_size <= 0 or kid_num_subsets <= 0 or kid_max_subset_size < 2:
        raise ValueError("Metric batching and KID subset settings must be positive")
    resolved_device = torch.device(device)
    model = _feature_model(resolved_device)
    real_features = _features(
        real_paths,
        model=model,
        device=resolved_device,
        batch_size=batch_size,
    )
    synthetic_features = _features(
        synthetic_paths,
        model=model,
        device=resolved_device,
        batch_size=batch_size,
    )
    fid_value = float(fid.fid_from_feats(real_features, synthetic_features))
    random_state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        kid_value = float(
            fid.kernel_distance(
                real_features,
                synthetic_features,
                num_subsets=int(kid_num_subsets),
                max_subset_size=int(kid_max_subset_size),
            )
        )
    finally:
        np.random.set_state(random_state)
    if not np.isfinite(fid_value) or not np.isfinite(kid_value):
        raise FloatingPointError(f"Non-finite fidelity metrics: FID={fid_value}, KID={kid_value}")
    return FidelityMetrics(
        fid=fid_value,
        kid=kid_value,
        n_real=len(real_paths),
        n_synthetic=len(synthetic_paths),
    )
