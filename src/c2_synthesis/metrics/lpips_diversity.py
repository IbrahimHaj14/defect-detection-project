"""Mean pairwise LPIPS diversity for one synthetic class."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Sequence

import lpips
import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

_MODELS: dict[str, nn.Module] = {}


def _model(device: torch.device) -> nn.Module:
    key = str(device)
    if key not in _MODELS:
        metric = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
        metric.requires_grad_(False)
        _MODELS[key] = metric
    return _MODELS[key]


def _load_tensor(path: str | Path, resolution: int) -> Tensor:
    with Image.open(path) as image_file:
        image = image_file.convert("RGB").resize(
            (resolution, resolution),
            resample=Image.Resampling.BICUBIC,
        )
        array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div(127.5).sub(1.0)


def mean_pairwise_lpips(
    image_paths: Sequence[str | Path],
    *,
    device: str | torch.device,
    resolution: int,
    batch_size: int,
) -> float:
    """Return the exact mean LPIPS over every unordered image pair."""

    if len(image_paths) < 2:
        raise ValueError("LPIPS diversity requires at least two synthetic images")
    if resolution <= 0 or batch_size <= 0:
        raise ValueError("LPIPS resolution and batch_size must be positive")
    missing = [str(path) for path in image_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Synthetic image does not exist: {missing[0]}")

    resolved_device = torch.device(device)
    tensors = [_load_tensor(path, resolution) for path in image_paths]
    pairs = list(combinations(range(len(tensors)), 2))
    metric = _model(resolved_device)
    total = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            left = torch.stack([tensors[i] for i, _ in batch_pairs]).to(resolved_device)
            right = torch.stack([tensors[j] for _, j in batch_pairs]).to(resolved_device)
            values = metric(left, right).detach().float().flatten()
            if not bool(torch.isfinite(values).all().item()):
                raise FloatingPointError("LPIPS diversity produced a non-finite value")
            total += float(values.sum().item())
            count += int(values.numel())
    value = total / count
    if not np.isfinite(value):
        raise FloatingPointError("Mean pairwise LPIPS is non-finite")
    return float(value)
