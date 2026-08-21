"""LPIPS-based Low-Fidelity Selection (LFS) for generated defects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, TypeAlias

import lpips
import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from src.c2_synthesis.utils.logging_utils import get_logger

ImageInput: TypeAlias = Image.Image | np.ndarray | Tensor
MaskInput: TypeAlias = Image.Image | np.ndarray | Tensor
ScoreFunction: TypeAlias = Callable[[ImageInput, ImageInput, MaskInput], float]

logger = get_logger(__name__)


@dataclass(frozen=True)
class LFSDecision:
    """Score and keep/reject decision for one generated candidate."""

    sample: Any
    score: float
    accepted: bool


@dataclass(frozen=True)
class LFSBatchResult:
    """Per-class percentile threshold and all candidate decisions."""

    decisions: tuple[LFSDecision, ...]
    threshold: float
    percentile: float
    acceptance_rate: float

    @property
    def accepted(self) -> tuple[Any, ...]:
        return tuple(decision.sample for decision in self.decisions if decision.accepted)

    @property
    def rejected(self) -> tuple[Any, ...]:
        return tuple(decision.sample for decision in self.decisions if not decision.accepted)


_LPIPS_MODELS: dict[str, nn.Module] = {}


def _lpips_model(device: torch.device) -> nn.Module:
    key = str(device)
    if key not in _LPIPS_MODELS:
        model = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
        model.requires_grad_(False)
        _LPIPS_MODELS[key] = model
    return _LPIPS_MODELS[key]


def _image_tensor(image: ImageInput, device: torch.device) -> Tensor:
    if isinstance(image, Tensor):
        tensor = image.detach()
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 3:
            raise ValueError("Image tensor must have shape [3,H,W] or [1,3,H,W]")
        if tensor.shape[0] != 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        tensor = tensor.float()
        if tensor.numel() and tensor.max().item() > 1.0:
            tensor = tensor / 255.0
    else:
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
        else:
            array = np.asarray(image, dtype=np.float32)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("Image array must have shape HxWx3")
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        if tensor.numel() and tensor.max().item() > 1.0:
            tensor = tensor / 255.0
    if tensor.shape[0] != 3:
        raise ValueError("LPIPS images must contain exactly three channels")
    return tensor.clamp(0.0, 1.0).unsqueeze(0).to(device)


def _mask_tensor(mask: MaskInput, shape: tuple[int, int], device: torch.device) -> Tensor:
    if isinstance(mask, Image.Image):
        array = np.asarray(mask.convert("L"), dtype=np.uint8)
        tensor = torch.from_numpy(array.copy())
    elif isinstance(mask, Tensor):
        tensor = mask.detach()
    else:
        tensor = torch.from_numpy(np.asarray(mask).copy())
    while tensor.ndim > 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError("LFS mask must have shape HxW")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"LFS mask shape {tuple(tensor.shape)} does not match image {shape}")
    threshold = 0.0 if tensor.numel() and tensor.max().item() <= 1 else 127.0
    return tensor.gt(threshold).float().unsqueeze(0).unsqueeze(0).to(device)


def lfs_score(
    gen_image: ImageInput,
    clean_image: ImageInput,
    mask: MaskInput,
    *,
    model: nn.Module | None = None,
    device: str | torch.device | None = None,
) -> float:
    r"""Compute System Spec section 4.7.

    .. math:: s_{LFS}=LPIPS(m\odot\hat{x},m\odot y)

    Images are mapped to LPIPS' ``[-1, 1]`` domain before applying the binary
    mask, leaving exactly zero outside the defect support for both inputs.
    """

    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    generated = _image_tensor(gen_image, resolved_device)
    clean = _image_tensor(clean_image, resolved_device)
    if generated.shape != clean.shape:
        raise ValueError(
            f"Generated and clean image shapes differ: {generated.shape} vs {clean.shape}"
        )
    binary_mask = _mask_tensor(mask, tuple(generated.shape[-2:]), resolved_device)
    metric = model if model is not None else _lpips_model(resolved_device)
    metric = metric.to(resolved_device).eval()
    generated_masked = (generated * 2.0 - 1.0) * binary_mask
    clean_masked = (clean * 2.0 - 1.0) * binary_mask
    with torch.inference_mode():
        score = metric(generated_masked, clean_masked)
    value = float(score.detach().float().mean().item())
    if not np.isfinite(value):
        raise FloatingPointError("LFS produced a non-finite LPIPS score")
    return value


def _sample_field(sample: Any, name: str) -> Any:
    if isinstance(sample, Mapping):
        if name not in sample:
            raise KeyError(f"LFS sample mapping is missing '{name}'")
        return sample[name]
    try:
        return getattr(sample, name)
    except AttributeError as error:
        raise AttributeError(f"LFS sample is missing '{name}'") from error


def filter_batch(
    samples: Sequence[Any],
    percentile: float = 25.0,
    *,
    scorer: ScoreFunction | None = None,
    model: nn.Module | None = None,
    device: str | torch.device | None = None,
) -> LFSBatchResult:
    """Reject candidates below the per-class LPIPS percentile threshold."""

    if not samples:
        raise ValueError("filter_batch requires at least one candidate")
    if not 0.0 <= float(percentile) <= 100.0:
        raise ValueError("percentile must lie in [0, 100]")

    scores: list[float] = []
    for sample in samples:
        if scorer is None:
            score = lfs_score(
                _sample_field(sample, "generated_image"),
                _sample_field(sample, "clean_image"),
                _sample_field(sample, "mask"),
                model=model,
                device=device,
            )
        else:
            score = float(
                scorer(
                    _sample_field(sample, "generated_image"),
                    _sample_field(sample, "clean_image"),
                    _sample_field(sample, "mask"),
                )
            )
        if not np.isfinite(score):
            raise FloatingPointError("LFS scorer returned a non-finite value")
        scores.append(score)

    threshold = float(np.percentile(np.asarray(scores, dtype=np.float64), percentile))
    decisions = tuple(
        LFSDecision(sample=sample, score=score, accepted=score >= threshold)
        for sample, score in zip(samples, scores, strict=True)
    )
    acceptance_rate = sum(decision.accepted for decision in decisions) / len(decisions)
    logger.info(
        "LFS: percentile=%.2f threshold=%.6f accepted=%d/%d rate=%.2f%%",
        float(percentile),
        threshold,
        sum(decision.accepted for decision in decisions),
        len(decisions),
        acceptance_rate * 100.0,
    )
    return LFSBatchResult(
        decisions=decisions,
        threshold=threshold,
        percentile=float(percentile),
        acceptance_rate=acceptance_rate,
    )
