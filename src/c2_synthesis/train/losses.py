"""Loss terms for C2 LoRA and learned-token fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class C2Losses:
    """The differentiable total objective and its three reported terms."""

    total: Tensor
    defect: Tensor
    object: Tensor
    attention: Tensor


def _validate_noise_tensors(noise: Tensor, noise_pred: Tensor) -> None:
    if noise.shape != noise_pred.shape:
        raise ValueError(
            f"noise and noise_pred must have identical shapes, got "
            f"{tuple(noise.shape)} and {tuple(noise_pred.shape)}"
        )
    if noise.ndim != 4:
        raise ValueError("noise tensors must have shape [batch, channels, height, width]")


def _validate_mask(mask_latent: Tensor, reference: Tensor) -> Tensor:
    if mask_latent.ndim != 4 or mask_latent.shape[1] != 1:
        raise ValueError("mask_latent must have shape [batch, 1, height, width]")
    if mask_latent.shape[0] != reference.shape[0] or mask_latent.shape[-2:] != reference.shape[-2:]:
        raise ValueError("mask_latent batch and spatial dimensions must match the noise")
    return mask_latent.float().clamp(0.0, 1.0)


def defect_loss(noise: Tensor, noise_pred: Tensor, mask_latent: Tensor) -> Tensor:
    r"""Implement System Spec section 4.1.

    .. math:: L_{def}=E[\|M\odot(\epsilon-\epsilon_\theta)\|_2^2]

    The reduction averages over masked pixels only (and over latent channels),
    rather than over the full latent plane. This explicit normalisation keeps
    the gradient magnitude useful for ECF micro-defects with very small masks.
    """

    _validate_noise_tensors(noise, noise_pred)
    mask = _validate_mask(mask_latent, noise)
    squared_error = (noise.float() - noise_pred.float()).square()
    denominator = mask.sum() * noise.shape[1]
    if denominator.item() <= 0:
        raise ValueError("defect_loss requires at least one masked latent pixel")
    return (squared_error * mask).sum() / denominator


def object_loss(
    noise: Tensor,
    noise_pred: Tensor,
    mask_latent: Tensor,
    alpha: float,
) -> Tensor:
    r"""Implement System Spec section 4.2.

    .. math:: L_{obj}=E[\|M'\odot(\epsilon-\epsilon_\theta)\|_2^2],
              \quad M'=M+\alpha(1-M)
    """

    _validate_noise_tensors(noise, noise_pred)
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    mask = _validate_mask(mask_latent, noise)
    weighted_mask = mask + float(alpha) * (1.0 - mask)
    weighted_error = weighted_mask * (noise.float() - noise_pred.float())
    return weighted_error.square().mean()


def attention_loss(attn_map_token: Tensor, mask_latent: Tensor) -> Tensor:
    r"""Implement System Spec section 4.3.

    .. math:: L_{attn}=E[\|A_{token}-M\|_2^2]
    """

    if attn_map_token.ndim == 3:
        attn_map_token = attn_map_token.unsqueeze(1)
    if attn_map_token.ndim != 4 or attn_map_token.shape[1] != 1:
        raise ValueError("attn_map_token must have shape [batch, 1, height, width]")
    mask = _validate_mask(mask_latent, attn_map_token)
    return F.mse_loss(attn_map_token.float(), mask)


def total_c2_loss(
    noise: Tensor,
    defect_noise_pred: Tensor,
    object_noise_pred: Tensor,
    defect_mask_latent: Tensor,
    object_mask_latent: Tensor,
    attn_map_token: Tensor,
    *,
    alpha: float,
    lambda_obj: float,
    lambda_attn: float,
) -> C2Losses:
    r"""Implement System Spec section 4.4.

    .. math:: L_{C2}=L_{def}+\lambda_{obj}L_{obj}+\lambda_{attn}L_{attn}
    """

    if float(lambda_obj) < 0.0 or float(lambda_attn) < 0.0:
        raise ValueError("loss weights must be non-negative")
    loss_defect = defect_loss(noise, defect_noise_pred, defect_mask_latent)
    loss_object = object_loss(noise, object_noise_pred, object_mask_latent, alpha)
    loss_attention = attention_loss(attn_map_token, defect_mask_latent)
    loss_total = (
        loss_defect
        + float(lambda_obj) * loss_object
        + float(lambda_attn) * loss_attention
    )
    return C2Losses(
        total=loss_total,
        defect=loss_defect,
        object=loss_object,
        attention=loss_attention,
    )
