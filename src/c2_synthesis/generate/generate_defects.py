"""Mask-aligned C2 inference with latent and exact pixel compositing."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import mlflow
import numpy as np
import torch
import yaml
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
from torch import Tensor
from torch.nn import functional as F

from src.c2_synthesis.data.mask_bank import real_mask_transplant
from src.c2_synthesis.data.pair_builder import build_pairs, load_dataset_config
from src.c2_synthesis.data.patch_extractor import composite_crop_back, extract_defect_crop
from src.c2_synthesis.generate.low_fidelity_selection import LFSBatchResult, filter_batch
from src.c2_synthesis.train.token_manager import LearnedTokenManager
from src.c2_synthesis.train.train_lora_defect import (
    DEFAULT_CONFIG_PATH as DEFAULT_TRAIN_CONFIG_PATH,
    inject_cross_attention_lora,
    load_lora_checkpoint,
    load_training_config,
)
from src.c2_synthesis.utils.image_io import load_image_rgb, load_mask_binary
from src.c2_synthesis.utils.logging_utils import get_logger
from src.c2_synthesis.utils.mlflow_utils import start_c2_run
from src.c2_synthesis.utils.seed import set_global_seed

logger = get_logger(__name__)

_C2_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_CONFIG_PATH = _C2_ROOT / "configs" / "sd15_inpaint_base.yaml"
_OUTPUT_ROOT = _REPO_ROOT / "outputs" / "synthetic"
_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
_META_KEYS = {
    "uid",
    "dataset",
    "class",
    "source_clean_image",
    "source_mask",
    "generation_mode",
    "crop_offset",
    "crop_bbox",
    "lora_checkpoint",
    "token",
    "seed",
    "num_inference_steps",
    "guidance_scale",
    "lpips_score",
    "lfs_passed",
}


@dataclass(frozen=True)
class GenerationCandidate:
    """One in-memory candidate before the per-class LFS decision."""

    uid: str
    generated_image: Image.Image
    clean_image: Image.Image
    mask: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GenerationResult:
    """Candidates, filtering decisions, persisted triples, and measurements."""

    candidates: tuple[GenerationCandidate, ...]
    lfs: LFSBatchResult
    persisted_uids: tuple[str, ...]
    output_dir: Path
    elapsed_seconds: float
    peak_vram_gib: float


class LFSAcceptanceRateError(RuntimeError):
    """Raised when the diagnostic acceptance-rate gate signals bad calibration."""


def _repo_path(path_value: str | Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    return path if path.is_absolute() else _REPO_ROOT / path


def _relative_posix(path_value: str | Path) -> str:
    path = _repo_path(path_value).resolve()
    try:
        return path.relative_to(_REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_base_config(path: str | Path = DEFAULT_BASE_CONFIG_PATH) -> dict[str, Any]:
    """Load the Phase 0 inference configuration without changing it."""

    with Path(path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    required = {
        "base_model_id",
        "resolution",
        "scheduler",
        "num_inference_steps",
        "guidance_scale",
        "enable_dynamic_cfg",
        "guidance_schedule",
        "mask_dilation_px",
        "dtype",
        "vae_decode_dtype",
        "smoke_test",
    }
    if not isinstance(config, dict) or not required.issubset(config):
        missing = required.difference(config if isinstance(config, dict) else {})
        raise ValueError(f"Base inference config is missing keys: {sorted(missing)}")
    if str(config["scheduler"]).upper() != "DDIM":
        raise ValueError("Phase 3 latent compositing requires the configured DDIM scheduler")
    if str(config["dtype"]) != "bfloat16":
        raise ValueError("Phase 3 UNet inference requires native bfloat16")
    if str(config["vae_decode_dtype"]) not in _DTYPES:
        raise ValueError(f"Unsupported VAE decode dtype: {config['vae_decode_dtype']}")
    if not isinstance(config["enable_dynamic_cfg"], bool):
        raise ValueError("enable_dynamic_cfg must be a boolean")
    dilation = config["mask_dilation_px"]
    if isinstance(dilation, bool) or not isinstance(dilation, int) or dilation < 0:
        raise ValueError("mask_dilation_px must be a non-negative integer")
    schedule = config["guidance_schedule"]
    if not isinstance(schedule, dict) or set(schedule) != {"start", "end", "mode"}:
        raise ValueError("guidance_schedule requires exactly start, end, and mode")
    if str(schedule["mode"]).lower() != "linear":
        raise ValueError("Only linear dynamic CFG is supported")
    if float(schedule["start"]) < 0.0 or float(schedule["end"]) < 0.0:
        raise ValueError("Dynamic CFG endpoints must be non-negative")
    return config


def guidance_scale_for_step(
    base_config: Mapping[str, Any],
    step_index: int,
    step_count: int,
) -> float:
    """Return fixed CFG or the NPI paper's increasing linear schedule.

    The first denoising step (``t=T``) uses ``start`` and the last (``t=0``)
    uses ``end``. Dynamic CFG follows arXiv:2604.22850: low early guidance
    establishes structure before stronger late guidance refines detail.
    """

    if step_count <= 0 or not 0 <= step_index < step_count:
        raise ValueError("CFG step index/count are invalid")
    if not bool(base_config["enable_dynamic_cfg"]):
        return float(base_config["guidance_scale"])
    schedule = base_config["guidance_schedule"]
    start = float(schedule["start"])
    end = float(schedule["end"])
    progress = 1.0 if step_count == 1 else step_index / (step_count - 1)
    return start + (end - start) * progress


def dilate_binary_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Expand a binary conditioning mask without changing its stored label.

    This inference-only canvas expansion is motivated by mask-grounded spatial
    control in MAGIC (arXiv:2507.02314) and GroundingAnomaly
    (arXiv:2604.08301). The original, non-dilated mask remains authoritative
    for the output triple and final pixel composite.
    """

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError("Generation mask must be two-dimensional")
    if isinstance(radius_px, bool) or not isinstance(radius_px, int) or radius_px < 0:
        raise ValueError("Mask dilation radius must be a non-negative integer")
    if radius_px == 0:
        return binary.copy()
    tensor = torch.from_numpy(binary.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(
        tensor,
        kernel_size=2 * radius_px + 1,
        stride=1,
        padding=radius_px,
    )
    return dilated.squeeze(0).squeeze(0).gt(0).to(torch.uint8).numpy()


def _pil_to_tensor(image: Image.Image, device: torch.device, dtype: torch.dtype) -> Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .div(127.5)
        .sub(1.0)
        .to(device=device, dtype=dtype)
    )


def _mask_to_tensor(mask: np.ndarray, device: torch.device, dtype: torch.dtype) -> Tensor:
    binary = np.asarray(mask) > 0
    if binary.ndim != 2:
        raise ValueError("Generation mask must have shape HxW")
    return torch.from_numpy(binary.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(
        device=device,
        dtype=dtype,
    )


def _encode_latents(vae: torch.nn.Module, image_tensor: Tensor) -> Tensor:
    return vae.encode(image_tensor).latent_dist.mode() * float(vae.config.scaling_factor)


def _latent_mask(mask_tensor: Tensor, latent_size: tuple[int, int]) -> Tensor:
    return F.adaptive_max_pool2d(mask_tensor.float(), latent_size).clamp(0.0, 1.0)


def _decoded_is_valid(decoded: Tensor, black_epsilon: float) -> bool:
    if not bool(torch.isfinite(decoded).all().item()):
        return False
    normalised = (decoded.float() / 2.0 + 0.5).clamp(0.0, 1.0)
    return float(normalised.mean().item()) > black_epsilon


def _decode_latents_safely(
    pipe: StableDiffusionInpaintPipeline,
    latents: Tensor,
    configured_dtype: torch.dtype,
    black_epsilon: float,
) -> Image.Image:
    """Decode with the Phase 0 fp32 safeguard on NaN or black frames."""

    attempted: list[torch.dtype] = [configured_dtype]
    if configured_dtype != torch.float32:
        attempted.append(torch.float32)
    last_error: Exception | None = None
    for decode_dtype in attempted:
        try:
            pipe.vae.to(device=latents.device, dtype=decode_dtype)
            decoded = pipe.vae.decode(
                latents.to(dtype=decode_dtype) / float(pipe.vae.config.scaling_factor),
                return_dict=False,
            )[0]
            if not _decoded_is_valid(decoded, black_epsilon):
                raise RuntimeError(f"VAE produced a NaN or black frame in {decode_dtype}")
            return pipe.image_processor.postprocess(decoded, output_type="pil")[0].convert("RGB")
        except (RuntimeError, ValueError) as error:
            last_error = error
            logger.warning("VAE decode failed in %s; applying safeguard", decode_dtype)
    raise RuntimeError("VAE decode safeguard exhausted all configured dtypes") from last_error


def _pixel_composite(
    generated: Image.Image,
    clean: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    """Implement System Spec section 4.6 with exact uint8 selection."""

    generated_pixels = np.asarray(generated.convert("RGB"), dtype=np.uint8)
    clean_pixels = np.asarray(clean.convert("RGB"), dtype=np.uint8)
    binary_mask = np.asarray(mask) > 0
    if generated_pixels.shape != clean_pixels.shape:
        raise ValueError("Generated and clean images must have identical shapes")
    if binary_mask.shape != clean_pixels.shape[:2]:
        raise ValueError("Pixel-composite mask must match the image dimensions")
    final_pixels = np.where(binary_mask[..., None], generated_pixels, clean_pixels)
    return Image.fromarray(final_pixels.astype(np.uint8), mode="RGB")


def assert_pixel_composite_invariant(
    generated: Image.Image,
    clean: Image.Image,
    mask: np.ndarray,
) -> None:
    """Assert every non-defect output byte equals its clean-source byte."""

    generated_pixels = np.asarray(generated.convert("RGB"), dtype=np.uint8)
    clean_pixels = np.asarray(clean.convert("RGB"), dtype=np.uint8)
    binary_mask = np.asarray(mask) > 0
    if generated_pixels.shape != clean_pixels.shape or binary_mask.shape != clean_pixels.shape[:2]:
        raise AssertionError("Pixel-composite invariant inputs have incompatible shapes")
    if not np.array_equal(generated_pixels[~binary_mask], clean_pixels[~binary_mask]):
        raise AssertionError("Pixel composite changed one or more (1-mask) source bytes")


def _same_location(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("Cannot locate an empty real mask")
    return int(xs.min()), int(ys.min())


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "class"


class DefectGenerator:
    """A loaded class-specific SD 1.5 inpainting adapter."""

    def __init__(
        self,
        base_config: Mapping[str, Any],
        train_config: Mapping[str, Any],
        checkpoint_dir: Path,
    ) -> None:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Phase 3 requires CUDA with native bfloat16 support")
        self.base_config = dict(base_config)
        self.train_config = dict(train_config)
        self.device = torch.device("cuda")
        self.unet_dtype = _DTYPES[str(base_config["dtype"])]
        self.decode_dtype = _DTYPES[str(base_config["vae_decode_dtype"])]
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            str(base_config["base_model_id"]),
            torch_dtype=self.unet_dtype,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.unet.requires_grad_(False)
        inject_cross_attention_lora(
            self.pipe.unet,
            rank=int(train_config["rank"]),
            alpha=int(train_config["lora_alpha"]),
            target_modules=list(train_config["target_modules"]),
        )
        load_lora_checkpoint(self.pipe.unet, checkpoint_dir / "lora.safetensors")
        self.token_manager = LearnedTokenManager.load(
            self.pipe.tokenizer,
            self.pipe.text_encoder,
            checkpoint_dir / "token.pt",
        )
        self.token_manager.close()
        self.pipe.text_encoder.requires_grad_(False).eval().to(
            device=self.device,
            dtype=self.unet_dtype,
        )
        self.pipe.unet.eval().to(device=self.device, dtype=self.unet_dtype)
        self.pipe.vae.requires_grad_(False).eval().to(device=self.device, dtype=self.decode_dtype)
        prompt_template = str(train_config["defect_prompt_template"])
        self.prompt = prompt_template.format(token=self.token_manager.token)

    def generate(
        self,
        clean_image: Image.Image,
        mask: np.ndarray,
        *,
        seed: int,
    ) -> Image.Image:
        """Inpaint one candidate with System Spec section 4.5 compositing."""

        if clean_image.size != (mask.shape[1], mask.shape[0]):
            raise ValueError("Generation image and mask dimensions differ")
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        original_mask = (np.asarray(mask) > 0).astype(np.uint8)
        conditioning_mask = dilate_binary_mask(
            original_mask,
            int(self.base_config["mask_dilation_px"]),
        )
        clean_tensor = _pil_to_tensor(clean_image, self.device, self.decode_dtype)
        mask_tensor = _mask_to_tensor(
            conditioning_mask,
            self.device,
            self.decode_dtype,
        )
        with torch.inference_mode():
            clean_latents = _encode_latents(self.pipe.vae, clean_tensor).to(self.unet_dtype)
            masked_latents = _encode_latents(
                self.pipe.vae,
                clean_tensor * (1.0 - mask_tensor),
            ).to(self.unet_dtype)
            mask_latents = _latent_mask(mask_tensor, tuple(clean_latents.shape[-2:])).to(
                self.unet_dtype
            )
            noise = torch.randn(
                clean_latents.shape,
                generator=generator,
                device=self.device,
                dtype=self.unet_dtype,
            )
            self.pipe.scheduler.set_timesteps(
                int(self.base_config["num_inference_steps"]),
                device=self.device,
            )
            timesteps = self.pipe.scheduler.timesteps
            latents = self.pipe.scheduler.add_noise(clean_latents, noise, timesteps[:1])
            if bool(self.base_config["enable_dynamic_cfg"]):
                schedule = self.base_config["guidance_schedule"]
                do_guidance = max(
                    float(schedule["start"]),
                    float(schedule["end"]),
                ) > 1.0
            else:
                do_guidance = float(self.base_config["guidance_scale"]) > 1.0
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                self.prompt,
                self.device,
                1,
                do_guidance,
                negative_prompt="",
            )
            if do_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            for step_index, timestep in enumerate(timesteps):
                latent_model_input = (
                    torch.cat([latents, latents], dim=0) if do_guidance else latents
                )
                latent_model_input = self.pipe.scheduler.scale_model_input(
                    latent_model_input,
                    timestep,
                )
                input_mask = (
                    torch.cat([mask_latents, mask_latents], dim=0)
                    if do_guidance
                    else mask_latents
                )
                input_masked_latents = (
                    torch.cat([masked_latents, masked_latents], dim=0)
                    if do_guidance
                    else masked_latents
                )
                model_input = torch.cat(
                    [latent_model_input, input_mask, input_masked_latents],
                    dim=1,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    noise_prediction = self.pipe.unet(
                        model_input,
                        timestep,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]
                if do_guidance:
                    prediction_unconditional, prediction_text = noise_prediction.chunk(2)
                    guidance_scale = guidance_scale_for_step(
                        self.base_config,
                        step_index,
                        len(timesteps),
                    )
                    noise_prediction = prediction_unconditional + guidance_scale * (
                        prediction_text - prediction_unconditional
                    )
                latents = self.pipe.scheduler.step(
                    noise_prediction,
                    timestep,
                    latents,
                    generator=generator,
                    return_dict=False,
                )[0]

                # Section 4.5: after every denoising step, reassert the known
                # clean latent at the next noise level. At the final step the
                # known region is the unnoised clean latent itself.
                if step_index + 1 < len(timesteps):
                    next_timestep = timesteps[step_index + 1].reshape(1)
                    known_latents = self.pipe.scheduler.add_noise(
                        clean_latents,
                        noise,
                        next_timestep,
                    )
                else:
                    known_latents = clean_latents
                latents = mask_latents * latents + (1.0 - mask_latents) * known_latents

            decoded = _decode_latents_safely(
                self.pipe,
                latents,
                self.decode_dtype,
                float(self.base_config["smoke_test"]["black_mean_epsilon"]),
            )
        # Dilation affects only the denoising canvas. The hard composite uses
        # the original label, preserving the Phase 3 bit-identical invariant.
        return _pixel_composite(decoded, clean_image, original_mask)


def _whole_candidate_inputs(
    clean_path: str,
    mask_path: str,
    resolution: int,
) -> tuple[Image.Image, np.ndarray, tuple[int, int], tuple[int, int, int, int]]:
    clean = load_image_rgb(_repo_path(clean_path)).resize(
        (resolution, resolution),
        resample=Image.Resampling.BICUBIC,
    )
    source_mask = load_mask_binary(_repo_path(mask_path))
    resized_mask = np.asarray(
        Image.fromarray(source_mask * 255).resize(
            (resolution, resolution),
            resample=Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    )
    resized_mask = (resized_mask > 127).astype(np.uint8)
    transplanted = real_mask_transplant(clean, resized_mask, _same_location(resized_mask))
    return clean, transplanted, (0, 0), (0, 0, resolution, resolution)


def _patch_candidate_inputs(
    clean_path: str,
    defect_image_path: str,
    mask_path: str,
    crop_size: int,
) -> tuple[
    Image.Image,
    Image.Image,
    np.ndarray,
    tuple[int, int],
    tuple[int, int, int, int],
]:
    full_clean = load_image_rgb(_repo_path(clean_path))
    defect_image = load_image_rgb(_repo_path(defect_image_path))
    source_mask = load_mask_binary(_repo_path(mask_path))
    crop = extract_defect_crop(defect_image, source_mask, size=crop_size)
    if crop is None:
        raise ValueError(f"Configured defect pair is not crop-eligible: {mask_path}")
    source_x, source_y, crop_width, crop_height = crop.crop_bbox
    if crop_width > full_clean.width or crop_height > full_clean.height:
        raise ValueError(
            "Configured crop dimensions do not fit the selected clean frame: "
            f"crop={(crop_width, crop_height)}, clean={full_clean.size}"
        )

    # ECF clean frames can differ in size from the real-defect source. Preserve
    # the source crop's normalised object location, then clamp it in bounds on
    # the selected clean frame instead of reusing invalid absolute coordinates.
    normalised_centre_x = (source_x + crop_width / 2.0) / defect_image.width
    normalised_centre_y = (source_y + crop_height / 2.0) / defect_image.height
    target_centre_x = round(normalised_centre_x * full_clean.width)
    target_centre_y = round(normalised_centre_y * full_clean.height)
    crop_x = max(0, min(target_centre_x - crop_width // 2, full_clean.width - crop_width))
    crop_y = max(0, min(target_centre_y - crop_height // 2, full_clean.height - crop_height))
    target_crop_bbox = (crop_x, crop_y, crop_width, crop_height)
    clean_crop = full_clean.crop(
        (crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)
    )
    if clean_crop.size != (crop_size, crop_size):
        clean_crop = clean_crop.resize(
            (crop_size, crop_size),
            resample=Image.Resampling.BICUBIC,
        )
    transplanted = real_mask_transplant(
        clean_crop,
        crop.mask,
        _same_location(crop.mask),
    )
    return full_clean, clean_crop, transplanted, (crop_x, crop_y), target_crop_bbox


def _full_patch_mask(
    full_size: tuple[int, int],
    crop_mask: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    crop_x, crop_y, crop_width, crop_height = crop_bbox
    resized = np.asarray(
        Image.fromarray((crop_mask > 0).astype(np.uint8) * 255).resize(
            (crop_width, crop_height),
            resample=Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    )
    full_mask = np.zeros((full_size[1], full_size[0]), dtype=np.uint8)
    full_mask[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width] = resized > 127
    return full_mask


def _candidate_metadata(
    *,
    uid: str,
    dataset: str,
    class_name: str,
    clean_path: str,
    mask_path: str,
    generation_mode: str,
    crop_offset: tuple[int, int],
    crop_bbox: tuple[int, int, int, int],
    checkpoint_path: Path,
    token: str,
    seed: int,
    base_config: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = {
        "uid": uid,
        "dataset": dataset,
        "class": class_name,
        "source_clean_image": Path(clean_path.replace("\\", "/")).as_posix(),
        "source_mask": Path(mask_path.replace("\\", "/")).as_posix(),
        "generation_mode": generation_mode,
        "crop_offset": [int(value) for value in crop_offset],
        "crop_bbox": [int(value) for value in crop_bbox],
        "lora_checkpoint": _relative_posix(checkpoint_path),
        "token": token,
        "seed": int(seed),
        "num_inference_steps": int(base_config["num_inference_steps"]),
        "guidance_scale": float(base_config["guidance_scale"]),
        "lpips_score": 0.0,
        "lfs_passed": False,
    }
    if set(metadata) != _META_KEYS:
        raise RuntimeError("Internal metadata schema mismatch")
    return metadata


def _temporary_path(final_path: Path) -> Path:
    return final_path.with_name(f".{final_path.stem}.{os.getpid()}.tmp{final_path.suffix}")


def _write_triple_atomic(candidate: GenerationCandidate, output_dir: Path) -> None:
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    meta_dir = output_dir / "meta"
    for directory in (images_dir, masks_dir, meta_dir):
        directory.mkdir(parents=True, exist_ok=True)

    image_path = images_dir / f"{candidate.uid}.png"
    mask_path = masks_dir / f"{candidate.uid}.png"
    meta_path = meta_dir / f"{candidate.uid}.json"
    temporary_image = _temporary_path(image_path)
    temporary_mask = _temporary_path(mask_path)
    temporary_meta = _temporary_path(meta_path)
    temporary_paths = (temporary_image, temporary_mask, temporary_meta)
    try:
        candidate.generated_image.convert("RGB").save(
            temporary_image,
            format="PNG",
        )
        Image.fromarray((candidate.mask > 0).astype(np.uint8) * 255, mode="L").save(
            temporary_mask,
            format="PNG",
        )
        with temporary_meta.open("w", encoding="utf-8") as meta_file:
            json.dump(candidate.metadata, meta_file, indent=2, sort_keys=True)
            meta_file.write("\n")
        os.replace(temporary_image, image_path)
        os.replace(temporary_mask, mask_path)
        os.replace(temporary_meta, meta_path)
    finally:
        for temporary_path in temporary_paths:
            if temporary_path.exists():
                temporary_path.unlink()


def _remove_rejected_triple(uid: str, output_dir: Path) -> None:
    """Remove only a known candidate UID if an earlier run had accepted it."""

    for relative_path in (f"images/{uid}.png", f"masks/{uid}.png", f"meta/{uid}.json"):
        candidate_path = output_dir / relative_path
        if candidate_path.is_file():
            candidate_path.unlink()


def generate_and_filter(
    *,
    candidate_count: int,
    lfs_percentile: float,
    acceptance_rate_bounds: tuple[float, float],
    accepted_target: int | None = None,
    train_config_path: str | Path = DEFAULT_TRAIN_CONFIG_PATH,
    base_config_path: str | Path = DEFAULT_BASE_CONFIG_PATH,
    output_root: str | Path = _OUTPUT_ROOT,
    pairs_override: Sequence[tuple[str, str]] | None = None,
) -> GenerationResult:
    """Generate candidates, apply LFS, persist accepted triples, and log MLflow.

    The Phase 4 sweep may pass the exact crop/object-eligible production pairs
    used to train the adapter. Normal Phase 3 calls omit the override and keep
    the original deterministic pairing behavior.
    """

    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if accepted_target is not None and not 0 < accepted_target <= candidate_count:
        raise ValueError("accepted_target must lie between 1 and candidate_count")
    minimum_rate, maximum_rate = (float(value) for value in acceptance_rate_bounds)
    if not 0.0 <= minimum_rate <= maximum_rate <= 1.0:
        raise ValueError("acceptance_rate_bounds must lie in [0,1] in ascending order")
    base_config = load_base_config(base_config_path)
    train_config = load_training_config(train_config_path)
    dataset_key = str(train_config["dataset_key"])
    class_name = str(train_config["class_name"])
    dataset_config = load_dataset_config(dataset_key)
    mode = str(dataset_config["generation_mode"])
    if mode not in {"whole", "patch"}:
        raise ValueError(f"Unsupported generation mode: {mode}")

    checkpoint_dir = (
        _repo_path(str(train_config["checkpoint_root"])) / dataset_key / class_name
    )
    checkpoint_path = checkpoint_dir / "lora.safetensors"
    token_path = checkpoint_dir / "token.pt"
    if not checkpoint_path.is_file() or not token_path.is_file():
        raise FileNotFoundError(f"Missing Phase 2 checkpoint under {checkpoint_dir}")
    if pairs_override is None:
        pairs, clean_paths = build_pairs(
            dataset_key,
            class_name,
            int(train_config["pair_budget"]),
        )
    else:
        pairs = list(pairs_override)
        _unused_pairs, clean_paths = build_pairs(dataset_key, class_name, 1)
        expected_pairs = int(train_config["pair_budget"])
        if len(pairs) != expected_pairs:
            raise ValueError(
                f"Generation pair override contains {len(pairs)} pairs; "
                f"expected configured budget {expected_pairs}"
            )
    if not pairs or not clean_paths:
        raise ValueError(f"No Phase 3 inputs for {dataset_key}/{class_name}")

    output_dir = _repo_path(output_root) / dataset_key / class_name
    set_global_seed(int(train_config["seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()
    run_params = {
        "base_model": base_config["base_model_id"],
        "dataset": dataset_key,
        "class": class_name,
        "candidate_count": candidate_count,
        "accepted_target": "all" if accepted_target is None else accepted_target,
        "lfs_percentile": float(lfs_percentile),
        "num_inference_steps": base_config["num_inference_steps"],
        "guidance_scale": base_config["guidance_scale"],
        "seed_start": train_config["seed"],
        "generation_mode": mode,
        "enable_dynamic_cfg": base_config["enable_dynamic_cfg"],
        "guidance_schedule": base_config["guidance_schedule"],
        "mask_dilation_px": base_config["mask_dilation_px"],
        "enable_masked_ti": train_config["enable_masked_ti"],
        "lora_checkpoint": checkpoint_path,
        "token_checkpoint": token_path,
    }
    candidates: list[GenerationCandidate] = []
    persisted_uids: list[str] = []

    with start_c2_run(
        f"c2-generate-{dataset_key}",
        f"{class_name}-{candidate_count}candidates",
        run_params,
    ):
        generator = DefectGenerator(base_config, train_config, checkpoint_dir)
        for index in range(candidate_count):
            defect_image_path, mask_path = pairs[index % len(pairs)]
            clean_path = clean_paths[index % len(clean_paths)]
            sample_seed = int(train_config["seed"]) + index
            uid = f"{_slug(class_name)}-{index:04d}-s{sample_seed:08d}"

            if mode == "whole":
                clean_image, mask, crop_offset, crop_bbox = _whole_candidate_inputs(
                    clean_path,
                    mask_path,
                    int(base_config["resolution"]),
                )
                generated_image = generator.generate(clean_image, mask, seed=sample_seed)
                final_clean = clean_image
                final_mask = mask
            else:
                crop_size = int(dataset_config["crop_size"])
                (
                    full_clean,
                    clean_crop,
                    crop_mask,
                    crop_offset,
                    crop_bbox,
                ) = _patch_candidate_inputs(
                    clean_path,
                    defect_image_path,
                    mask_path,
                    crop_size,
                )
                generated_crop = generator.generate(clean_crop, crop_mask, seed=sample_seed)
                generated_image = composite_crop_back(
                    full_clean,
                    generated_crop,
                    crop_offset,
                    crop_mask,
                    crop_bbox,
                )
                final_clean = full_clean
                final_mask = _full_patch_mask(full_clean.size, crop_mask, crop_bbox)

            assert_pixel_composite_invariant(generated_image, final_clean, final_mask)
            metadata = _candidate_metadata(
                uid=uid,
                dataset=dataset_key,
                class_name=class_name,
                clean_path=clean_path,
                mask_path=mask_path,
                generation_mode=mode,
                crop_offset=crop_offset,
                crop_bbox=crop_bbox,
                checkpoint_path=checkpoint_path,
                token=str(train_config["learned_token"]),
                seed=sample_seed,
                base_config=base_config,
            )
            candidates.append(
                GenerationCandidate(
                    uid=uid,
                    generated_image=generated_image,
                    clean_image=final_clean,
                    mask=final_mask,
                    metadata=metadata,
                )
            )
            logger.info("Generated candidate %d/%d: %s", index + 1, candidate_count, uid)

        lfs_result = filter_batch(
            candidates,
            percentile=float(lfs_percentile),
            device="cuda",
        )
        accepted_count = len(lfs_result.accepted)
        rejected_count = len(lfs_result.rejected)
        elapsed_before_write = time.perf_counter() - started_at
        peak_vram_gib = torch.cuda.max_memory_allocated() / (1024**3)
        mlflow.log_metrics(
            {
                "lfs_acceptance_rate": lfs_result.acceptance_rate,
                "lfs_threshold": lfs_result.threshold,
                "lfs_score_mean": float(
                    np.mean([decision.score for decision in lfs_result.decisions])
                ),
                "candidates": float(candidate_count),
                "accepted": float(accepted_count),
                "rejected": float(rejected_count),
                "wall_time_seconds": elapsed_before_write,
                "peak_vram_gib": peak_vram_gib,
            }
        )
        print(
            "LFS acceptance rate: "
            f"{lfs_result.acceptance_rate * 100.0:.2f}% "
            f"({accepted_count}/{candidate_count}); "
            f"percentile={lfs_result.percentile:.2f}, "
            f"threshold={lfs_result.threshold:.6f}"
        )
        if (
            lfs_result.acceptance_rate < minimum_rate
            or lfs_result.acceptance_rate > maximum_rate
        ):
            message = (
                f"Phase 3 stopped: LFS acceptance rate "
                f"{lfs_result.acceptance_rate * 100.0:.2f}% is outside the approved "
                f"{minimum_rate * 100.0:.0f}% to {maximum_rate * 100.0:.0f}% range."
            )
            logger.error(message)
            raise LFSAcceptanceRateError(message)
        if accepted_target is not None and accepted_count < accepted_target:
            raise RuntimeError(
                f"LFS accepted only {accepted_count} candidates, below requested "
                f"persisted target {accepted_target}"
            )

        decision_by_uid = {
            decision.sample.uid: decision for decision in lfs_result.decisions
        }
        accepted_candidates: list[GenerationCandidate] = []
        for candidate in candidates:
            decision = decision_by_uid[candidate.uid]
            within_target = accepted_target is None or len(accepted_candidates) < accepted_target
            if decision.accepted and within_target:
                accepted_metadata = dict(candidate.metadata)
                accepted_metadata["lpips_score"] = float(decision.score)
                accepted_metadata["lfs_passed"] = True
                accepted_candidate = replace(candidate, metadata=accepted_metadata)
                _write_triple_atomic(accepted_candidate, output_dir)
                accepted_candidates.append(accepted_candidate)
                persisted_uids.append(candidate.uid)
            else:
                _remove_rejected_triple(candidate.uid, output_dir)

        elapsed_seconds = time.perf_counter() - started_at
        logger.info(
            "Persisted %d LFS-accepted triples to %s in %.2f seconds",
            len(accepted_candidates),
            output_dir.as_posix(),
            elapsed_seconds,
        )

    return GenerationResult(
        candidates=tuple(candidates),
        lfs=lfs_result,
        persisted_uids=tuple(persisted_uids),
        output_dir=output_dir,
        elapsed_seconds=elapsed_seconds,
        peak_vram_gib=peak_vram_gib,
    )


def save_samples_grid(
    result: GenerationResult,
    path: str | Path,
    *,
    columns: int,
    thumbnail_size: int,
    zoom_to_mask: bool = False,
    zoom_context_scale: float = 2.0,
    minimum_zoom_size: int = 64,
    mask_overlay: bool = True,
) -> Path:
    """Save candidates with optional mask-centred zoom and overlay."""

    if columns <= 0 or thumbnail_size <= 0:
        raise ValueError("Grid columns and thumbnail_size must be positive")
    if zoom_context_scale < 1.0 or minimum_zoom_size <= 0:
        raise ValueError("Grid zoom settings must be positive and contain the mask")
    rows = math.ceil(len(result.candidates) / columns)
    label_height = max(18, thumbnail_size // 9)
    canvas = Image.new(
        "RGB",
        (columns * thumbnail_size, rows * (thumbnail_size + label_height)),
        color=(24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    decisions = {decision.sample.uid: decision for decision in result.lfs.decisions}
    for index, candidate in enumerate(result.candidates):
        decision = decisions[candidate.uid]
        image = candidate.generated_image.copy().convert("RGB")
        mask = candidate.mask > 0
        pixels = np.asarray(image, dtype=np.uint8).copy()
        if mask_overlay:
            pixels[mask] = (
                pixels[mask].astype(np.uint16) * 3
                + np.array([255, 0, 0], dtype=np.uint16)
            ) // 4
        if zoom_to_mask:
            mask_y, mask_x = np.nonzero(mask)
            if mask_x.size == 0:
                raise ValueError(f"Grid candidate has an empty mask: {candidate.uid}")
            bbox_width = int(mask_x.max()) - int(mask_x.min()) + 1
            bbox_height = int(mask_y.max()) - int(mask_y.min()) + 1
            crop_size = max(
                minimum_zoom_size,
                math.ceil(max(bbox_width, bbox_height) * zoom_context_scale),
            )
            crop_size = min(crop_size, pixels.shape[1], pixels.shape[0])
            centre_x = (int(mask_x.min()) + int(mask_x.max())) // 2
            centre_y = (int(mask_y.min()) + int(mask_y.max())) // 2
            crop_x = max(0, min(centre_x - crop_size // 2, pixels.shape[1] - crop_size))
            crop_y = max(0, min(centre_y - crop_size // 2, pixels.shape[0] - crop_size))
            pixels = pixels[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size]
        tile = Image.fromarray(pixels.astype(np.uint8)).resize(
            (thumbnail_size, thumbnail_size),
            resample=Image.Resampling.BICUBIC,
        )
        x = (index % columns) * thumbnail_size
        y = (index // columns) * (thumbnail_size + label_height)
        canvas.paste(tile, (x, y))
        border = (40, 210, 80) if decision.accepted else (230, 65, 65)
        draw.rectangle(
            (x, y, x + thumbnail_size - 1, y + thumbnail_size - 1),
            outline=border,
            width=3,
        )
        label = f"{index + 1:02d} {'PASS' if decision.accepted else 'REJECT'} {decision.score:.3f}"
        draw.text((x + 4, y + thumbnail_size + 2), label, fill=border)

    output_path = _repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", dpi=(300, 300))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-count", type=int, required=True)
    parser.add_argument("--lfs-percentile", type=float, required=True)
    parser.add_argument("--min-acceptance-rate", type=float, required=True)
    parser.add_argument("--max-acceptance-rate", type=float, required=True)
    parser.add_argument("--accepted-target", type=int)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG_PATH)
    args = parser.parse_args()
    result = generate_and_filter(
        candidate_count=args.candidate_count,
        lfs_percentile=args.lfs_percentile,
        acceptance_rate_bounds=(
            args.min_acceptance_rate,
            args.max_acceptance_rate,
        ),
        accepted_target=args.accepted_target,
        train_config_path=args.train_config,
        base_config_path=args.base_config,
    )
    print(f"Accepted triples: {len(result.persisted_uids)} under {result.output_dir}")


if __name__ == "__main__":
    main()
