"""Phase 0 acceptance test for the C2 Stable Diffusion inpainting stack."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Explicit imports are part of the Phase 0 acceptance gate.
from src.c2_synthesis.utils import image_io, logging_utils, mlflow_utils, seed  # noqa: E402

CONFIG_PATH = REPO_ROOT / "src/c2_synthesis/configs/sd15_inpaint_base.yaml"
DEVICE = "cuda"
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    required_keys = {
        "base_model_id",
        "resolution",
        "scheduler",
        "num_inference_steps",
        "guidance_scale",
        "dtype",
        "vae_decode_dtype",
        "smoke_test",
    }
    missing_keys = required_keys.difference(config)
    if missing_keys:
        raise AssertionError(f"Missing Phase 0 config keys: {sorted(missing_keys)}")
    return config


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    try:
        return _DTYPES[dtype_name]
    except KeyError as error:
        raise ValueError(f"Unsupported configured dtype: {dtype_name}") from error


def _make_smoke_inputs(config: dict[str, Any]) -> tuple[Image.Image, Image.Image]:
    resolution = int(config["resolution"])
    smoke_config = config["smoke_test"]
    grey_value = int(smoke_config["input_gray_value"])
    x0, y0, x1, y1 = (int(value) for value in smoke_config["mask_box"])

    if not (0 <= x0 < x1 <= resolution and 0 <= y0 < y1 <= resolution):
        raise AssertionError("smoke-test mask_box must be inside the configured resolution")

    input_image = Image.new(
        "RGB",
        (resolution, resolution),
        color=(grey_value, grey_value, grey_value),
    )
    mask_array = np.zeros((resolution, resolution), dtype=np.uint8)
    mask_array[y0:y1, x0:x1] = 255
    mask_image = Image.fromarray(mask_array, mode="L")
    return input_image, mask_image


def _decoded_tensor_is_valid(decoded: torch.Tensor, black_epsilon: float) -> bool:
    if not torch.isfinite(decoded).all().item():
        return False
    normalised = (decoded.float() / 2.0 + 0.5).clamp(0.0, 1.0)
    return normalised.mean().item() > black_epsilon


def _decode_latents_safely(
    pipe: StableDiffusionInpaintPipeline,
    latents: torch.Tensor,
    configured_dtype: torch.dtype,
    black_epsilon: float,
) -> tuple[Image.Image, torch.dtype, bool]:
    """Decode in the configured dtype, falling back to fp32 on invalid output."""

    def decode_once(decode_dtype: torch.dtype) -> tuple[torch.Tensor, Image.Image]:
        pipe.vae.to(device=DEVICE, dtype=decode_dtype)
        decode_latents = latents.to(device=DEVICE, dtype=decode_dtype)
        decoded = pipe.vae.decode(
            decode_latents / pipe.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        if not _decoded_tensor_is_valid(decoded, black_epsilon):
            raise RuntimeError(f"VAE produced NaN or black output in {decode_dtype}")
        output_image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        return decoded, output_image

    try:
        _, output = decode_once(configured_dtype)
        return output, configured_dtype, False
    except (RuntimeError, ValueError) as error:
        if configured_dtype == torch.float32:
            raise RuntimeError("fp32 VAE safeguard decode failed") from error
        _, output = decode_once(torch.float32)
        return output, torch.float32, True


def main() -> None:
    started_at = time.perf_counter()

    assert callable(seed.set_global_seed)
    assert callable(logging_utils.get_logger)
    assert callable(image_io.load_image_rgb)
    assert callable(image_io.load_mask_binary)
    assert callable(image_io.save_image)
    assert callable(image_io.save_mask)
    assert callable(mlflow_utils.start_c2_run)

    if not torch.cuda.is_available():
        raise AssertionError("Phase 0 requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise AssertionError("Phase 0 requires native CUDA bf16 support")

    config = _load_config()
    pipeline_dtype = _resolve_dtype(str(config["dtype"]))
    vae_decode_dtype = _resolve_dtype(str(config["vae_decode_dtype"]))
    if pipeline_dtype != torch.bfloat16:
        raise AssertionError("The Phase 0 pipeline dtype must be bfloat16")

    smoke_config = config["smoke_test"]
    seed.set_global_seed(int(smoke_config["seed"]))
    input_image, mask_image = _make_smoke_inputs(config)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        config["base_model_id"],
        torch_dtype=pipeline_dtype,
    )
    if str(config["scheduler"]).upper() != "DDIM":
        raise AssertionError("Phase 0 requires the DDIM scheduler")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=DEVICE).manual_seed(int(smoke_config["seed"]))
    with torch.inference_mode():
        latent_output = pipe(
            prompt=str(smoke_config["prompt"]),
            image=input_image,
            mask_image=mask_image,
            height=int(config["resolution"]),
            width=int(config["resolution"]),
            num_inference_steps=int(config["num_inference_steps"]),
            guidance_scale=float(config["guidance_scale"]),
            generator=generator,
            output_type="latent",
        ).images
        output_image, used_vae_dtype, used_fallback = _decode_latents_safely(
            pipe=pipe,
            latents=latent_output,
            configured_dtype=vae_decode_dtype,
            black_epsilon=float(smoke_config["black_mean_epsilon"]),
        )

    expected_size = (int(config["resolution"]), int(config["resolution"]))
    assert output_image.size == expected_size, (
        f"Expected output size {expected_size}, got {output_image.size}"
    )
    assert output_image.mode == "RGB", f"Expected RGB output, got {output_image.mode}"

    input_pixels = np.asarray(input_image, dtype=np.float32) / 255.0
    output_pixels = np.asarray(output_image, dtype=np.float32) / 255.0
    mask_pixels = np.asarray(mask_image, dtype=np.uint8)

    assert np.isfinite(output_pixels).all(), "Decoded output contains NaN or infinity"
    output_mean = float(output_pixels.mean())
    assert output_mean > float(smoke_config["black_mean_epsilon"]), "Decoded output is all black"

    unmasked_pixels = mask_pixels == 0
    assert unmasked_pixels.any(), "Smoke-test mask leaves no unmasked pixels"
    unmasked_mad = float(np.abs(output_pixels - input_pixels)[unmasked_pixels].mean())
    tolerance = float(smoke_config["unmasked_mad_tolerance"])
    assert unmasked_mad < tolerance, (
        f"Raw pipeline changed the unmasked region excessively: "
        f"MAD={unmasked_mad:.6f}, tolerance={tolerance:.6f}"
    )

    elapsed_seconds = time.perf_counter() - started_at
    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024**3)

    print("Phase 0 verification passed")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Pipeline dtype: {pipeline_dtype}")
    print(f"VAE decode dtype: {used_vae_dtype}")
    print(f"VAE fp32 fallback used: {used_fallback}")
    print(f"Output: {output_image.mode} {output_image.size}")
    print(f"Output mean [0,1]: {output_mean:.6f}")
    print(f"Unmasked MAD [0,1]: {unmasked_mad:.6f} (< {tolerance:.6f})")
    print(f"Peak VRAM: {peak_vram_gib:.3f} GiB")
    print(f"Elapsed time: {elapsed_seconds:.2f} s")


if __name__ == "__main__":
    main()
