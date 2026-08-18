"""Safe PIL-backed image and binary-mask I/O for C2."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TypeAlias

import numpy as np
from PIL import Image

PathInput: TypeAlias = str | PathLike[str]
ImageInput: TypeAlias = Image.Image | np.ndarray


def _normalise_path(path: PathInput) -> Path:
    """Accept manifest-style POSIX paths on every supported platform."""

    return Path(str(path).replace("\\", "/"))


def load_image_rgb(path: PathInput) -> Image.Image:
    """Load an image as a detached three-channel RGB PIL image."""

    image_path = _normalise_path(path)
    with Image.open(image_path) as image:
        return image.convert("RGB").copy()


def load_mask_binary(path: PathInput) -> np.ndarray:
    """Load a mask as a single-channel ``uint8`` array containing {0, 1}.

    Masks are binarised defensively at the permanent C2 contract threshold:
    pixels greater than 127 are defects.
    """

    mask_path = _normalise_path(path)
    with Image.open(mask_path) as mask:
        mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)
    return (mask_array > 127).astype(np.uint8)


def _array_to_rgb_image(array: np.ndarray) -> Image.Image:
    values = np.asarray(array)
    if values.ndim not in (2, 3):
        raise ValueError("image array must have shape HxW or HxWxC")

    if np.issubdtype(values.dtype, np.floating):
        finite_values = values[np.isfinite(values)]
        if finite_values.size != values.size:
            raise ValueError("image array contains NaN or infinite values")
        if finite_values.size and finite_values.max() <= 1.0:
            values = values * 255.0

    values = np.clip(values, 0, 255).astype(np.uint8)
    return Image.fromarray(values).convert("RGB")


def save_image(image: ImageInput, path: PathInput) -> str:
    """Save an RGB image and return its forward-slash POSIX path."""

    output_path = _normalise_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgb_image = image.convert("RGB") if isinstance(image, Image.Image) else _array_to_rgb_image(image)
    rgb_image.save(output_path)
    return output_path.as_posix()


def save_mask(mask: ImageInput, path: PathInput) -> str:
    """Save a single-channel binary PNG with values exactly in {0, 255}."""

    output_path = _normalise_path(path)
    if output_path.suffix.lower() != ".png":
        raise ValueError("C2 masks must be saved as PNG files")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(mask, Image.Image):
        mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)
    else:
        mask_array = np.asarray(mask)

    if mask_array.ndim != 2:
        raise ValueError("mask must be a single-channel HxW image")
    if not np.all(np.isfinite(mask_array)):
        raise ValueError("mask contains NaN or infinite values")

    threshold = 0 if mask_array.size and mask_array.max() <= 1 else 127
    binary_mask = (mask_array > threshold).astype(np.uint8) * 255
    Image.fromarray(binary_mask, mode="L").save(output_path, format="PNG")
    return output_path.as_posix()
