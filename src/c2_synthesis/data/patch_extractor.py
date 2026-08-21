"""Mask-centred ECF crop extraction and exact background compositing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeAlias

import numpy as np
from PIL import Image

from src.c2_synthesis.utils.image_io import load_image_rgb, load_mask_binary
from src.c2_synthesis.utils.logging_utils import get_logger

CropOffset: TypeAlias = tuple[int, int]
CropBBox: TypeAlias = tuple[int, int, int, int]
ImageInput: TypeAlias = Image.Image | np.ndarray
MaskInput: TypeAlias = Image.Image | np.ndarray
DefectPair: TypeAlias = tuple[str, str]

logger = get_logger(__name__)


@dataclass(frozen=True)
class DefectCrop:
    """A 256-space crop plus its exact placement on the full frame."""

    image: Image.Image
    mask: np.ndarray
    crop_offset: CropOffset
    crop_bbox: CropBBox
    resized_from_full_frame: bool


@dataclass(frozen=True)
class CropEligibilitySummary:
    """Per-class crop counts consumed by the Phase 4 fidelity report."""

    total_pairs: int
    usable_pairs: int
    excluded_unfittable_square: int
    excluded_empty_mask: int
    excluded_frame_too_small: int

    @property
    def crop_exclusion_count(self) -> int:
        return (
            self.excluded_unfittable_square
            + self.excluded_empty_mask
            + self.excluded_frame_too_small
        )


@dataclass(frozen=True)
class _CropGeometry:
    crop_bbox: CropBBox
    resized: bool


def _as_rgb_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("image must have shape HxW or HxWxC")
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("RGB")


def _as_binary_mask(mask: MaskInput) -> np.ndarray:
    if isinstance(mask, Image.Image):
        array = np.asarray(mask.convert("L"), dtype=np.uint8)
    else:
        array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must be a single-channel HxW image")
    threshold = 0 if array.size and array.max() <= 1 else 127
    return (array > threshold).astype(np.uint8)


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def _crop_geometry(
    mask: np.ndarray,
    image_size: tuple[int, int],
    size: int,
    *,
    warn_on_drop: bool,
) -> tuple[_CropGeometry | None, str | None]:
    image_width, image_height = image_size
    if size <= 0:
        raise ValueError("crop size must be positive")
    if mask.shape != (image_height, image_width):
        raise ValueError(
            f"Image/mask dimensions differ: image={image_size}, "
            f"mask={(mask.shape[1], mask.shape[0])}"
        )

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        if warn_on_drop:
            logger.warning("Dropping crop because the mask is empty")
        return None, "empty_mask"

    bbox_x0 = int(xs.min())
    bbox_y0 = int(ys.min())
    bbox_x1 = int(xs.max())
    bbox_y1 = int(ys.max())
    bbox_width = bbox_x1 - bbox_x0 + 1
    bbox_height = bbox_y1 - bbox_y0 + 1
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())

    if bbox_width <= size and bbox_height <= size:
        if image_width < size or image_height < size:
            if warn_on_drop:
                logger.warning(
                    "Dropping crop because frame %sx%s is smaller than crop size %s",
                    image_width,
                    image_height,
                    size,
                )
            return None, "frame_too_small"
        crop_x = _clamp(round(centroid_x - size / 2), 0, image_width - size)
        crop_y = _clamp(round(centroid_y - size / 2), 0, image_height - size)
        return _CropGeometry((crop_x, crop_y, size, size), resized=False), None

    square_size = max(bbox_width, bbox_height)
    if square_size > image_width or square_size > image_height:
        if warn_on_drop:
            logger.warning(
                "Dropping oversized defect: containing square %sx%s cannot fit "
                "inside frame %sx%s (bbox=%sx%s)",
                square_size,
                square_size,
                image_width,
                image_height,
                bbox_width,
                bbox_height,
            )
        return None, "unfittable_square"

    min_crop_x = max(0, bbox_x1 - square_size + 1)
    max_crop_x = min(bbox_x0, image_width - square_size)
    min_crop_y = max(0, bbox_y1 - square_size + 1)
    max_crop_y = min(bbox_y0, image_height - square_size)
    crop_x = _clamp(round(centroid_x - square_size / 2), min_crop_x, max_crop_x)
    crop_y = _clamp(round(centroid_y - square_size / 2), min_crop_y, max_crop_y)
    return _CropGeometry(
        (crop_x, crop_y, square_size, square_size),
        resized=True,
    ), None


def extract_defect_crop(
    image: ImageInput,
    mask: MaskInput,
    size: int = 256,
) -> DefectCrop | None:
    """Extract a mask-centred crop according to System Spec §3.4.

    Oversized containing squares are resized to ``size``. If the square
    cannot fit inside the frame, the sample is dropped with a warning.
    """

    rgb_image = _as_rgb_image(image)
    binary_mask = _as_binary_mask(mask)
    geometry, _ = _crop_geometry(
        binary_mask,
        rgb_image.size,
        size,
        warn_on_drop=True,
    )
    if geometry is None:
        return None

    crop_x, crop_y, crop_width, crop_height = geometry.crop_bbox
    crop_box = (crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)
    image_crop = rgb_image.crop(crop_box)
    mask_crop = binary_mask[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]

    if geometry.resized:
        image_crop = image_crop.resize((size, size), resample=Image.Resampling.BICUBIC)
        mask_image = Image.fromarray(mask_crop.astype(np.uint8) * 255)
        mask_crop = (
            np.asarray(
                mask_image.resize((size, size), resample=Image.Resampling.NEAREST),
                dtype=np.uint8,
            )
            > 127
        ).astype(np.uint8)

    return DefectCrop(
        image=image_crop,
        mask=mask_crop.astype(np.uint8),
        crop_offset=(crop_x, crop_y),
        crop_bbox=geometry.crop_bbox,
        resized_from_full_frame=geometry.resized,
    )


def composite_crop_back(
    full_clean: ImageInput,
    gen_crop: ImageInput,
    crop_offset: CropOffset,
    crop_mask: MaskInput,
    crop_bbox: CropBBox,
) -> Image.Image:
    """Resize and mask-composite a generated crop into its full frame.

    The operation implements ``m * generated + (1 - m) * clean``. Pixels
    outside the resized crop mask remain bit-identical to ``full_clean``.
    """

    clean_image = _as_rgb_image(full_clean)
    generated_image = _as_rgb_image(gen_crop)
    binary_mask = _as_binary_mask(crop_mask)
    if binary_mask.shape != (generated_image.height, generated_image.width):
        raise ValueError(
            "crop_mask dimensions must match gen_crop dimensions before placement"
        )

    crop_x, crop_y, crop_width, crop_height = (int(value) for value in crop_bbox)
    if tuple(int(value) for value in crop_offset) != (crop_x, crop_y):
        raise ValueError("crop_offset must equal the x/y origin of crop_bbox")
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("crop_bbox width and height must be positive")
    if (
        crop_x < 0
        or crop_y < 0
        or crop_x + crop_width > clean_image.width
        or crop_y + crop_height > clean_image.height
    ):
        raise ValueError("crop_bbox must lie fully inside full_clean")

    target_size = (crop_width, crop_height)
    if generated_image.size != target_size:
        generated_image = generated_image.resize(
            target_size,
            resample=Image.Resampling.BICUBIC,
        )
    mask_image = Image.fromarray(binary_mask * 255)
    if mask_image.size != target_size:
        mask_image = mask_image.resize(target_size, resample=Image.Resampling.NEAREST)

    clean_pixels = np.asarray(clean_image, dtype=np.uint8).copy()
    generated_pixels = np.asarray(generated_image, dtype=np.uint8)
    resized_mask = np.asarray(mask_image, dtype=np.uint8) > 127
    clean_region = clean_pixels[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ]
    clean_region[resized_mask] = generated_pixels[resized_mask]
    clean_pixels[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ] = clean_region
    return Image.fromarray(clean_pixels, mode="RGB")


def summarise_crop_eligibility(
    pairs: Sequence[DefectPair],
    size: int = 256,
    *,
    category: str | None = None,
) -> CropEligibilitySummary:
    """Count usable and excluded crops for one class without mutating data."""

    usable = 0
    excluded_unfittable = 0
    excluded_empty = 0
    excluded_small_frame = 0

    for image_path, mask_path in pairs:
        image = load_image_rgb(image_path)
        mask = load_mask_binary(mask_path)
        geometry, reason = _crop_geometry(
            mask,
            image.size,
            size,
            warn_on_drop=False,
        )
        if geometry is not None:
            usable += 1
        elif reason == "unfittable_square":
            excluded_unfittable += 1
        elif reason == "empty_mask":
            excluded_empty += 1
        elif reason == "frame_too_small":
            excluded_small_frame += 1

    summary = CropEligibilitySummary(
        total_pairs=len(pairs),
        usable_pairs=usable,
        excluded_unfittable_square=excluded_unfittable,
        excluded_empty_mask=excluded_empty,
        excluded_frame_too_small=excluded_small_frame,
    )
    logger.info(
        "Crop eligibility%s: total=%d usable=%d excluded=%d",
        f" for {category}" if category else "",
        summary.total_pairs,
        summary.usable_pairs,
        summary.crop_exclusion_count,
    )
    return summary
