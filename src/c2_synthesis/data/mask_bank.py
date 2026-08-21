"""Mask sourcing for C2 generation."""

from __future__ import annotations

from typing import Sequence, TypeAlias

import numpy as np
from PIL import Image

ImageInput: TypeAlias = Image.Image | np.ndarray
MaskInput: TypeAlias = Image.Image | np.ndarray
Location: TypeAlias = tuple[int, int]


def _image_size(image: ImageInput) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        return image.size
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("clean_image must have shape HxW or HxWxC")
    return int(array.shape[1]), int(array.shape[0])


def _binary_mask(mask: MaskInput) -> np.ndarray:
    if isinstance(mask, Image.Image):
        array = np.asarray(mask.convert("L"), dtype=np.uint8)
    else:
        array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("real_mask must be a single-channel HxW array")
    threshold = 0 if array.size and array.max() <= 1 else 127
    return (array > threshold).astype(np.uint8)


def real_mask_transplant(
    clean_image: ImageInput,
    real_mask: MaskInput,
    location: Location,
) -> np.ndarray:
    """Place a real mask's tight defect support at an in-bounds location.

    ``location`` is the desired ``(x, y)`` top-left corner of the real mask's
    tight non-zero bounding box. The returned mask matches ``clean_image`` and
    contains values exactly in ``{0, 1}``.
    """

    clean_width, clean_height = _image_size(clean_image)
    binary = _binary_mask(real_mask)
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        raise ValueError("Cannot transplant an empty real defect mask")

    source_x0, source_x1 = int(xs.min()), int(xs.max()) + 1
    source_y0, source_y1 = int(ys.min()), int(ys.max()) + 1
    support = binary[source_y0:source_y1, source_x0:source_x1]
    target_x, target_y = (int(value) for value in location)
    if target_x < 0 or target_y < 0:
        raise ValueError("Mask transplant location must be non-negative")
    if target_x + support.shape[1] > clean_width or target_y + support.shape[0] > clean_height:
        raise ValueError(
            "Transplanted mask would exceed clean-image bounds: "
            f"location={location}, support={(support.shape[1], support.shape[0])}, "
            f"clean={(clean_width, clean_height)}"
        )

    transplanted = np.zeros((clean_height, clean_width), dtype=np.uint8)
    transplanted[
        target_y : target_y + support.shape[0],
        target_x : target_x + support.shape[1],
    ] = support
    return transplanted


def generate_masks(
    clean_images: Sequence[ImageInput],
    *,
    count: int,
    seed: int,
) -> list[np.ndarray]:
    """Optional learned mask-generator interface reserved for the ablation.

    Phase 3 deliberately uses real-mask transplantation as its primary mask
    source. The generated-mask model is implemented only in the later mask
    source ablation; keeping this explicit interface prevents silent fallback.
    """

    del clean_images, count, seed
    raise NotImplementedError(
        "Learned mask generation is an optional Phase 6 ablation; "
        "use real_mask_transplant during Phase 3."
    )
