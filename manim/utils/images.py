"""Image manipulation utilities."""

from __future__ import annotations

__all__ = [
    "get_full_raster_image_path",
    "drag_pixels",
    "invert_image",
    "change_to_rgba_array",
    "new_texture_key",
]

import itertools
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from .. import config
from ..utils.file_ops import seek_full_path_from_defaults

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manim.typing import PixelArray, RGBAPixelArray, RGBPixelArray


_texture_keys = itertools.count(1)


def new_texture_key() -> int:
    """A process-unique number naming one immutable pixel array.

    The renderer keeps a GPU texture per key, so the owner of a read-only
    array takes a new key whenever it replaces the array; drawing with the
    same key again reuses the upload.
    """
    return next(_texture_keys)


def get_full_raster_image_path(image_file_name: str | PurePath) -> Path:
    return seek_full_path_from_defaults(
        image_file_name,
        default_dir=config.get_dir("assets_dir"),
        extensions=[".jpg", ".jpeg", ".png", ".gif", ".ico"],
    )


def get_full_vector_image_path(image_file_name: str | PurePath) -> Path:
    return seek_full_path_from_defaults(
        image_file_name,
        default_dir=config.get_dir("assets_dir"),
        extensions=[".svg"],
    )


def drag_pixels(frames: Sequence[PixelArray]) -> list[np.ndarray]:
    curr = frames[0]
    new_frames: list[np.ndarray] = []
    for frame in frames:
        curr += (curr == 0) * np.array(frame)
        new_frames.append(np.array(curr))
    return new_frames


def invert_image(image: PixelArray) -> Image.Image:
    arr = np.array(image)
    arr = (255 * np.ones(arr.shape)).astype(arr.dtype) - arr
    return Image.fromarray(arr)


def change_to_rgba_array(image: RGBPixelArray, dtype: str = "uint8") -> RGBAPixelArray:
    """Converts an RGB array into RGBA with the alpha value opacity maxed."""
    pa = image
    if len(pa.shape) == 2:
        pa = pa.reshape(list(pa.shape) + [1])
    if pa.shape[2] == 1:
        pa = pa.repeat(3, axis=2)
    if pa.shape[2] == 3:
        alphas = 255 * np.ones(
            list(pa.shape[:2]) + [1],
            dtype=dtype,
        )
        pa = np.append(pa, alphas, axis=2)
    return pa
