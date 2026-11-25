"""Skia/Vulkan raster targets: one GPU surface each, read back on request."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from PIL import Image

from manim.scene.video_segment_encoder import YUV420Frame
from manim.utils.images import get_full_raster_image_path, new_texture_key

if TYPE_CHECKING:
    from manim.camera import Camera
    from manim.typing import RGBAPixelArray

__all__ = ["GpuMemory", "SkiaRenderTarget"]


class GpuMemory(NamedTuple):
    """GPU memory in use, in bytes.

    ``device`` is the driver's count for the device-local heaps, every
    allocation of the process included (None when the driver lacks
    ``VK_EXT_memory_budget``); ``skia`` is Skia's budgeted resource cache,
    of which ``skia_purgeable`` is only kept for reuse.
    """

    device: int | None
    skia: int
    skia_purgeable: int


@dataclass(frozen=True, slots=True)
class _RasterSettings:
    """The size of one target, and the size stroke widths are relative to."""

    pixel_width: int
    pixel_height: int
    base_pixel_width: int
    base_pixel_height: int
    #: Scene units per unit of ``stroke_width``.
    line_width_multiple: float = 0.01

    def __post_init__(self) -> None:
        if (
            min(
                self.pixel_width,
                self.pixel_height,
                self.base_pixel_width,
                self.base_pixel_height,
            )
            <= 0
        ):
            raise ValueError("Raster dimensions must be positive.")

    def resized(self, *, pixel_width: int, pixel_height: int) -> _RasterSettings:
        return _RasterSettings(
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            base_pixel_width=self.base_pixel_width,
            base_pixel_height=self.base_pixel_height,
            line_width_multiple=self.line_width_multiple,
        )


def _rgba_image(
    image: Image.Image | Path | str, size: tuple[int, int]
) -> RGBAPixelArray:
    """The pixels of ``image`` (a file or a PIL image) as RGBA, resized to ``size``."""
    if isinstance(image, (str, Path)):
        with Image.open(get_full_raster_image_path(image)) as source:
            rgba = source.convert("RGBA")
    else:
        rgba = image.convert("RGBA")
    if rgba.size != size:
        rgba = rgba.resize(size)
    return np.asarray(rgba, dtype=np.uint8).copy()


class _ImageFiles:
    """Image files read for one renderer, by path and size, shared by its targets."""

    def __init__(self) -> None:
        self._images: dict[tuple[str, tuple[int, int]], RGBAPixelArray] = {}
        # Texture keys of the cached arrays by their id, valid while cached.
        self._keys: dict[int, int] = {}

    def get(
        self, image: Image.Image | Path | str, size: tuple[int, int]
    ) -> RGBAPixelArray:
        # Only files are cached: an image object can change in place, and its
        # string form names a memory address that another image can reuse.
        if not isinstance(image, (str, Path)):
            return _rgba_image(image, size)
        key = (str(image), size)
        cached = self._images.get(key)
        if cached is None:
            cached = self._images[key] = _rgba_image(image, size)
            cached.flags.writeable = False
            self._keys[id(cached)] = new_texture_key()
        return cached

    def texture_key(self, pixels: RGBAPixelArray) -> int | None:
        """The texture key of an array returned by :meth:`get`, or None when
        the array is not cached (it came from an image object).
        """
        return self._keys.get(id(pixels))

    def clear(self) -> None:
        """Forget the images read so far."""
        self._images.clear()
        self._keys.clear()


class SkiaRenderTarget:
    """One Skia/Ganesh surface on the renderer's Vulkan device.

    Everything is drawn on the GPU, which always holds the frame; pixels come
    back only through :meth:`read_pixels` or :meth:`request_yuv420`.
    """

    def __init__(
        self, settings: _RasterSettings, image_files: _ImageFiles, *, device
    ) -> None:
        self.settings = settings
        #: Image files read for this target, shared with the renderer's other targets.
        self.image_files = image_files
        self._native = device.create_skia_target(
            settings.pixel_width, settings.pixel_height
        )
        self._background_key: tuple[object, ...] | None = None
        self._image_batches = itertools.count()
        self._native.clear_rgba(0.0, 0.0, 0.0, 0.0)
        self._owner_thread = threading.get_ident()

    def _ensure_open(self) -> None:
        if self._owner_thread != threading.get_ident():
            raise RuntimeError(
                "SkiaRenderTarget is bound to the thread that created it; "
                "move rendering back to that thread instead of using the target cross-thread."
            )
        if self._native is None:
            raise RuntimeError("The Skia render target is closed.")

    @property
    def _closed(self) -> bool:
        return self._native is None

    @property
    def _size(self) -> tuple[int, int]:
        return self.settings.pixel_width, self.settings.pixel_height

    def reset(self, camera: Camera) -> None:
        """Clear to the camera's background: its colour, or its image."""
        self._ensure_open()
        if camera.background_image is None:
            color = camera.background_color.to_rgba_with_alpha(
                camera.background_opacity
            )
            self._native.clear_rgba(*color)
            return
        key = (camera.background_image, self._size)
        if key != self._background_key:
            background = self.image_files.get(camera.background_image, self._size)
            self._native.set_background_layer(
                np.ascontiguousarray(background, dtype=np.uint8)
            )
            self._background_key = key
        self._native.draw_background_layer()

    def capture_static_layer(self) -> None:
        """Keep the current surface on the GPU as this play's static background."""
        self._ensure_open()
        self._native.capture_static_layer()

    def restore_static_layer(self) -> bool:
        """Start a frame from the static background; False if none is captured."""
        self._ensure_open()
        return bool(self._native.restore_static_layer())

    def clear_static_layer(self) -> None:
        if self._native is not None:
            self._native.clear_static_layer()

    def get_context(self, camera: Camera):
        """The native canvas, with ``camera``'s view as its matrix."""
        self._ensure_open()
        center = camera.get_view_transform_center()
        settings = self.settings
        sx = settings.pixel_width / camera.frame_width
        sy = -settings.pixel_height / camera.frame_height
        # x' = sx * x + tx and y' = sy * y + ty (sy is negative: Skia is y-down);
        # both map the frame center to the middle of the target.
        tx = settings.pixel_width / 2 - center[0] * sx
        ty = settings.pixel_height / 2 - center[1] * sy
        self._native.set_matrix(sx, sy, tx, ty)
        return self._native

    def get_background_image(self, image: Image.Image | Path | str) -> RGBAPixelArray:
        """The pixels of a vectorized mobject's background image at this target's
        size (read-only for files, which are cached).
        """
        self._ensure_open()
        return self.image_files.get(image, self._size)

    def draw_image_quad(
        self,
        key: int | None,
        pixels,
        quad,
        alpha: float,
        sampling: int = 2,
        mix=None,
    ) -> None:
        """Draw an RGBA image into a device-space quad on the GPU.

        ``key`` names the read-only ``pixels`` (see
        :func:`~.new_texture_key`): the GPU texture uploaded for a key is
        reused while the key keeps being drawn.  Pass None for pixels that
        change every frame.  ``sampling`` is 0 (nearest), 1 (bilinear) or
        2 (bicubic).  ``mix``, ``(key, pixels, weight)``, blends a second
        image of the same size over the first by ``weight``.
        """
        self._ensure_open()
        pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] != 4:
            return
        if mix is not None:
            mix_key, mix_pixels, weight = mix
            mix = (
                mix_key,
                np.ascontiguousarray(mix_pixels, dtype=np.uint8),
                float(weight),
            )
        self._native.draw_image_quad(
            key,
            pixels,
            pixels.shape[1],
            pixels.shape[0],
            quad,
            float(alpha),
            int(sampling),
            mix,
        )

    def draw_target_quad(self, source: SkiaRenderTarget, quad, alpha: float) -> None:
        """Composite another target's GPU surface without a CPU readback."""
        self._ensure_open()
        source._ensure_open()
        self._native.draw_target_quad(source._native, quad, float(alpha))

    def new_image_batch(self) -> int:
        """A number for one batch of :meth:`set_source_image` calls."""
        return next(self._image_batches)

    def set_source_image(self, pixels, rgba, batch: int | None = None) -> None:
        """Make a full-frame background image, tinted by ``rgba``, the source
        of the next fills and strokes.

        Image files come from the renderer's ``_ImageFiles``, which gives them
        texture keys, so each is uploaded once.  Other arrays may change
        between frames: pass the same ``batch`` number for every call that
        uses them in one batch, and they are uploaded once for that batch.
        """
        self._ensure_open()
        pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
        key = self.image_files.texture_key(pixels)
        if key is not None:
            self._native.set_source_image(pixels, *(float(c) for c in rgba), key)
        else:
            self._native.set_source_image(
                pixels, *(float(c) for c in rgba), batch, batch is not None
            )

    def draw_point_pixels(
        self, centers: np.ndarray, colors: np.ndarray, half_size: float
    ) -> None:
        """Draw point-cloud samples as squares of device pixels: ``(n, 2)``
        float32 centers and ``(n, 4)`` uint8 RGBA colors.
        """
        self._ensure_open()
        self._native.draw_point_pixels(centers, colors, float(half_size))

    def draw_sksl_quad(self, source: str, uniforms: dict[str, object], quad) -> None:
        """Compile and draw a user SkSL effect over a device-space quad."""
        self._ensure_open()
        normalized = []
        for name, value in uniforms.items():
            if isinstance(value, (int, float)):
                normalized.append((str(name), [float(value)]))
            else:
                normalized.append((str(name), [float(v) for v in value]))
        self._native.draw_sksl_quad(source, normalized, quad)

    def request_yuv420(self) -> Callable[[], YUV420Frame] | None:
        """Start reading the frame as BT.709 YUV 4:2:0, converted on the GPU.

        Returns a function that waits for the read and returns the frame, or
        None when the size is odd and 4:2:0 cannot hold it.  The read is
        ordered before later drawing, so the next frame can be drawn before
        the function is called; call it before the next request.
        """
        self._ensure_open()
        settings = self.settings
        if settings.pixel_width % 2 or settings.pixel_height % 2:
            return None
        ticket = self._native.request_yuv420()
        shape = (settings.pixel_height * 3 // 2, settings.pixel_width)

        def collect() -> YUV420Frame:
            data = np.empty(shape, dtype=np.uint8)
            self._native.collect_yuv420_into(ticket, data)
            return YUV420Frame(data)

        return collect

    def read_pixels(self) -> RGBAPixelArray:
        """Read the frame back into a fresh top-left RGBA array.

        The array is new on every call: a video writer takes ownership of it.
        """
        self._ensure_open()
        settings = self.settings
        frame = np.empty(
            (settings.pixel_height, settings.pixel_width, 4), dtype=np.uint8
        )
        self._native.commit_into(frame)
        return frame

    def set_occlusion_culling(self, enabled: bool) -> None:
        """Whether depth pieces that cannot be seen (outside the frame, or
        behind mesh surfaces) are left out of drawing; on by default.  The
        frame is the same either way."""
        self._ensure_open()
        self._native.set_occlusion_culling(bool(enabled))

    def culling_stats(self) -> tuple[int, int]:
        """``(tested, culled)`` chains of depth pieces in the last display
        list."""
        self._ensure_open()
        return tuple(self._native.culling_stats())

    def memory_usage(self) -> GpuMemory:
        self._ensure_open()
        return GpuMemory(*self._native.memory_usage())

    def close(self) -> None:
        self._native = None
