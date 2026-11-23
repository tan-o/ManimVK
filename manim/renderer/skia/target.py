"""Skia/Vulkan raster targets for the classic Manim mobject family."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

from manim.renderer.cairo.target import (
    _CairoRasterSettings,
    _CairoRenderTarget,
    _ImageFiles,
)

if TYPE_CHECKING:
    from manim.camera import Camera
    from manim.typing import RGBAPixelArray


class SkiaRenderTarget(_CairoRenderTarget):
    """A Cairo-compatible target whose vector drawing is done by Skia/Ganesh.

    The public surface mirrors :class:`_CairoRenderTarget` so the existing
    classic-family drawing code can stay unchanged.  Everything is drawn on the
    GPU; ``pixels`` is only filled when a caller reads the frame back.
    """

    def __init__(
        self,
        settings: _CairoRasterSettings,
        image_files: _ImageFiles | None = None,
        *,
        device,
    ) -> None:
        super().__init__(settings, image_files)
        self._device = device
        self._native = device.create_skia_target(
            settings.pixel_width,
            settings.pixel_height,
        )
        self._gpu_dirty = False
        self._background_pending = False
        self._last_readback: RGBAPixelArray | None = None
        self._native.clear_rgba(0.0, 0.0, 0.0, 0.0)
        self._owner_thread = threading.get_ident()

    def _ensure_open(self) -> None:
        if self._owner_thread != threading.get_ident():
            raise RuntimeError(
                "SkiaRenderTarget is bound to the thread that created it; "
                "move rendering back to that thread instead of using the target cross-thread."
            )
        super()._ensure_open()
        if self._native is None:
            raise RuntimeError("The Skia render target is closed.")

    @property
    def pixels(self):
        """Expose the latest direct-read frame without a second full copy."""
        self._ensure_open()
        if self._last_readback is not None:
            return self._last_readback
        self._materialize_background()
        return self._pixels

    def reset(self, camera: Camera) -> None:
        """Clear to the camera background on the GPU.

        ``pixels`` is filled from the background only if this frame is read
        back with nothing drawn on it, so a cleared frame stays exact without
        a per-frame CPU copy.
        """
        self._ensure_open()
        key = self._camera_background_key(camera)
        if key != self._background_key:
            self._background = self._load_background(camera)
            self._background_key = key
            if camera.background_image is not None:
                self._native.set_background_layer(
                    np.ascontiguousarray(self._background, dtype=np.uint8)
                )
        if camera.background_image is None:
            color = camera.background_color.to_rgba_with_alpha(camera.background_opacity)
            self._native.clear_rgba(*color)
        else:
            self._native.draw_background_layer()
        self._gpu_dirty = False
        self._last_readback = None
        self._background_pending = True

    def _materialize_background(self) -> None:
        """Fill ``pixels`` with the background of a frame nothing was drawn on."""
        if self._background_pending and not self._gpu_dirty:
            np.copyto(self._pixels, self._background)
            self._background_pending = False

    def clear(self) -> None:
        self._ensure_open()
        self._native.clear_rgba(0.0, 0.0, 0.0, 0.0)
        self._pixels.fill(0)
        self._gpu_dirty = False
        self._last_readback = None
        self._background_pending = False

    def set_pixels(self, pixels: RGBAPixelArray) -> None:
        self._ensure_open()
        if pixels.shape != self._pixels.shape or pixels.dtype != np.uint8:
            raise ValueError(
                f"Skia target pixels must be uint8 with shape {self._pixels.shape}; "
                f"got {pixels.dtype} {pixels.shape}."
            )
        np.copyto(self._pixels, pixels)
        self._native.load_pixels(self._pixels)
        self._gpu_dirty = False
        self._last_readback = None
        self._background_pending = False

    def _mark_gpu_written(self) -> None:
        """The GPU surface changed; ``pixels`` is stale until the next read."""
        self._gpu_dirty = True
        self._last_readback = None
        self._background_pending = False

    def capture_static_layer(self) -> None:
        """Keep the current surface on the GPU as this play's static background."""
        self._ensure_open()
        self._native.capture_static_layer()

    def restore_static_layer(self) -> bool:
        """Start a frame from the static background; False if none is captured."""
        self._ensure_open()
        if not self._native.restore_static_layer():
            return False
        self._mark_gpu_written()
        return True

    def clear_static_layer(self) -> None:
        if self._native is not None:
            self._native.clear_static_layer()

    def get_scratch_target(self) -> SkiaRenderTarget:
        self._ensure_open()
        if self._scratch_target is None:
            self._scratch_target = SkiaRenderTarget(
                self.settings,
                self.image_files,
                device=self._device,
            )
        return self._scratch_target

    def get_context(self, camera: Camera):
        self._ensure_open()
        center = camera.get_view_transform_center()
        settings = self.settings
        sx = settings.pixel_width / camera.frame_width
        sy = -settings.pixel_height / camera.frame_height
        tx = settings.pixel_width / 2 - center[0] * sx
        # y' = sy * (y - center_y) + pixel_height / 2.  sy is negative to flip
        # Manim's y-up coordinates; the center term must therefore be
        # subtracted, not added.
        ty = settings.pixel_height / 2 - center[1] * sy
        self._native.set_matrix(sx, sy, tx, ty)
        self._mark_gpu_written()
        return self._native

    def draw_image_quad(self, pixels, width: int, height: int, quad, alpha: float) -> None:
        """Draw an RGBA image into a device-space quad on the GPU."""
        self._ensure_open()
        self._native.draw_image_quad(pixels, int(width), int(height), quad, float(alpha))
        self._mark_gpu_written()

    def draw_target_quad(self, source: SkiaRenderTarget, quad, alpha: float) -> None:
        """Composite another target's GPU surface without a CPU readback."""
        self._ensure_open()
        source._ensure_open()
        self._native.draw_target_quad(source._native, quad, float(alpha))
        self._mark_gpu_written()

    def fill_path_with_image(self, pixels) -> None:
        """Fill the current path with a full-frame background image.

        Read-only arrays are image files cached for the renderer's lifetime
        (see ``_ImageFiles``), so their identity keys the GPU upload; other
        arrays may change and are uploaded on each call.
        """
        self._ensure_open()
        pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
        key = None if pixels.flags.writeable else id(pixels)
        self._native.fill_path_with_image(pixels, key)
        self._mark_gpu_written()

    def draw_point_pixels(self, points, half_size: float) -> None:
        """Draw point-cloud samples in device pixels without CPU compositing."""
        self._ensure_open()
        self._native.draw_point_pixels(points, float(half_size))
        self._mark_gpu_written()

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
        self._mark_gpu_written()

    def commit(self) -> None:
        """Bring the GPU surface content back into ``pixels``."""
        self._materialize_background()
        if self._gpu_dirty:
            self._pixels[:, :] = np.frombuffer(
                bytes(self._native.commit()),
                dtype=np.uint8,
            ).reshape(self._pixels.shape)
            self._gpu_dirty = False
            self._last_readback = None
        elif self._last_readback is not None:
            # A direct read only populated the returned snapshot.  Keep the
            # reusable buffer authoritative again when a caller explicitly
            # asks for commit().
            np.copyto(self._pixels, self._last_readback)
            self._last_readback = None

    def read_pixels(self) -> RGBAPixelArray:
        """Return a fresh top-left RGBA frame, reading directly into it.

        A video writer takes ownership of the returned array, so the target
        cannot hand back its reusable ``_pixels`` buffer.  Allocating the
        destination first and using ``commit_into`` avoids the temporary
        ``bytes`` object and the extra copy in the old commit-then-copy path.
        """
        if self._gpu_dirty:
            frame = np.empty_like(self._pixels)
            self._native.commit_into(frame)
            self._gpu_dirty = False
            self._last_readback = frame
            return frame
        if self._last_readback is not None:
            return self._last_readback.copy()
        self._materialize_background()
        return self._pixels.copy()

    def close(self) -> None:
        if self._closed:
            return
        self._native = None
        super().close()
