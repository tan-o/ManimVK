"""Skia/Vulkan raster targets for the classic Manim mobject family."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from manim.renderer.cairo.target import (
    _CairoRasterSettings,
    _CairoRenderTarget,
    _ImageFiles,
)

if TYPE_CHECKING:
    from manim.renderer.cairo.camera import Camera
    from manim.typing import RGBAPixelArray


class SkiaRenderTarget(_CairoRenderTarget):
    """A Cairo-compatible target whose vector drawing is done by Skia/Ganesh.

    The public surface mirrors :class:`_CairoRenderTarget` so the existing
    classic-family drawing code can stay unchanged. Point-cloud and image
    mobjects still composite on the CPU; before the next vector batch their
    pixels are uploaded back into the GPU surface.
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
        self._cpu_dirty = False
        self._last_readback: RGBAPixelArray | None = None
        self._native.clear_rgba(0.0, 0.0, 0.0, 0.0)

    def _ensure_open(self) -> None:
        super()._ensure_open()
        if self._native is None:
            raise RuntimeError("The Skia render target is closed.")

    @property
    def pixels(self):
        """Expose the latest direct-read frame without a second full copy."""
        self._ensure_open()
        if self._last_readback is not None:
            return self._last_readback
        return self._pixels

    def _upload_if_cpu_dirty(self) -> None:
        if self._cpu_dirty:
            self._native.load_pixels(self._pixels.tobytes())
            self._cpu_dirty = False

    def reset(self, camera: Camera) -> None:
        super().reset(camera)
        color = camera.background_color.to_rgba_with_alpha(camera.background_opacity)
        self._native.clear_rgba(*color)
        if camera.background_image is not None:
            self._native.load_pixels(self._pixels.tobytes())
        self._gpu_dirty = False
        self._cpu_dirty = False
        self._last_readback = None

    def clear(self) -> None:
        # Drop the direct-read snapshot before the base implementation fills
        # ``pixels``; otherwise it would fill the snapshot and leave _pixels
        # untouched.
        self._last_readback = None
        super().clear()
        self._native.clear_rgba(0.0, 0.0, 0.0, 0.0)
        self._gpu_dirty = False
        self._cpu_dirty = False

    def set_pixels(self, pixels: RGBAPixelArray) -> None:
        super().set_pixels(pixels)
        self._native.load_pixels(self._pixels.tobytes())
        self._gpu_dirty = False
        self._cpu_dirty = False
        self._last_readback = None

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
        self._upload_if_cpu_dirty()
        center = camera.get_view_transform_center()
        settings = self.settings
        sx = settings.pixel_width / camera.frame_width
        sy = -settings.pixel_height / camera.frame_height
        tx = settings.pixel_width / 2 - center[0] * sx
        # y' = sy * y + ty, with sy negative.  At the frame center this is
        # exactly pixel_height / 2.
        ty = settings.pixel_height / 2 + center[1] * sy
        self._native.set_matrix(sx, sy, tx, ty)
        self._gpu_dirty = True
        return self._native

    def mark_cpu_dirty(self) -> None:
        """Tell the target that point-cloud/image code changed ``pixels``."""
        if self._last_readback is not None:
            # CPU compositing may have mutated the snapshot returned by
            # read_pixels(); bring that frame back into the reusable buffer
            # before the next GPU upload.
            np.copyto(self._pixels, self._last_readback)
            self._last_readback = None
        self._cpu_dirty = True

    def draw_vertices(self, positions, colors) -> None:
        """Draw pre-projected triangles with per-vertex RGBA colours."""
        self._ensure_open()
        self._native.draw_vertices(
            [(float(x), float(y)) for x, y in positions],
            [
                (
                    int(round(float(r) * 255)),
                    int(round(float(g) * 255)),
                    int(round(float(b) * 255)),
                    int(round(float(a) * 255)),
                )
                for r, g, b, a in colors
            ],
        )
        self._gpu_dirty = True

    def set_anti_alias(self, enabled: bool) -> None:
        """Toggle edge antialiasing for the next vector batch."""
        self._ensure_open()
        self._native.set_anti_alias(bool(enabled))

    def draw_hsb_quad(self, time: float) -> None:
        """Evaluate the full-screen HSB shader on the Skia Vulkan backend."""
        self._ensure_open()
        self._native.draw_hsb_quad(float(time))
        self._gpu_dirty = True

    def draw_design3_quad(self, time: float) -> None:
        """Evaluate Manim's raymarched ``design_3`` shader on Vulkan."""
        self._ensure_open()
        self._native.draw_design3_quad(float(time))
        self._gpu_dirty = True

    def commit(self) -> None:
        """Bring the GPU surface content back into ``pixels``."""
        if self._gpu_dirty:
            self._pixels[:, :] = np.frombuffer(
                bytes(self._native.commit()),
                dtype=np.uint8,
            ).reshape(self._pixels.shape)
            self._gpu_dirty = False
            self._cpu_dirty = False
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
            self._cpu_dirty = False
            self._last_readback = frame
            return frame
        if self._last_readback is not None:
            return self._last_readback.copy()
        return self._pixels.copy()

    def close(self) -> None:
        if self._closed:
            return
        self._native = None
        super().close()
