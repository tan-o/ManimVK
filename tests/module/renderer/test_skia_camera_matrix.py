"""The Skia view matrix must honour a non-zero camera-frame center.

A sign error in the Y translation shifted every point by ``2 * sy * center_y``
pixels and made zoomed camera scenes render black.  This test records the
matrix passed to the native target without needing a GPU.
"""

from __future__ import annotations

import numpy as np

from manim.camera import Camera
from manim.renderer.skia.target import SkiaRenderTarget, _ImageFiles, _RasterSettings


class _RecordingNative:
    def __init__(self) -> None:
        self.matrix = None

    def clear_rgba(self, *args) -> None:
        pass

    def set_matrix(self, sx, sy, tx, ty) -> None:
        self.matrix = (sx, sy, tx, ty)


class _RecordingDevice:
    def __init__(self) -> None:
        self.native = _RecordingNative()

    def create_skia_target(self, width, height):
        return self.native


def test_camera_center_maps_to_pixel_center_with_negative_y_scale() -> None:
    settings = _RasterSettings(
        pixel_width=854,
        pixel_height=480,
        base_pixel_width=854,
        base_pixel_height=480,
    )
    camera = Camera(frame_center=np.array([0.0, -2.97, 0.0]))
    target = SkiaRenderTarget(settings, _ImageFiles(), device=_RecordingDevice())
    target.get_context(camera)
    sx, sy, tx, ty = target._native.matrix

    assert sy < 0
    # The frame center must land exactly at the middle of the target.
    x = 0.0 * sx + tx
    y = -2.97 * sy + ty
    assert np.isclose(x, settings.pixel_width / 2)
    assert np.isclose(y, settings.pixel_height / 2)
