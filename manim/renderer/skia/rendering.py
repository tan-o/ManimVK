"""Drawing context for the Skia-backed classic renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from manim.renderer.cairo.rendering import (
    _CAP_STYLE_MAP,
    _LINE_JOIN_MAP,
    _CairoDrawingContext,
)

if TYPE_CHECKING:

    pass


class SkiaDrawingContext(_CairoDrawingContext):
    """Cairo-compatible drawing semantics implemented with Skia paths."""

    def _set_cairo_context_color(self, context, rgbas, vmobject) -> None:
        if len(rgbas) == 1:
            # Manim stores straight RGBA, matching Skia's set_source_rgba.
            context.set_source_rgba(*rgbas[0])
            return

        points = vmobject.get_gradient_start_and_end_points()
        points = self.camera.transform_points_pre_display(vmobject, points)
        stops = [
            (offset, rgba[0], rgba[1], rgba[2], rgba[3])
            for rgba, offset in zip(
                rgbas,
                np.linspace(0, 1, len(rgbas)),
                strict=True,
            )
        ]
        context.set_source_linear_gradient(
            points[0][0],
            points[0][1],
            points[1][0],
            points[1][1],
            stops,
        )

    def _apply_stroke(self, context, vmobject, background: bool = False) -> None:
        width = vmobject.get_stroke_width(background)
        if width == 0:
            return
        self._set_cairo_context_color(
            context,
            self.camera.get_stroke_rgbas(vmobject, background=background),
            vmobject,
        )
        context.set_line_width(
            width * self.target.settings.cairo_line_width_multiple,
        )
        context.set_line_join(int(_LINE_JOIN_MAP[vmobject.joint_type]))
        context.set_line_cap(int(_CAP_STYLE_MAP[vmobject.cap_style]))
        context.stroke_preserve()

    def _display_non_background_colored_vmobjects(self, vmobjects) -> None:
        # The whole batch is drawn on the GPU.  Do not read it back here:
        # read_pixels() commits once at the end of the frame.  The old eager
        # commit made vector-only scenes pay two full GPU-to-CPU copies.
        super()._display_non_background_colored_vmobjects(vmobjects)

    def _display_background_colored_vmobjects(self, image, vmobjects) -> None:
        if not vmobjects:
            return
        # CPU compositing below needs the vector content currently on the GPU.
        self.target.commit()
        scratch = self.target.get_scratch_target()
        scratch.clear()
        scratch_context = SkiaDrawingContext(
            camera=self.camera,
            target=scratch,
            image_resolver=self.image_resolver,
        )
        scratch_context._display_non_background_colored_vmobjects(vmobjects)
        scratch.commit()
        background = self.target.get_background_image(image)
        colored = np.asarray(
            background * scratch.pixels.astype(float) / 255,
            dtype=np.uint8,
        )
        self._overlay_rgba_array(self.target.pixels, colored)
        self.target.mark_cpu_dirty()

    def _display_point_cloud_mobjects(self, pmobjects) -> None:
        # Point clouds are written into target.pixels on the CPU.
        self.target.commit()
        super()._display_point_cloud_mobjects(pmobjects)
        self.target.mark_cpu_dirty()

    def _display_image_mobjects(self, image_mobjects) -> None:
        # Images are alpha-composited into target.pixels on the CPU.
        self.target.commit()
        super()._display_image_mobjects(image_mobjects)
        self.target.mark_cpu_dirty()
