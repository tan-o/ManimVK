"""Drawing context for the Skia-backed classic renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from manim.mobject.sksl import SkSLMobject
from manim.mobject.types.image_mobject import ImageMobjectFromCamera
from manim.renderer.cairo.rendering import (
    _CAP_STYLE_MAP,
    _LINE_JOIN_MAP,
    _CairoDrawingContext,
)

if TYPE_CHECKING:

    pass


class SkiaDrawingContext(_CairoDrawingContext):
    """Cairo-compatible drawing semantics implemented with Skia paths."""

    def draw(
        self,
        mobjects,
        *,
        include_submobjects: bool = True,
        excluded_mobjects=None,
    ) -> None:
        import itertools as it

        from manim.mobject.mobject import Mobject
        from manim.mobject.types.image_mobject import AbstractImageMobject
        from manim.mobject.types.point_cloud_mobject import PMobject
        from manim.mobject.types.vectorized_mobject import VMobject

        self.target._ensure_open()
        display_funcs = {
            SkSLMobject: self._display_sksl_mobjects,
            VMobject: self._display_vectorized_mobjects,
            PMobject: self._display_point_cloud_mobjects,
            ImageMobjectFromCamera: self._display_image_mobjects,
            AbstractImageMobject: self._display_image_mobjects,
            Mobject: lambda batch: None,
        }

        def type_or_raise(mobject):
            for mobject_type in display_funcs:
                if isinstance(mobject, mobject_type):
                    return mobject_type
            raise TypeError(
                f"Displaying an object of class {type(mobject).__name__} is not supported",
            )

        with self.camera._drawing():
            to_display = self.camera.get_mobjects_to_display(
                mobjects,
                include_submobjects=include_submobjects,
                excluded_mobjects=excluded_mobjects,
            )
            for group_type, group in it.groupby(to_display, type_or_raise):
                display_funcs[group_type](list(group))

    def _display_sksl_mobjects(self, mobjects) -> None:
        for mobject in mobjects:
            coords = self._points_to_subpixel_coords(mobject, mobject.points)
            quad = [
                (float(coords[0][0]), float(coords[0][1])),
                (float(coords[1][0]), float(coords[1][1])),
                (float(coords[2][0]), float(coords[2][1])),
                (float(coords[3][0]), float(coords[3][1])),
            ]
            self.target.draw_sksl_quad(mobject.source, mobject.uniforms, quad)

    def _set_cairo_context_path(self, context, vmobject) -> None:
        """Send a VMobject's whole path to Skia in one native call.

        Same subpaths and closing rule as the Cairo version, which issues one
        call per point.
        """
        nppcc = vmobject.n_points_per_cubic_curve
        if nppcc != 4:
            # The native call consumes cubic Beziers only.
            super()._set_cairo_context_path(context, vmobject)
            return
        points = self.camera.transform_points_pre_display(vmobject, vmobject.points)
        subpaths: list[tuple[int, int, bool]] = []
        if len(points):
            for start, end in vmobject.get_subpath_split_indices_from_points(
                points, n_dims=2
            ):
                start, end = int(start), int(end)
                if end - start < nppcc:
                    continue
                closed = bool(
                    vmobject.consider_points_equals_2d(points[start], points[end - 1])
                )
                subpaths.append((start, end, closed))
        context.set_cubic_path(np.ascontiguousarray(points, dtype=np.float64), subpaths)

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
        background = self.target.get_background_image(image)
        context = self.target.get_context(self.camera)
        for vmobject in vmobjects:
            fill_opacity = float(vmobject.get_fill_opacity())
            if fill_opacity > 0:
                # Cairo masks the background image by the object's own alpha:
                # fill opacity 0 means *no image fill*, even though the path
                # exists.  Filling the path with the image unconditionally
                # turns every open StreamLine into a large opaque sliver: the
                # fill rule implicitly closes the open path.
                self._set_cairo_context_path(context, vmobject)
                self._apply_image_stroke(context, vmobject, background, is_background=True)
                self.target.fill_path_with_image(background)
            self._set_cairo_context_path(context, vmobject)
            self._apply_image_stroke(context, vmobject, background)

    def _sample_background_color(self, background, vmobject, point) -> tuple[float, ...]:
        coords = self._points_to_pixel_coords(
            vmobject,
            np.asarray([point], dtype=np.float64),
        )[0]
        x = int(np.clip(coords[0], 0, background.shape[1] - 1))
        y = int(np.clip(coords[1], 0, background.shape[0] - 1))
        rgba = background[y, x].astype(float) / 255.0
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]))

    def _apply_image_stroke(self, context, vmobject, image_pixels, *, is_background=False) -> None:
        """Paint a stroke with the image colors at the path endpoints.

        Cairo draws the object into an alpha scratch and multiplies the
        background image by that alpha, so the image -- not the white
        ``color_using_background_image`` placeholder -- colors the stroke.
        """
        width = vmobject.get_stroke_width(background=is_background)
        if width == 0:
            return
        opacity = float(vmobject.get_stroke_opacity(background=is_background))
        if opacity <= 0:
            return
        color = self._sample_background_color(
            image_pixels,
            vmobject,
            self.camera.transform_points_pre_display(
                vmobject,
                np.asarray([vmobject.get_center()], dtype=np.float64),
            )[0],
        )
        context.set_source_rgba(color[0], color[1], color[2], opacity)
        context.set_line_width(width * self.target.settings.cairo_line_width_multiple)
        context.set_line_join(int(_LINE_JOIN_MAP[vmobject.joint_type]))
        context.set_line_cap(int(_CAP_STYLE_MAP[vmobject.cap_style]))
        context.stroke_preserve()

    def _display_point_cloud_mobjects(self, pmobjects) -> None:
        for pmobject in pmobjects:
            self._draw_point_cloud_native(pmobject)

    def _draw_point_cloud_native(self, pmobject) -> None:
        points = pmobject.points
        if len(points) == 0:
            return
        pixel_coords = self._points_to_pixel_coords(pmobject, points)
        # Cairo expands every point into an integer ``thickness`` x
        # ``thickness`` block of pixels.  Calling ``_thickened_coordinates``
        # here *and* drawing a square of half-size ``thickness / 2`` doubles
        # both the point count and the footprint.  Draw one square per point
        # instead, anchored at the same top-left pixel Cairo uses.
        thickness = max(1, int(self._adjusted_thickness(pmobject.stroke_width)))
        rgbas = np.asarray(pmobject.rgbas, dtype=np.float64).reshape(-1, 4)
        if len(rgbas) == 0:
            return
        int_rgbas = (255 * rgbas).astype(np.uint8)
        on_screen = self._on_screen_pixels(pixel_coords)
        pixel_coords = pixel_coords[on_screen]
        int_rgbas = int_rgbas[on_screen]
        entries = [
            (
                float(x) + thickness / 2,
                float(y) + thickness / 2,
                int(r),
                int(g),
                int(b),
                int(a),
            )
            for (x, y), (r, g, b, a) in zip(pixel_coords, int_rgbas, strict=True)
        ]
        self.target.draw_point_pixels(entries, thickness / 2)

    def _display_image_mobjects(self, image_mobjects) -> None:
        for image_mobject in image_mobjects:
            self._draw_image_mobject_native(image_mobject)

    def _draw_image_mobject_native(self, image_mobject) -> None:
        source = (
            self.image_resolver(image_mobject)
            if isinstance(image_mobject, ImageMobjectFromCamera)
            else image_mobject.get_pixel_array()
        )
        if source is None:
            return
        target_coords = self._points_to_subpixel_coords(
            image_mobject,
            image_mobject.points,
        )
        # ImageMobject points are top-left, top-right, bottom-left, bottom-right.
        corners = [
            target_coords[0],
            target_coords[1],
            target_coords[3],
            target_coords[2],
        ]
        quad = [(float(x), float(y)) for x, y in corners]
        from manim.renderer.skia.target import SkiaRenderTarget

        if isinstance(source, SkiaRenderTarget):
            self.target.draw_target_quad(source, quad, 1.0)
            return
        source_pixels = np.ascontiguousarray(source, dtype=np.uint8)
        if source_pixels.ndim != 3 or source_pixels.shape[2] != 4:
            return
        self.target.draw_image_quad(
            source_pixels.tobytes(),
            source_pixels.shape[1],
            source_pixels.shape[0],
            quad,
            1.0,
        )
