"""Drawing mobjects into a :class:`~.SkiaRenderTarget` through one camera."""

from __future__ import annotations

import itertools as it
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import manim_core
import numpy as np
from PIL import Image

from manim.constants import CapStyleType, LineJointType
from manim.mobject.mobject import Mobject
from manim.mobject.sksl import SkSLMobject
from manim.mobject.three_d.mesh_surface import MeshSurface
from manim.mobject.types.image_mobject import AbstractImageMobject, ImageMobjectFromCamera
from manim.mobject.types.point_cloud_mobject import PMobject
from manim.mobject.types.vectorized_mobject import VMobject

from .target import SkiaRenderTarget

if TYPE_CHECKING:
    from manim.camera import Camera
    from manim.typing import Point3D_Array

# The native target's join and cap codes.
_LINE_JOIN_MAP = {
    LineJointType.AUTO: 0,
    LineJointType.ROUND: 1,
    LineJointType.BEVEL: 2,
    LineJointType.MITER: 0,
}

_CAP_STYLE_MAP = {
    CapStyleType.AUTO: 0,
    CapStyleType.ROUND: 1,
    CapStyleType.BUTT: 0,
    CapStyleType.SQUARE: 2,
}

# ImageMobject.resampling_algorithm to the native sampling: nearest, bilinear
# or bicubic (Catmull-Rom).
_SAMPLING = {
    Image.Resampling.NEAREST: 0,
    Image.Resampling.BOX: 1,
    Image.Resampling.BILINEAR: 1,
    Image.Resampling.HAMMING: 1,
    Image.Resampling.BICUBIC: 2,
    Image.Resampling.LANCZOS: 2,
}

# Skia paths are single precision and break down on huge coordinates, such as
# a function plotted through a pole.  Curves reaching beyond this many scene
# units are drawn as straight lines between their anchors, clamped to it
# (manim_core.tame_huge_curves): off-screen either way, and a pole still shows
# as a vertical line.
_COORDINATE_LIMIT = 1e5

_ImageResolver = Callable[[ImageMobjectFromCamera], "SkiaRenderTarget | np.ndarray | None"]


class SkiaDrawingContext:
    """Draw mobjects into one renderer-owned target using one camera."""

    def __init__(self, *, camera: Camera, target: SkiaRenderTarget, image_resolver: _ImageResolver) -> None:
        self.camera = camera
        self.target = target
        self.image_resolver = image_resolver

    def draw(
        self,
        mobjects: Iterable[Mobject],
        *,
        include_submobjects: bool = True,
        excluded_mobjects: list[Mobject] | None = None,
    ) -> None:
        self.target._ensure_open()
        display_funcs: dict[type[Mobject], Callable[[list[Any]], None]] = {
            SkSLMobject: self._display_sksl_mobjects,
            VMobject: self._display_vectorized_items,
            MeshSurface: self._display_mesh_surfaces,
            PMobject: self._display_point_cloud_mobjects,
            ImageMobjectFromCamera: self._display_image_mobjects,
            AbstractImageMobject: self._display_image_mobjects,
            Mobject: lambda batch: None,
        }
        takes_items = {VMobject, MeshSurface}

        def type_or_raise(item) -> type[Mobject]:
            for mobject_type in display_funcs:
                if isinstance(item[0], mobject_type):
                    return mobject_type
            raise TypeError(
                f"Displaying an object of class {type(item[0]).__name__} is not supported",
            )

        mobjects = list(mobjects)
        roots = {id(member): index for index, root in enumerate(mobjects) for member in root.get_family()}
        self._subpath_tables: dict[int, tuple] = {}
        self._vertex_layers: dict[int, tuple] = {}
        with self.camera._drawing():
            to_display = self.camera.get_mobjects_to_display(
                mobjects,
                include_submobjects=include_submobjects,
                excluded_mobjects=excluded_mobjects,
            )
            items = self.camera.get_draw_items(to_display, roots)
            for group_type, group in it.groupby(items, type_or_raise):
                group = list(group)
                if group_type in takes_items:
                    display_funcs[group_type](group)
                else:
                    display_funcs[group_type]([mobject for mobject, _ in group])

    # -- vectorized mobjects ---------------------------------------------

    def _display_vectorized_items(self, items) -> None:
        """Draw whole VMobjects in runs and depth pieces one at a time."""
        whole = []
        for vmobject, pieces in items:
            if pieces is None:
                whole.append(vmobject)
                continue
            self._display_vectorized_mobjects(whole)
            whole = []
            self._display_vmobject_pieces(vmobject, pieces)
        self._display_vectorized_mobjects(whole)

    def _display_vectorized_mobjects(self, vmobjects: list[VMobject]) -> None:
        for image, run in it.groupby(vmobjects, lambda vmobject: vmobject.get_background_image()):
            if image:
                self._display_background_colored_vmobjects(image, list(run))
            else:
                context = self.target.get_context(self.camera)
                for vmobject in run:
                    self._set_path(context, vmobject)
                    self._paint(context, vmobject)

    def _paint(self, context, vmobject: VMobject) -> None:
        """Background stroke, fill, then stroke, on the current path."""
        self._apply_stroke(context, vmobject, background=True)
        self._apply_fill(context, vmobject)
        self._apply_stroke(context, vmobject)

    def _set_path(self, context, vmobject: VMobject) -> None:
        """Send a VMobject's whole path in one native call; manim_core splits
        it into subpaths and finds the closed ones.
        """
        points = self.camera.transform_points_pre_display(vmobject, vmobject.points)
        points, subpaths = manim_core.cubic_subpaths(
            points,
            vmobject.n_points_per_cubic_curve,
            2,
            vmobject.tolerance_for_point_equality,
            _COORDINATE_LIMIT,
        )
        context.set_cubic_path(points, subpaths)

    def _subpath_table(self, vmobject: VMobject):
        """Projected points and (start, end, closed) per subpath, once per frame.

        Subpaths are split on the 3D points, as in
        :meth:`.VMobject.get_depth_pieces`, so piece indices line up.
        """
        key = id(vmobject)
        cached = self._subpath_tables.get(key)
        if cached is not None and cached[0] is vmobject:
            return cached[1], cached[2]
        split = vmobject.get_subpath_split_indices_from_points(vmobject.points)
        points = np.ascontiguousarray(
            manim_core.tame_huge_curves(
                self.camera.transform_points_pre_display(vmobject, vmobject.points),
                vmobject.n_points_per_cubic_curve,
                _COORDINATE_LIMIT,
            ),
            dtype=np.float64,
        )
        if len(points) != len(vmobject.points):
            table = []
        else:
            starts, ends = split[:, 0], split[:, 1]
            first, last = points[starts, :2], points[ends - 1, :2]
            tolerance = vmobject.tolerance_for_point_equality
            closed = np.all(np.abs(first - last) <= tolerance + 1e-5 * np.abs(last), axis=1)
            table = list(zip(starts.tolist(), ends.tolist(), closed.tolist(), strict=True))
        self._subpath_tables[key] = (vmobject, points, table)
        return points, table

    def _display_vmobject_pieces(self, vmobject: VMobject, pieces) -> None:
        points, table = self._subpath_table(vmobject)
        if not table:
            return
        context = self.target.get_context(self.camera)
        context.set_cubic_path(points, [table[index] for index in pieces.tolist()])
        self._paint(context, vmobject)

    def _set_source(self, context, rgbas, vmobject: VMobject) -> None:
        if len(rgbas) == 1:
            # Manim stores straight RGBA, as set_source_rgba takes it.
            context.set_source_rgba(*rgbas[0])
            return
        points = vmobject.get_gradient_start_and_end_points()
        points = self.camera.transform_points_pre_display(vmobject, points)
        stops = [
            (offset, rgba[0], rgba[1], rgba[2], rgba[3])
            for rgba, offset in zip(rgbas, np.linspace(0, 1, len(rgbas)), strict=True)
        ]
        context.set_source_linear_gradient(points[0][0], points[0][1], points[1][0], points[1][1], stops)

    def _apply_fill(self, context, vmobject: VMobject) -> None:
        rgbas = self.camera.get_fill_rgbas(vmobject)
        if not np.any(rgbas[:, 3]):
            # A transparent fill draws nothing; skip tessellating the path.
            return
        self._set_source(context, rgbas, vmobject)
        context.fill_preserve()

    def _set_stroke_style(self, context, vmobject: VMobject, width: float) -> None:
        context.set_line_width(width * self.target.settings.line_width_multiple)
        context.set_line_join(_LINE_JOIN_MAP[vmobject.joint_type])
        context.set_line_cap(_CAP_STYLE_MAP[vmobject.cap_style])

    def _apply_stroke(self, context, vmobject: VMobject, background: bool = False) -> None:
        width = vmobject.get_stroke_width(background)
        if width == 0:
            return
        self._set_source(context, self.camera.get_stroke_rgbas(vmobject, background=background), vmobject)
        self._set_stroke_style(context, vmobject, width)
        context.stroke_preserve()

    def _display_background_colored_vmobjects(self, image, vmobjects: list[VMobject]) -> None:
        """Draw vmobjects coloured by a background image: each fill and stroke
        paints with the image, faded by the opacity it would have.
        """
        background = self.target.get_background_image(image)
        batch = self.target.new_image_batch()
        context = self.target.get_context(self.camera)
        for vmobject in vmobjects:
            self._set_path(context, vmobject)
            for kind in ("background_stroke", "fill", "stroke"):
                if kind == "fill":
                    rgbas = self.camera.get_fill_rgbas(vmobject)
                    width = None
                else:
                    background_stroke = kind == "background_stroke"
                    width = vmobject.get_stroke_width(background_stroke)
                    if width == 0:
                        continue
                    rgbas = self.camera.get_stroke_rgbas(vmobject, background=background_stroke)
                if not np.any(rgbas[:, 3]):
                    continue
                self.target.set_source_image(background, (1.0, 1.0, 1.0, float(rgbas[:, 3].mean())), batch)
                if width is None:
                    context.fill_preserve()
                else:
                    self._set_stroke_style(context, vmobject, width)
                    context.stroke_preserve()

    # -- meshes, SkSL, point clouds and images ------------------------------

    def _display_mesh_surfaces(self, items) -> None:
        for surface, pieces in items:
            key = id(surface)
            layers = self._vertex_layers.get(key)
            if layers is None or layers[0] is not surface:
                layers = (surface, surface.get_vertex_layers(self.camera))
                self._vertex_layers[key] = layers
            mesh = surface.get_shaded_mesh(self.camera, pieces, layers[1])
            if mesh is None:
                continue
            positions, uvs, colors = mesh
            positions = self.camera.transform_points_pre_display(surface, positions)
            # Pieces only come from a depth-sorting camera.  There a translucent
            # surface clears what was drawn behind it to the background first,
            # as a depth buffer would hide it.
            occlude = None
            if pieces is not None and surface.get_opacity() < 1:
                occlude = tuple(
                    float(c)
                    for c in self.camera.background_color.to_rgba_with_alpha(self.camera.background_opacity)
                )
            self.target.get_context(self.camera)
            texture, texture_key = surface.get_texture_atlas()
            self.target.draw_textured_mesh(
                texture_key,
                texture,
                np.ascontiguousarray(positions[:, :2], dtype=np.float32),
                np.ascontiguousarray(uvs, dtype=np.float32),
                np.ascontiguousarray(colors, dtype=np.float32),
                surface.get_opacity(),
                occlude,
            )

    def _display_sksl_mobjects(self, mobjects: list[SkSLMobject]) -> None:
        for mobject in mobjects:
            coords = self._points_to_subpixel_coords(mobject, mobject.points)
            quad = [(float(x), float(y)) for x, y in coords[:4]]
            self.target.draw_sksl_quad(mobject.source, mobject.uniforms, quad)

    def _display_point_cloud_mobjects(self, pmobjects: list[PMobject]) -> None:
        for pmobject in pmobjects:
            self._display_point_cloud(pmobject)

    def _display_point_cloud(self, pmobject: PMobject) -> None:
        """Draw each point as a square block of ``stroke_width`` device pixels."""
        points = pmobject.points
        if len(points) == 0:
            return
        rgbas = np.asarray(pmobject.rgbas, dtype=np.float64).reshape(-1, 4)
        if len(rgbas) == 0:
            return
        pixel_coords = self._points_to_subpixel_coords(pmobject, points).astype(np.int64)
        # The block covers pixels x + k for k in range(-size // 2 + 1, size // 2 + 1).
        size = max(1, int(self._adjusted_thickness(pmobject.stroke_width)))
        low, high = -size // 2 + 1, size // 2 + 1
        centers = pixel_coords + (low + high) / 2
        int_rgbas = (255 * rgbas).astype(np.uint8)
        if len(int_rgbas) != len(centers):
            int_rgbas = np.resize(int_rgbas, (len(centers), 4))
        settings = self.target.settings
        on_screen = (
            (pixel_coords[:, 0] >= 0)
            & (pixel_coords[:, 0] < settings.pixel_width)
            & (pixel_coords[:, 1] >= 0)
            & (pixel_coords[:, 1] < settings.pixel_height)
        )
        entries = [
            (float(x), float(y), int(r), int(g), int(b), int(a))
            for (x, y), (r, g, b, a) in zip(centers[on_screen], int_rgbas[on_screen], strict=True)
        ]
        self.target.draw_point_pixels(entries, size / 2)

    def _adjusted_thickness(self, thickness: float) -> float:
        """A point size in pixels, scaled with the target from its base size."""
        settings = self.target.settings
        base_sum = settings.base_pixel_height + settings.base_pixel_width
        target_sum = settings.pixel_height + settings.pixel_width
        return 1 + (thickness - 1) * base_sum / target_sum

    def _display_image_mobjects(self, image_mobjects: list[AbstractImageMobject]) -> None:
        for image_mobject in image_mobjects:
            self._display_image_mobject(image_mobject)

    def _display_image_mobject(self, image_mobject: AbstractImageMobject) -> None:
        target_coords = self._points_to_subpixel_coords(image_mobject, image_mobject.points)
        # ImageMobject points are top-left, top-right, bottom-left, bottom-right.
        quad = [(float(x), float(y)) for x, y in target_coords[[0, 1, 3, 2]]]
        sampling = _SAMPLING.get(image_mobject.resampling_algorithm, 2)
        if isinstance(image_mobject, ImageMobjectFromCamera):
            source = self.image_resolver(image_mobject)
            if isinstance(source, SkiaRenderTarget):
                self.target.draw_target_quad(source, quad, 1.0)
            elif source is not None:
                self.target.draw_image_quad(None, source, quad, 1.0, sampling)
            return
        # The source pixels keep their GPU texture across frames; opacity and
        # a transform into another image are applied while drawing.
        mix = image_mobject.get_image_mix()
        if mix is not None:
            pixels, key, alpha = mix
            mix = (key, pixels, alpha)
        self.target.draw_image_quad(
            image_mobject.texture_key,
            image_mobject.pixel_array,
            quad,
            image_mobject.get_opacity(),
            sampling,
            mix,
        )

    def _points_to_subpixel_coords(self, mobject: Mobject, points: Point3D_Array) -> np.ndarray:
        """Device-pixel coordinates of ``points``, with the canvas matrix's view
        center (ThreeDCamera.project_points() has already removed frame_center).
        """
        points = self.camera.transform_points_pre_display(mobject, points)
        shifted_points = points - self.camera.get_view_transform_center()
        settings = self.target.settings
        result = np.zeros((len(points), 2))
        result[:, 0] = shifted_points[:, 0] * settings.pixel_width / self.camera.frame_width + settings.pixel_width / 2
        result[:, 1] = (
            -shifted_points[:, 1] * settings.pixel_height / self.camera.frame_height + settings.pixel_height / 2
        )
        return result
