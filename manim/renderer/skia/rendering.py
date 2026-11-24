"""Drawing mobjects into a :class:`~.SkiaRenderTarget` through one camera."""

from __future__ import annotations

import itertools as it
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import manim_core
import numpy as np
from PIL import Image

from manim.camera import DrawOrder
from manim.constants import CapStyleType, LineJointType
from manim.mobject.mobject import Mobject
from manim.mobject.sksl import SkSLMobject
from manim.mobject.three_d.mesh_surface import MeshSurface
from manim.mobject.types.image_mobject import (
    AbstractImageMobject,
    ImageMobjectFromCamera,
)
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

_ImageResolver = Callable[
    [ImageMobjectFromCamera], "SkiaRenderTarget | np.ndarray | None"
]

# Kinds of mobject, each drawn its own way.
_OTHER, _VMOBJECT, _BACKGROUND_COLORED, _MESH, _SKSL, _POINT_CLOUD, _IMAGE = range(7)


def _kind(mobject: Mobject) -> int:
    if isinstance(mobject, SkSLMobject):
        return _SKSL
    if isinstance(mobject, MeshSurface):
        return _MESH
    if isinstance(mobject, VMobject):
        return _BACKGROUND_COLORED if mobject.get_background_image() else _VMOBJECT
    if isinstance(mobject, PMobject):
        return _POINT_CLOUD
    if isinstance(mobject, (AbstractImageMobject, ImageMobjectFromCamera)):
        return _IMAGE
    return _OTHER


def _slice(order: DrawOrder, start: int, end: int) -> DrawOrder:
    return order._replace(
        owner=order.owner[start:end],
        start=order.start[start:end],
        end=order.end[start:end],
    )


def _single(vmobject: VMobject, pieces: np.ndarray | None) -> DrawOrder:
    """A draw order of one run: ``pieces`` of ``vmobject``, or all of it."""
    parts = (
        np.zeros(0, dtype=np.int64)
        if pieces is None
        else np.asarray(pieces, dtype=np.int64)
    )
    return DrawOrder(
        [vmobject],
        np.zeros(1, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        np.array([len(parts)], dtype=np.int64),
        parts,
    )


class SkiaDrawingContext:
    """Draw mobjects into one renderer-owned target using one camera."""

    def __init__(
        self,
        *,
        camera: Camera,
        target: SkiaRenderTarget,
        image_resolver: _ImageResolver,
    ) -> None:
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
        mobjects = list(mobjects)
        roots = {
            id(member): index
            for index, root in enumerate(mobjects)
            for member in root.get_family()
        }
        self._subpath_tables: dict[int, tuple] = {}
        self._specs: dict[tuple[int, bool], tuple] = {}
        self._meshes: dict[int, tuple] = {}
        with self.camera._drawing():
            to_display = self.camera.get_mobjects_to_display(
                mobjects,
                include_submobjects=include_submobjects,
                excluded_mobjects=excluded_mobjects,
            )
            order = self.camera.get_draw_order(to_display, roots)
            if not len(order.owner):
                return
            kinds = np.array(
                [_kind(mobject) for mobject in order.mobjects], dtype=np.int64
            )[order.owner]
            # Consecutive runs of one kind of mobject are drawn together;
            # VMobjects and meshes share one display list.
            batches = np.where(kinds == _MESH, _VMOBJECT, kinds)
            starts = np.flatnonzero(np.diff(batches, prepend=-1))
            ends = np.append(starts[1:], len(kinds))
            for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
                if batches[start] == _VMOBJECT:
                    self._display_list(order, kinds, start, end)
                else:
                    self._display_runs(int(kinds[start]), order, start, end)

    def _display_runs(self, kind: int, order: DrawOrder, start: int, end: int) -> None:
        runs = list(_slice(order, start, end).runs())
        if kind == _BACKGROUND_COLORED:
            # Whole runs sharing an image share one upload of it.
            for image, group in it.groupby(
                runs,
                lambda run: run[0].get_background_image() if run[1] is None else None,
            ):
                group = list(group)
                if image:
                    self._display_background_colored_vmobjects(
                        image, [vmobject for vmobject, _ in group]
                    )
                else:
                    for vmobject, pieces in group:
                        self._display_list(
                            _single(vmobject, pieces), np.array([_VMOBJECT]), 0, 1
                        )
        elif kind == _SKSL:
            self._display_sksl_mobjects([mobject for mobject, _ in runs])
        elif kind == _POINT_CLOUD:
            self._display_point_cloud_mobjects([mobject for mobject, _ in runs])
        elif kind == _IMAGE:
            self._display_image_mobjects([mobject for mobject, _ in runs])

    # -- vectorized mobjects ---------------------------------------------

    def _display_list(
        self, order: DrawOrder, kinds: np.ndarray, start: int, end: int
    ) -> None:
        """Draw runs of VMobjects and meshes in one native call: each mobject
        once (a VMobject's points, subpaths and paints; a mesh's shaded
        triangles), then the runs in order.
        """
        owners = order.owner[start:end]
        run_start, run_end = order.start[start:end], order.end[start:end]
        slots, first, inverse = np.unique(
            owners, return_index=True, return_inverse=True
        )
        pieced = np.zeros(len(slots), dtype=bool)
        np.logical_or.at(pieced, inverse, run_end > run_start)
        objects, meshes = [], []
        index = np.empty(len(slots), dtype=np.int64)
        for slot, (owner, kind, has_pieces) in enumerate(
            zip(
                slots.tolist(),
                kinds[start:end][first].tolist(),
                pieced.tolist(),
                strict=True,
            )
        ):
            mobject = order.mobjects[owner]
            if kind == _MESH:
                index[slot] = len(meshes)
                meshes.append(self._mesh_spec(mobject))
            else:
                index[slot] = len(objects)
                objects.append(self._cached_vmobject_spec(mobject, has_pieces))
        runs = np.stack(
            [
                np.where(kinds[start:end] == _MESH, 1, 0),
                index[inverse],
                run_start,
                run_end,
            ],
            axis=1,
        ).astype(np.int64)
        context = self.target.get_context(self.camera)
        context.draw_display_list(
            objects, meshes, runs, np.ascontiguousarray(order.parts, dtype=np.int64)
        )

    def _cached_vmobject_spec(self, vmobject: VMobject, pieced: bool) -> tuple:
        # A VMobject drawn in several display lists of a frame is prepared once.
        cached = self._specs.get((id(vmobject), pieced))
        if cached is None or cached[0] is not vmobject:
            cached = (vmobject, self._vmobject_spec(vmobject, pieced))
            self._specs[id(vmobject), pieced] = cached
        return cached[1]

    def _vmobject_spec(self, vmobject: VMobject, pieced: bool) -> tuple:
        """``(points, subpaths, paints)`` for :meth:`draw_vmobjects`.

        Depth pieces index the subpaths split on the 3D points
        (:meth:`_subpath_table`); a whole VMobject uses manim_core's split.
        """
        points, subpaths = (
            self._subpath_table(vmobject) if pieced else self._whole_path(vmobject)
        )
        paints = []
        gradient = None
        for kind in ("background_stroke", "fill", "stroke"):
            if kind == "fill":
                rgbas = self.camera.get_fill_rgbas(vmobject)
                if not np.any(rgbas[:, 3]):
                    # A transparent fill draws nothing; skip tessellating the path.
                    continue
                width = None
            else:
                background = kind == "background_stroke"
                width = vmobject.get_stroke_width(background)
                if width == 0:
                    continue
                rgbas = self.camera.get_stroke_rgbas(vmobject, background=background)
                width = float(width * self.target.settings.line_width_multiple)
            line = None
            if len(rgbas) > 1:
                if gradient is None:
                    ends = self.camera.transform_points_pre_display(
                        vmobject, vmobject.get_gradient_start_and_end_points()
                    )
                    gradient = (
                        float(ends[0][0]),
                        float(ends[0][1]),
                        float(ends[1][0]),
                        float(ends[1][1]),
                    )
                line = gradient
            # Manim stores straight RGBA, as Skia's colours take it.
            stops = [
                (float(offset), float(r), float(g), float(b), float(a))
                for (r, g, b, a), offset in zip(
                    rgbas, np.linspace(0, 1, len(rgbas)), strict=True
                )
            ]
            paints.append(
                (
                    width,
                    _LINE_JOIN_MAP[vmobject.joint_type],
                    _CAP_STYLE_MAP[vmobject.cap_style],
                    stops,
                    line,
                )
            )
        return points, subpaths, paints

    def _whole_path(self, vmobject: VMobject) -> tuple[np.ndarray, np.ndarray]:
        """Projected points and ``(start, end, closed)`` subpath rows of a
        whole VMobject; manim_core splits it and finds the closed subpaths.
        """
        return manim_core.cubic_subpaths(
            self.camera.transform_points_pre_display(vmobject, vmobject.points),
            vmobject.n_points_per_cubic_curve,
            2,
            vmobject.tolerance_for_point_equality,
            _COORDINATE_LIMIT,
        )

    def _set_path(self, context, vmobject: VMobject) -> None:
        context.set_cubic_path(*self._whole_path(vmobject))

    def _subpath_table(self, vmobject: VMobject) -> tuple[np.ndarray, np.ndarray]:
        """Projected points and ``(start, end, closed)`` rows per subpath, once
        per frame.

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
        if len(points) != len(vmobject.points) or not len(split):
            table = np.zeros((0, 3), dtype=np.int64)
        else:
            starts, ends = split[:, 0], split[:, 1]
            first, last = points[starts, :2], points[ends - 1, :2]
            tolerance = vmobject.tolerance_for_point_equality
            closed = np.all(
                np.abs(first - last) <= tolerance + 1e-5 * np.abs(last), axis=1
            )
            table = np.ascontiguousarray(
                np.stack([starts, ends, closed], axis=1), dtype=np.int64
            )
        self._subpath_tables[key] = (vmobject, points, table)
        return points, table

    def _set_stroke_style(self, context, vmobject: VMobject, width: float) -> None:
        context.set_line_width(width * self.target.settings.line_width_multiple)
        context.set_line_join(_LINE_JOIN_MAP[vmobject.joint_type])
        context.set_line_cap(_CAP_STYLE_MAP[vmobject.cap_style])

    def _display_background_colored_vmobjects(
        self, image, vmobjects: list[VMobject]
    ) -> None:
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
                    rgbas = self.camera.get_stroke_rgbas(
                        vmobject, background=background_stroke
                    )
                if not np.any(rgbas[:, 3]):
                    continue
                self.target.set_source_image(
                    background, (1.0, 1.0, 1.0, float(rgbas[:, 3].mean())), batch
                )
                if width is None:
                    context.fill_preserve()
                else:
                    self._set_stroke_style(context, vmobject, width)
                    context.stroke_preserve()

    # -- meshes, SkSL, point clouds and images ------------------------------

    def _mesh_spec(self, surface: MeshSurface) -> tuple:
        """A mesh for :meth:`draw_display_list`, once per frame: the whole
        surface shaded and projected, float32 positions, texture coordinates
        and colours with ``per_triangle`` rows per triangle (one triangle per
        layer), so depth pieces select rows.
        """
        key = id(surface)
        cached = self._meshes.get(key)
        if cached is not None and cached[0] is surface:
            return cached[1]
        texture, texture_key = surface.get_texture_atlas()
        mesh = surface.get_shaded_mesh(self.camera)
        if mesh is None:
            empty = np.zeros((0, 2), dtype=np.float32)
            positions, uvs, colors, per_triangle = (
                empty,
                empty,
                np.zeros((0, 4), np.float32),
                3,
            )
        else:
            positions, uvs, colors = mesh
            positions = self.camera.transform_points_pre_display(surface, positions)
            per_triangle = len(positions) // len(surface.triangles)
        # Depth pieces of a translucent surface first clear what was drawn
        # behind them to the background, as a depth buffer would hide it.
        occlude = None
        if surface.get_opacity() < 1:
            occlude = tuple(
                float(c)
                for c in self.camera.background_color.to_rgba_with_alpha(
                    self.camera.background_opacity
                )
            )
        spec = (
            texture_key,
            texture,
            texture.shape[1],
            texture.shape[0],
            np.ascontiguousarray(positions[:, :2], dtype=np.float32),
            np.ascontiguousarray(uvs, dtype=np.float32),
            np.ascontiguousarray(colors, dtype=np.float32),
            per_triangle,
            float(surface.get_opacity()),
            occlude,
        )
        self._meshes[key] = (surface, spec)
        return spec

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
        pixel_coords = self._points_to_subpixel_coords(pmobject, points).astype(
            np.int64
        )
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
        self.target.draw_point_pixels(
            np.ascontiguousarray(centers[on_screen], dtype=np.float32),
            np.ascontiguousarray(int_rgbas[on_screen]),
            size / 2,
        )

    def _adjusted_thickness(self, thickness: float) -> float:
        """A point size in pixels, scaled with the target from its base size."""
        settings = self.target.settings
        base_sum = settings.base_pixel_height + settings.base_pixel_width
        target_sum = settings.pixel_height + settings.pixel_width
        return 1 + (thickness - 1) * base_sum / target_sum

    def _display_image_mobjects(
        self, image_mobjects: list[AbstractImageMobject]
    ) -> None:
        for image_mobject in image_mobjects:
            self._display_image_mobject(image_mobject)

    def _display_image_mobject(self, image_mobject: AbstractImageMobject) -> None:
        target_coords = self._points_to_subpixel_coords(
            image_mobject, image_mobject.points
        )
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

    def _points_to_subpixel_coords(
        self, mobject: Mobject, points: Point3D_Array
    ) -> np.ndarray:
        """Device-pixel coordinates of ``points``, with the canvas matrix's view
        center (ThreeDCamera.project_points() has already removed frame_center).
        """
        points = self.camera.transform_points_pre_display(mobject, points)
        shifted_points = points - self.camera.get_view_transform_center()
        settings = self.target.settings
        result = np.zeros((len(points), 2))
        result[:, 0] = (
            shifted_points[:, 0] * settings.pixel_width / self.camera.frame_width
            + settings.pixel_width / 2
        )
        result[:, 1] = (
            -shifted_points[:, 1] * settings.pixel_height / self.camera.frame_height
            + settings.pixel_height / 2
        )
        return result
