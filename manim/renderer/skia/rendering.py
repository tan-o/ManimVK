"""Drawing context for the Skia-backed classic renderer."""

from __future__ import annotations


import manim_core
import numpy as np

from manim.mobject.sksl import SkSLMobject
from manim.mobject.types.image_mobject import ImageMobjectFromCamera
from manim.renderer.cairo.rendering import (
    _CAP_STYLE_MAP,
    _LINE_JOIN_MAP,
    _CairoDrawingContext,
)

from PIL import Image

# ImageMobject.resampling_algorithm to the native sampling: PIL's perspective
# transform, which the Cairo renderer uses, knows nearest, bilinear and bicubic.
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
        from manim.mobject.three_d.mesh_surface import MeshSurface
        from manim.mobject.types.image_mobject import AbstractImageMobject
        from manim.mobject.types.point_cloud_mobject import PMobject
        from manim.mobject.types.vectorized_mobject import VMobject

        self.target._ensure_open()
        display_funcs = {
            SkSLMobject: self._display_sksl_mobjects,
            VMobject: self._display_vectorized_items,
            MeshSurface: self._display_mesh_surfaces,
            PMobject: self._display_point_cloud_mobjects,
            ImageMobjectFromCamera: self._display_image_mobjects,
            AbstractImageMobject: self._display_image_mobjects,
            Mobject: lambda batch: None,
        }
        takes_items = {VMobject, MeshSurface}

        def type_or_raise(item):
            for mobject_type in display_funcs:
                if isinstance(item[0], mobject_type):
                    return mobject_type
            raise TypeError(
                f"Displaying an object of class {type(item[0]).__name__} is not supported",
            )

        mobjects = list(mobjects)
        roots = {
            id(member): index
            for index, root in enumerate(mobjects)
            for member in root.get_family()
        }
        self._subpath_tables = {}
        self._vertex_layers = {}
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

    def _display_vectorized_items(self, items) -> None:
        """Draw whole VMobjects in batches and depth pieces one run at a time."""
        whole = []
        for vmobject, pieces in items:
            if pieces is None:
                whole.append(vmobject)
                continue
            self._display_vectorized_mobjects(whole)
            whole = []
            self._display_vmobject_pieces(vmobject, pieces)
        self._display_vectorized_mobjects(whole)

    def _subpath_table(self, vmobject):
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
            closed = np.all(
                np.abs(first - last) <= tolerance + 1e-5 * np.abs(last), axis=1
            )
            table = list(zip(starts.tolist(), ends.tolist(), closed.tolist()))
        self._subpath_tables[key] = (vmobject, points, table)
        return points, table

    def _display_vmobject_pieces(self, vmobject, pieces) -> None:
        points, table = self._subpath_table(vmobject)
        if not table:
            return
        context = self.target.get_context(self.camera)
        context.set_cubic_path(points, [table[index] for index in pieces.tolist()])
        self._apply_stroke(context, vmobject, background=True)
        self._apply_fill(context, vmobject)
        self._apply_stroke(context, vmobject)

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
                    for c in self.camera.background_color.to_rgba_with_alpha(
                        self.camera.background_opacity
                    )
                )
            self.target.get_context(self.camera)
            texture = surface.get_texture_atlas()
            self.target.draw_textured_mesh(
                id(texture),
                texture,
                np.ascontiguousarray(positions[:, :2], dtype=np.float32),
                np.ascontiguousarray(uvs, dtype=np.float32),
                np.ascontiguousarray(colors, dtype=np.float32),
                surface.get_opacity(),
                occlude,
            )

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

        manim_core splits it into subpaths and decides which are closed with
        the Cairo renderer's rules, in one call.
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

    def _apply_fill(self, context, vmobject) -> None:
        rgbas = self.camera.get_fill_rgbas(vmobject)
        if not np.any(rgbas[:, 3]):
            # A transparent fill draws nothing; skip tessellating the path.
            return
        self._set_cairo_context_color(context, rgbas, vmobject)
        context.fill_preserve()

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
        """Draw vmobjects coloured by a background image.

        Cairo draws them into a scratch target and uses its alpha as a mask
        on the image; here each fill and stroke paints with the image, faded
        by the opacity it would have, so the same coverage shows the image.
        """
        if not vmobjects:
            return
        background = self.target.get_background_image(image)
        batch = self.target.new_image_batch()
        context = self.target.get_context(self.camera)
        for vmobject in vmobjects:
            self._set_cairo_context_path(context, vmobject)
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
                self.target.set_source_image(
                    background, (1.0, 1.0, 1.0, float(rgbas[:, 3].mean())), batch
                )
                if width is None:
                    context.fill_preserve()
                else:
                    context.set_line_width(width * self.target.settings.cairo_line_width_multiple)
                    context.set_line_join(int(_LINE_JOIN_MAP[vmobject.joint_type]))
                    context.set_line_cap(int(_CAP_STYLE_MAP[vmobject.cap_style]))
                    context.stroke_preserve()

    def _display_point_cloud_mobjects(self, pmobjects) -> None:
        for pmobject in pmobjects:
            self._draw_point_cloud_native(pmobject)

    def _draw_point_cloud_native(self, pmobject) -> None:
        """Draw each point as the square block of pixels the Cairo renderer sets."""
        points = pmobject.points
        if len(points) == 0:
            return
        rgbas = np.asarray(pmobject.rgbas, dtype=np.float64).reshape(-1, 4)
        if len(rgbas) == 0:
            return
        pixel_coords = self._points_to_pixel_coords(pmobject, points)
        # Cairo sets pixels x + k for k in range(-t // 2 + 1, t // 2 + 1).
        size = max(1, int(self._adjusted_thickness(pmobject.stroke_width)))
        low, high = -size // 2 + 1, size // 2 + 1
        centers = pixel_coords + (low + high) / 2
        int_rgbas = (255 * rgbas).astype(np.uint8)
        if len(int_rgbas) != len(centers):
            int_rgbas = np.resize(int_rgbas, (len(centers), 4))
        on_screen = self._on_screen_pixels(pixel_coords)
        entries = [
            (float(x), float(y), int(r), int(g), int(b), int(a))
            for (x, y), (r, g, b, a) in zip(centers[on_screen], int_rgbas[on_screen], strict=True)
        ]
        self.target.draw_point_pixels(entries, size / 2)

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
        # PIL's perspective transform is the reference implementation used by
        # Cairo, and no GPU sampler matches its bicubic kernel exactly.  Raster
        # the plain ImageMobject on the CPU (the same code path as Cairo), then
        # upload the finished frame 1:1, where no resampling can introduce a
        # different phase.  ImageMobjectFromCamera stays on the GPU path above.
        full_image = self._rasterize_image_mobject_cpu(image_mobject, source_pixels)
        if full_image is not None:
            settings = self.target.settings
            width, height = settings.pixel_width, settings.pixel_height
            self.target.draw_image_quad(
                full_image.tobytes(),
                width,
                height,
                [(0.0, 0.0), (float(width), 0.0), (float(width), float(height)),
                 (0.0, float(height))],
                1.0,
                0,
            )
            return
        self.target.draw_image_quad(
            source_pixels.tobytes(),
            source_pixels.shape[1],
            source_pixels.shape[0],
            quad,
            1.0,
            _SAMPLING.get(image_mobject.resampling_algorithm, 2),
        )

    def _rasterize_image_mobject_cpu(self, image_mobject, source_pixels):
        """Perspective-warp an ImageMobject exactly like the Cairo renderer.

        Returns a full-frame RGBA PIL image, or None when the quad is degenerate.
        """
        from manim.utils.space_ops import cross2d

        sub_image = Image.fromarray(source_pixels, mode="RGBA")
        original_coords = np.array(
            [
                [0, 0],
                [sub_image.width, 0],
                [0, sub_image.height],
                [sub_image.width, sub_image.height],
            ]
        )
        target_coords = self._points_to_subpixel_coords(
            image_mobject,
            image_mobject.points,
        )
        int_target_coords = target_coords.astype(np.int64)
        settings = self.target.settings
        shift_vector = np.maximum(int_target_coords.min(axis=0), 0)
        upper_bounds = np.minimum(
            int_target_coords.max(axis=0),
            (settings.pixel_width, settings.pixel_height),
        )
        target_size = (
            int(upper_bounds[0] - shift_vector[0]),
            int(upper_bounds[1] - shift_vector[1]),
        )
        if min(target_size) <= 0:
            return None
        target_coords -= shift_vector

        ordered_vertices = [target_coords[index] for index in (0, 1, 3, 2)]
        sides = [
            ordered_vertices[(index + 1) % 4] - ordered_vertices[index]
            for index in range(4)
        ]
        side_lengths = np.linalg.norm(sides, axis=1)
        longest_index = int(np.argmax(side_lengths))
        longest_side = sides[longest_index]
        longest_length = side_lengths[longest_index]
        if longest_length == 0:
            return None
        previous_side = sides[(longest_index - 1) % 4]
        next_side = sides[(longest_index + 1) % 4]
        height_1 = abs(cross2d(longest_side, previous_side)) / longest_length
        height_2 = abs(cross2d(longest_side, next_side)) / longest_length
        if max(height_1, height_2) < 0.5:
            return None

        homography_matrix = []
        for (x, y), (target_x, target_y) in zip(
            target_coords,
            original_coords,
            strict=True,
        ):
            homography_matrix.append(
                [x, y, 1, 0, 0, 0, -target_x * x, -target_x * y],
            )
            homography_matrix.append(
                [0, 0, 0, x, y, 1, -target_y * x, -target_y * y],
            )
        matrix = np.array(homography_matrix, dtype=np.float64)
        target = original_coords.reshape(8).astype(np.float64)
        try:
            coefficients = np.linalg.solve(matrix, target)
        except np.linalg.LinAlgError:
            return None

        sub_image = sub_image.transform(
            size=target_size,
            method=Image.Transform.PERSPECTIVE,
            data=coefficients,
            resample=image_mobject.resampling_algorithm,
        )
        full_image = Image.new(
            "RGBA",
            (settings.pixel_width, settings.pixel_height),
            (0, 0, 0, 0),
        )
        full_image.paste(
            sub_image,
            box=(
                int(shift_vector[0]),
                int(shift_vector[1]),
                int(shift_vector[0] + target_size[0]),
                int(shift_vector[1] + target_size[1]),
            ),
        )
        return full_image
