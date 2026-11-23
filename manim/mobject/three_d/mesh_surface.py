"""Parametric surfaces drawn as one shaded, optionally textured, triangle mesh."""

from __future__ import annotations

__all__ = ["MeshSurface", "TexturedSurface"]

from functools import lru_cache
from pathlib import PurePath
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from manim.mobject.mobject import Mobject
from manim.mobject.three_d.three_dimensions import Surface
from manim.utils.bezier import interpolate
from manim.utils.color import ManimColor, ParsableManimColor, interpolate_color
from manim.utils.images import get_full_raster_image_path

if TYPE_CHECKING:
    from typing_extensions import Self

    from manim.camera import Camera

# How far either side of facing the light the day and night images blend.
_DARK_SHIFT = 0.2
# Rows between atlas regions, filled with the neighbouring edge rows so that
# mipmapped sampling near one region's edge does not pick up the next region.
_GUTTER = 32
# Height of the white region that, tinted by vertex colours, gives solid
# colours, highlights and shadows.
_SOLID = 64


@lru_cache(maxsize=None)
def _texture_atlas(
    image_file: str | None, dark_image_file: str | None
) -> tuple[np.ndarray, dict[str, float | None]]:
    """Stack the day image, the night image and a white region in one texture.

    Skia's ``drawVertices`` takes one shader, so one texture holds every
    layer a triangle is drawn with; vertex colours tint and fade each layer.
    The array is kept for the process lifetime and is read-only, so its
    ``id`` keys the GPU upload.  With no image this is the shared white
    texture of every :class:`MeshSurface`.
    """
    regions: list[np.ndarray] = []
    if image_file is not None:
        regions.append(np.asarray(Image.open(get_full_raster_image_path(image_file)).convert("RGBA")))
        height, width = regions[0].shape[:2]
        if dark_image_file is not None:
            night = Image.open(get_full_raster_image_path(dark_image_file)).convert("RGBA")
            regions.append(np.asarray(night.resize((width, height), Image.Resampling.LANCZOS)))
    else:
        width, height = _SOLID, _SOLID
    regions.append(np.full((_SOLID, width, 4), 255, dtype=np.uint8))

    rows, offsets = [], []
    top = 0
    for index, region in enumerate(regions):
        if index:
            rows.append(np.repeat(regions[index - 1][-1:], _GUTTER // 2, axis=0))
            rows.append(np.repeat(region[:1], _GUTTER // 2, axis=0))
            top += _GUTTER
        offsets.append(top)
        rows.append(region)
        top += len(region)
    atlas = np.ascontiguousarray(np.concatenate(rows))
    atlas.flags.writeable = False
    layout: dict[str, float | None] = {
        "width": width,
        "height": height,
        "day": offsets[0] if image_file is not None else None,
        "night": offsets[1] if dark_image_file is not None else None,
        "white": offsets[-1] + _SOLID / 2,
    }
    return atlas, layout


class MeshSurface(Mobject):
    """A :class:`.Surface` drawn as one smoothly shaded triangle mesh.

    Where a :class:`.Surface` is a group of flat faces, each drawn on its own,
    this is one mesh lit per vertex like ManimGL surfaces, and drawn in one GPU
    call.  In a :class:`.ThreeDScene` its triangles are ordered by depth
    together with other depth-tested mobjects (see
    :meth:`.Mobject.apply_depth_test`), so that it hides what is behind it.
    A translucent mesh surface hides depth-tested mobjects behind it too, as a
    depth buffer would, and shows the camera background through itself.
    Drawn by the Vulkan renderer only.

    Parameters
    ----------
    uv_surface
        The surface to draw; its faces give the mesh, including any
        transformation applied to it.
    color
        The colour; defaults to the colour of the surface's first face.
    opacity
        The opacity; defaults to that of the surface's first face.
    shading
        ``(reflectiveness, gloss, shadow)``: how much facing the camera's light
        source brightens the surface, the strength of the highlight reflected
        towards the camera, and how much facing away darkens it.

    Examples
    --------
    .. code-block:: python

        class SmoothSphere(ThreeDScene):
            def construct(self):
                self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES)
                self.add(MeshSurface(Sphere(radius=2), color=BLUE_E))
    """

    def __init__(
        self,
        uv_surface: Surface,
        color: ParsableManimColor | None = None,
        opacity: float | None = None,
        shading: tuple[float, float, float] = (0.3, 0.2, 0.4),
        **kwargs: Any,
    ) -> None:
        if not isinstance(uv_surface, Surface):
            raise TypeError("uv_surface must be a Surface")
        faces = {(face.u_index, face.v_index): face for face in uv_surface.list_of_faces}
        nu = 1 + max(i for i, _ in faces)
        nv = 1 + max(j for _, j in faces)
        grid = np.zeros((nu + 1, nv + 1, 3))
        for (i, j), face in faces.items():
            # Faces are the corners (u1, v1), (u2, v1), (u2, v2), (u1, v2) as
            # straight cubic segments: anchors at points 0, 4, 8 and 12.
            corners = face.points[[0, 4, 8, 12]]
            grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1] = corners
        first_face = faces[(0, 0)]
        if color is None:
            color = first_face.get_fill_color()
        if opacity is None:
            opacity = first_face.get_fill_opacity()

        super().__init__(color=color, **kwargs)
        self.opacity = float(opacity)
        self.shading = tuple(float(value) for value in shading)
        self.resolution = (nu, nv)
        self.depth_test = True
        self.points = grid.reshape(-1, 3)

        u, v = np.meshgrid(np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1), indexing="ij")
        self.im_coords = np.column_stack([u.ravel(), 1 - v.ravel()])
        index = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
        a, b = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
        c, d = index[1:, 1:].ravel(), index[:-1, 1:].ravel()
        self.triangles = np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])

    # -- colour and opacity ----------------------------------------------
    def get_opacity(self) -> float:
        return self.opacity

    def set_opacity(self, opacity: float, family: bool = True) -> Self:
        self.opacity = float(opacity)
        return self

    def set_color(
        self, color: ParsableManimColor, opacity: float | None = None, family: bool = True
    ) -> Self:
        self.color = ManimColor.parse(color)
        if opacity is not None:
            self.opacity = float(opacity)
        return self

    def fade(self, darkness: float = 0.5, family: bool = True) -> Self:
        self.opacity *= 1 - darkness
        return self

    def interpolate_color(self, mobject1: Mobject, mobject2: Mobject, alpha: float) -> Self:
        self.color = interpolate_color(mobject1.color, mobject2.color, alpha)
        self.opacity = float(interpolate(mobject1.opacity, mobject2.opacity, alpha))
        return self

    # -- drawing ---------------------------------------------------------
    def _atlas_files(self) -> tuple[str | None, str | None]:
        return None, None

    def get_texture_atlas(self) -> np.ndarray:
        return _texture_atlas(*self._atlas_files())[0]

    def get_depth_pieces(self) -> np.ndarray:
        """The corners of each triangle, shape ``(n, 3, 3)``."""
        return self.points[self.triangles]

    def _vertex_normals(self) -> np.ndarray:
        corners = self.points[self.triangles]
        face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        normals = np.zeros_like(self.points)
        for k in range(3):
            np.add.at(normals, self.triangles[:, k], face_normals)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        return normals / np.where(lengths == 0, 1, lengths)

    def _base_layers(
        self, layout: dict, facing: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """(uv, rgba) per vertex for the layers under the lighting layers."""
        count = len(self.points)
        white = np.tile([layout["width"] / 2, layout["white"]], (count, 1))
        rgba = np.tile([*self.color.to_rgb(), 1.0], (count, 1))
        return [(white, rgba)]

    def get_shaded_mesh(
        self, camera: Camera, pieces: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Vertices, texture coordinates and colours of the triangles to draw.

        Each triangle is repeated once per layer: the base colour or image(s),
        then a white highlight and a black shadow.  Every layer is a texel
        times its vertex colour blended over the layers before it, which
        evaluates ManimGL's surface lighting at the vertices.
        """
        triangles = self.triangles if pieces is None else self.triangles[pieces]
        if len(triangles) == 0 or len(self.points) == 0:
            return None
        _, layout = _texture_atlas(*self._atlas_files())
        points = self.points
        normals = self._vertex_normals()
        to_light = camera.light_source.get_center() - points
        to_light /= np.maximum(np.linalg.norm(to_light, axis=1, keepdims=True), 1e-12)
        facing = np.einsum("ij,ij->i", normals, to_light)

        layers = self._base_layers(layout, facing)
        white = np.tile([layout["width"] / 2, layout["white"]], (len(points), 1))
        reflectiveness, gloss, shadow = self.shading
        if reflectiveness or gloss:
            bright = np.maximum(facing, 0) * reflectiveness
            if gloss:
                to_camera = _camera_position(camera) - points
                to_camera /= np.maximum(np.linalg.norm(to_camera, axis=1, keepdims=True), 1e-12)
                reflection = 2 * facing[:, None] * normals - to_light
                light_to_camera = np.einsum("ij,ij->i", reflection, to_camera)
                bright = bright + gloss * np.exp(-3 * (1 - light_to_camera) ** 2)
            highlight = np.ones((len(points), 4))
            highlight[:, 3] = np.clip(bright, 0, 1)
            layers.append((white, highlight))
        if shadow:
            darkening = np.zeros((len(points), 4))
            darkening[:, 3] = np.clip(np.maximum(-facing, 0) * shadow, 0, 1)
            layers.append((white, darkening))

        count = len(layers)
        positions = np.repeat(points[triangles][:, None], count, axis=1)
        uvs = np.stack([uv[triangles] for uv, _ in layers], axis=1)
        colors = np.stack([rgba[triangles] for _, rgba in layers], axis=1)
        return positions.reshape(-1, 3), uvs.reshape(-1, 2), colors.reshape(-1, 4)


class TexturedSurface(MeshSurface):
    """A :class:`.Surface` with an image stretched over it.

    The image is mapped by the surface's ``(u, v)`` parameters: ``u`` runs
    left to right across the image and ``v`` bottom to top.  With
    ``dark_image_file``, the side facing away from the camera's light source
    shows that image instead, blended across the terminator, like the night
    side of the Earth.  Drawn like a :class:`MeshSurface`.

    Parameters
    ----------
    uv_surface
        The surface to cover.
    image_file
        The image, found like :class:`.ImageMobject` files.
    dark_image_file
        Optional image for the side facing away from the light.
    opacity
        The opacity of the whole surface.
    shading
        See :class:`MeshSurface`.

    Examples
    --------
    .. code-block:: python

        class Earth(ThreeDScene):
            def construct(self):
                self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES)
                earth = TexturedSurface(Sphere(radius=3), "earth_day.jpg", "earth_night.jpg")
                self.add(earth)
                self.play(Rotate(earth, TAU, axis=OUT), run_time=8)
    """

    def __init__(
        self,
        uv_surface: Surface,
        image_file: str | PurePath,
        dark_image_file: str | PurePath | None = None,
        opacity: float = 1.0,
        shading: tuple[float, float, float] = (0.3, 0.2, 0.4),
        **kwargs: Any,
    ) -> None:
        self.image_file = str(image_file)
        self.dark_image_file = None if dark_image_file is None else str(dark_image_file)
        super().__init__(uv_surface, opacity=opacity, shading=shading, **kwargs)

    def _atlas_files(self) -> tuple[str | None, str | None]:
        return self.image_file, self.dark_image_file

    def _base_layers(
        self, layout: dict, facing: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        width, height = layout["width"], layout["height"]
        image_x = self.im_coords[:, 0] * (width - 1) + 0.5
        image_y = self.im_coords[:, 1] * (height - 1) + 0.5
        count = len(self.points)
        layers = []
        if layout["night"] is not None:
            layers.append((np.column_stack([image_x, image_y + layout["night"]]), np.ones((count, 4))))
            day = np.clip((facing + _DARK_SHIFT) / (2 * _DARK_SHIFT), 0, 1)
            day_rgba = np.ones((count, 4))
            day_rgba[:, 3] = day * day * (3 - 2 * day)
        else:
            day_rgba = np.ones((count, 4))
        layers.append((np.column_stack([image_x, image_y + layout["day"]]), day_rgba))
        return layers


def _camera_position(camera: Camera) -> np.ndarray:
    center = np.asarray(camera.frame_center, dtype=float)
    rotation = getattr(camera, "get_rotation_matrix", None)
    if rotation is None:
        return center + np.array([0.0, 0.0, 20.0])
    return center + camera.get_focal_distance() * rotation()[2]
