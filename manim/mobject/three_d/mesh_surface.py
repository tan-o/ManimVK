"""Parametric surfaces drawn as one shaded, optionally textured, triangle mesh."""

from __future__ import annotations

__all__ = ["MeshSurface", "TexturedSurface"]

from functools import cache
from pathlib import PurePath
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from manim.mobject.mobject import Mobject
from manim.mobject.three_d.three_dimensions import Surface
from manim.utils.bezier import interpolate
from manim.utils.color import ManimColor, ParsableManimColor, interpolate_color
from manim.utils.images import get_full_raster_image_path, new_texture_key

if TYPE_CHECKING:
    from typing import Self


@cache
def _image(image_file: str) -> tuple[np.ndarray, int]:
    """An image file's RGBA pixels, read-only and kept for the process
    lifetime, and the key of its GPU texture.
    """
    pixels = np.ascontiguousarray(
        Image.open(get_full_raster_image_path(image_file)).convert("RGBA")
    )
    pixels.flags.writeable = False
    return pixels, new_texture_key()


class MeshSurface(Mobject):
    """A :class:`.Surface` drawn as one smoothly shaded triangle mesh.

    Where a :class:`.Surface` is a group of flat faces, each drawn on its own,
    this is one mesh lit per vertex like ManimGL's surfaces (with ManimGL's
    lighting), and drawn in one GPU call.  In a :class:`.ThreeDScene`, opaque
    mesh surfaces hide each other per pixel through a Vulkan depth buffer,
    and their triangles are ordered by depth together with other
    depth-tested mobjects (see :meth:`.Mobject.apply_depth_test`).  A
    translucent mesh surface hides depth-tested mobjects behind it too, as a
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
        faces = {
            (face.u_index, face.v_index): face for face in uv_surface.list_of_faces
        }
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

        u, v = np.meshgrid(
            np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1), indexing="ij"
        )
        self.im_coords = np.column_stack([u.ravel(), 1 - v.ravel()])
        index = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
        a, b = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
        c, d = index[1:, 1:].ravel(), index[:-1, 1:].ravel()
        self.triangles = np.concatenate(
            [np.column_stack([a, b, c]), np.column_stack([a, c, d])]
        )

    # -- colour and opacity ----------------------------------------------
    def get_opacity(self) -> float:
        return self.opacity

    def set_opacity(self, opacity: float, family: bool = True) -> Self:
        self.opacity = float(opacity)
        return self

    def set_color(
        self,
        color: ParsableManimColor,
        opacity: float | None = None,
        family: bool = True,
    ) -> Self:
        self.color = ManimColor.parse(color)
        if opacity is not None:
            self.opacity = float(opacity)
        return self

    def fade(self, darkness: float = 0.5, family: bool = True) -> Self:
        self.opacity *= 1 - darkness
        return self

    def interpolate_color(
        self, mobject1: Mobject, mobject2: Mobject, alpha: float
    ) -> Self:
        self.color = interpolate_color(mobject1.color, mobject2.color, alpha)
        self.opacity = float(interpolate(mobject1.opacity, mobject2.opacity, alpha))
        return self

    # -- drawing ---------------------------------------------------------
    def get_images(self) -> list[tuple[np.ndarray, int]]:
        """The read-only RGBA images drawn on the surface (none, the day
        image, or the day and night images) with their texture keys.
        """
        return []

    def get_depth_pieces(self) -> np.ndarray:
        """The corners of each triangle, shape ``(n, 3, 3)``."""
        return self.points[self.triangles]

    def get_vertex_normals(self) -> np.ndarray:
        """Unit normals at the vertices, pointing along ``du x dv``.

        A vertex takes the area-weighted mean normal of the triangles around
        it, counting every vertex at the same position (the seam where ``u``
        wraps, the poles of a sphere) as one, so that the lighting is smooth
        across seams and degenerate triangles leave no vertex without a
        normal.
        """
        if len(self.points) == 0:
            return np.zeros((0, 3))
        corners = self.points[self.triangles]
        face_normals = np.cross(
            corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
        )
        scale = max(float(np.ptp(self.points)), 1e-12)
        _, welded = np.unique(
            np.round(self.points / scale, 9), axis=0, return_inverse=True
        )
        welded = welded.ravel()
        sums = np.zeros((welded.max() + 1, 3))
        np.add.at(
            sums, welded[self.triangles.ravel()], np.repeat(face_normals, 3, axis=0)
        )
        normals = sums[welded]
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        return normals / np.where(lengths == 0, 1, lengths)


class TexturedSurface(MeshSurface):
    """A :class:`.Surface` with an image stretched over it.

    The image is mapped by the surface's ``(u, v)`` parameters: ``u`` runs
    left to right across the image and ``v`` bottom to top.  With
    ``dark_image_file``, the side facing away from the camera's light source
    shows that image instead, blended across the terminator, like the night
    side of the Earth.  As in ManimGL, the images are mixed and lit per
    pixel.  Drawn like a :class:`MeshSurface`.

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
                earth = TexturedSurface(
                    Sphere(radius=3), "earth_day.jpg", "earth_night.jpg"
                )
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

    def get_images(self) -> list[tuple[np.ndarray, int]]:
        files = [self.image_file, self.dark_image_file]
        return [_image(file) for file in files if file is not None]
