"""Camera views and projection controls."""

from __future__ import annotations

__all__ = ["Camera", "DrawOrder", "MovingCamera", "MultiCamera", "ThreeDCamera"]

import contextlib
import operator as op
from collections.abc import Callable, Iterable, Iterator
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Self, overload

import manim_core
import numpy as np

from manim._config import config
from manim.animation.animation import Animation
from manim.constants import DEGREES, DOWN, LEFT, ORIGIN, OUT
from manim.mobject.frame import ScreenRectangle
from manim.mobject.mobject import Mobject, _AnimationBuilder
from manim.mobject.three_d.three_d_utils import (
    get_3d_vmob_end_corner,
    get_3d_vmob_end_corner_unit_normal,
    get_3d_vmob_start_corner,
    get_3d_vmob_start_corner_unit_normal,
)
from manim.mobject.types.image_mobject import ImageMobjectFromCamera
from manim.mobject.types.point_cloud_mobject import Point
from manim.mobject.types.vectorized_mobject import VMobject
from manim.mobject.value_tracker import ValueTracker
from manim.utils.color import WHITE, ManimColor, ParsableManimColor, get_shaded_rgb
from manim.utils.deprecation import deprecated
from manim.utils.family import extract_mobject_family_members
from manim.utils.iterables import list_difference_update
from manim.utils.space_ops import rotation_matrix_zxz

if TYPE_CHECKING:
    from manim.typing import (
        FloatRGBA_Array,
        MatrixMN,
        Point3D,
        Point3D_Array,
        Point3DLike,
    )


class DrawOrder(NamedTuple):
    """Mobjects in drawing order, as runs held in arrays.

    Run ``k`` draws ``mobjects[owner[k]]``: whole when ``start[k] == end[k]``,
    otherwise its depth pieces ``parts[start[k]:end[k]]`` (indices into
    :meth:`.Mobject.get_depth_pieces`).  ``key[k]`` is the depth key
    ``1 / (focal_distance - z)`` of the nearest of them (larger is nearer),
    NaN for a run that is not drawn in depth.  Arrays, not one Python object
    per run: a depth-sorted field of arrows has tens of thousands of runs.
    """

    mobjects: list[Mobject]
    owner: np.ndarray
    start: np.ndarray
    end: np.ndarray
    parts: np.ndarray
    key: np.ndarray

    @classmethod
    def whole(cls, mobjects: list[Mobject]) -> DrawOrder:
        """Every mobject drawn whole, in the given order."""
        zeros = np.zeros(len(mobjects), dtype=np.int64)
        return cls(
            mobjects,
            np.arange(len(mobjects), dtype=np.int64),
            zeros,
            zeros,
            np.zeros(0, dtype=np.int64),
            np.full(len(mobjects), np.nan),
        )

    def runs(self) -> Iterator[tuple[Mobject, np.ndarray | None]]:
        """``(mobject, pieces or None)`` per run, for drawing a few runs one by one."""
        for owner, start, end in zip(
            self.owner.tolist(), self.start.tolist(), self.end.tolist(), strict=True
        ):
            yield self.mobjects[owner], (self.parts[start:end] if end > start else None)


class Camera:
    """Configure the camera view and background.

    A camera has an animatable frame, background settings, and methods for ordering
    and projecting mobjects. The renderer is responsible for drawing this view
    into an image.

    By default, the frame width is ``config.frame_width`` and its height is derived
    from the viewport's pixel aspect ratio. Supplying one frame dimension derives
    the other from that ratio; supplying both dimensions or a custom ``frame`` uses
    the dimensions you specify. Frame dimensions are measured in Manim units.
    """

    def __init__(
        self,
        background_image: str | None = None,
        frame_center: Point3DLike | Mobject | None = None,
        frame: Mobject | None = None,
        default_frame_stroke_color: ManimColor = WHITE,
        default_frame_stroke_width: float = 0,
        use_z_index: bool = True,
        frame_height: float | None = None,
        frame_width: float | None = None,
        background_color: ParsableManimColor | None = None,
        background_opacity: float | None = None,
    ) -> None:
        self.background_image = background_image
        self.use_z_index = use_z_index

        if frame is None:
            if frame_width is None:
                resolved_width = (
                    config.frame_width
                    if frame_height is None
                    else frame_height * config.aspect_ratio
                )
            else:
                resolved_width = frame_width
            resolved_height = (
                resolved_width / config.aspect_ratio
                if frame_height is None
                else frame_height
            )
            if resolved_height <= 0 or resolved_width <= 0:
                raise ValueError("Camera frame dimensions must be positive.")
            frame = ScreenRectangle(
                aspect_ratio=resolved_width / resolved_height,
                height=resolved_height,
            )
            frame.set_stroke(
                ManimColor(default_frame_stroke_color),
                default_frame_stroke_width,
            )
        else:
            if frame_height is not None:
                if frame_height <= 0:
                    raise ValueError("Camera frame height must be positive.")
                frame.stretch_to_fit_height(frame_height)
            if frame_width is not None:
                if frame_width <= 0:
                    raise ValueError("Camera frame width must be positive.")
                frame.stretch_to_fit_width(frame_width)
        self.frame = frame
        if frame_center is not None:
            self.frame_center = frame_center

        self._background_color = ManimColor(
            background_color
            if background_color is not None
            else config.background_color
        )
        self._background_opacity = (
            background_opacity
            if background_opacity is not None
            else config.background_opacity
        )

    @property
    def background_color(self) -> ManimColor:
        return self._background_color

    @background_color.setter
    def background_color(self, color: ParsableManimColor) -> None:
        self._background_color = ManimColor(color)

    @property
    def background_opacity(self) -> float:
        return self._background_opacity

    @background_opacity.setter
    def background_opacity(self, alpha: float) -> None:
        self._background_opacity = alpha

    @property
    def frame_height(self) -> float:
        """Height of the camera frame in Manim units."""
        return self.frame.height

    @frame_height.setter
    def frame_height(self, frame_height: float) -> None:
        self.frame.stretch_to_fit_height(frame_height)

    @property
    def frame_width(self) -> float:
        """Width of the camera frame in Manim units."""
        return self.frame.width

    @frame_width.setter
    def frame_width(self, frame_width: float) -> None:
        self.frame.stretch_to_fit_width(frame_width)

    @property
    def frame_center(self) -> Point3D:
        """Center of the camera frame in scene coordinates."""
        return self.frame.get_center()

    @frame_center.setter
    def frame_center(self, frame_center: Point3DLike | Mobject) -> None:
        self.frame.move_to(frame_center)

    def get_width(self) -> float:
        """Compatibility alias for the camera frame width."""
        return self.frame.width

    def get_height(self) -> float:
        """Compatibility alias for the camera frame height."""
        return self.frame.height

    def get_center(self) -> Point3D:
        """Return the camera frame center (OpenGL-camera-compatible API)."""
        return self.frame.get_center()

    def shift(self, delta: Point3DLike) -> Self:
        """Shift the camera frame, matching the OpenGL camera API."""
        self.frame.shift(delta)
        return self

    def move_to(self, point: Point3DLike | Mobject) -> Self:
        """Move the camera frame center to ``point``."""
        self.frame.move_to(point)
        return self

    def scale(self, factor: float, **kwargs: Any) -> Self:
        """Scale the camera frame around its center or ``about_point``."""
        self.frame.scale(factor, **kwargs)
        return self

    def get_mobjects_to_display(
        self,
        mobjects: Iterable[Mobject],
        include_submobjects: bool = True,
        excluded_mobjects: list[Mobject] | None = None,
    ) -> list[Mobject]:
        """Return the mobjects and included submobjects in drawing order."""
        if include_submobjects:
            mobjects = extract_mobject_family_members(
                mobjects,
                use_z_index=self.use_z_index,
                only_those_with_points=True,
            )
            if excluded_mobjects:
                all_excluded = extract_mobject_family_members(
                    excluded_mobjects,
                    use_z_index=self.use_z_index,
                )
                mobjects = list_difference_update(mobjects, all_excluded)
        return list(mobjects)

    def get_draw_order(
        self,
        mobjects: list[Mobject],
        roots: dict[int, int] | None = None,
    ) -> DrawOrder:
        """The runs that draw the mobjects from :meth:`get_mobjects_to_display`.

        A flat camera draws every mobject whole, in display order.
        """
        return DrawOrder.whole(mobjects)

    def is_in_frame(self, mobject: Mobject) -> bool:
        """Whether ``mobject`` intersects the camera frame's bounds."""
        center = self.frame_center
        height = self.frame_height
        width = self.frame_width
        return not reduce(
            op.or_,
            [
                mobject.get_right()[0] < center[0] - width / 2,
                mobject.get_bottom()[1] > center[1] + height / 2,
                mobject.get_left()[0] > center[0] + width / 2,
                mobject.get_top()[1] < center[1] - height / 2,
            ],
        )

    def get_mobjects_indicating_movement(self) -> list[Mobject]:
        """Return mobjects whose animation requires the scene to be redrawn.

        The scene uses these controls to decide whether it can reuse an image of
        its stationary mobjects during an animation.
        """
        return [self.frame]

    @overload
    def auto_zoom(
        self,
        mobjects: Iterable[Mobject],
        margin: float = 0,
        only_mobjects_in_frame: bool = False,
        animate: Literal[False] = False,
    ) -> Mobject: ...

    @overload
    def auto_zoom(
        self,
        mobjects: Iterable[Mobject],
        margin: float = 0,
        only_mobjects_in_frame: bool = False,
        animate: Literal[True] = True,
    ) -> _AnimationBuilder: ...

    def auto_zoom(
        self,
        mobjects: Iterable[Mobject],
        margin: float = 0,
        only_mobjects_in_frame: bool = False,
        animate: bool = True,
    ) -> _AnimationBuilder | Mobject:
        """Move and resize the frame to contain the supplied 2D mobjects.

        The frame keeps its aspect ratio and fits the mobjects' bounding box in
        the direction that needs the larger frame; ``margin`` is added to that
        width or height (in total, not on each side).

        Raises
        ------
        ValueError
            If the fitted width or height plus ``margin`` is not positive, for
            example for mobjects without size and no positive margin.
        """
        left, right, top, bottom = self._get_bounding_box(
            mobjects, only_mobjects_in_frame
        )
        x = (left + right) / 2
        y = (top + bottom) / 2
        # Fit the side that needs the larger frame; compare new_width / width
        # with new_height / height without dividing.
        if (right - left) * self.frame.height > (top - bottom) * self.frame.width:
            side, size = "width", right - left + margin
        else:
            side, size = "height", top - bottom + margin
        if size <= 0:
            raise ValueError(
                f"Camera.auto_zoom() would give the frame a {side} of {size:g}; "
                "use a larger margin."
            )
        target = self.frame.animate if animate else self.frame
        return target.set_x(x).set_y(y).set(**{side: size})

    def _get_bounding_box(
        self,
        mobjects: Iterable[Mobject],
        only_mobjects_in_frame: bool,
    ) -> tuple[float, float, float, float]:
        bounds: tuple[float, float, float, float] | None = None
        for mobject in mobjects:
            if mobject is self.frame or (
                only_mobjects_in_frame and not self.is_in_frame(mobject)
            ):
                continue
            mobject_bounds = (
                float(mobject.get_left()[0]),
                float(mobject.get_right()[0]),
                float(mobject.get_top()[1]),
                float(mobject.get_bottom()[1]),
            )
            if bounds is None:
                bounds = mobject_bounds
            else:
                bounds = (
                    min(bounds[0], mobject_bounds[0]),
                    max(bounds[1], mobject_bounds[1]),
                    max(bounds[2], mobject_bounds[2]),
                    min(bounds[3], mobject_bounds[3]),
                )
        if bounds is None:
            raise ValueError(
                "Could not determine the bounding box of the mobjects given to "
                "Camera.auto_zoom().",
            )
        return bounds

    @contextlib.contextmanager
    def _drawing(self) -> Iterator[None]:
        """Scope in which one frame is drawn; the camera does not change in it."""
        yield

    def get_view_transform_center(self) -> Point3D:
        """Return the center applied by the renderer's 2D view transform."""
        return self.frame_center

    def get_stroke_rgbas(
        self,
        vmobject: VMobject,
        background: bool = False,
    ) -> FloatRGBA_Array:
        """Return stroke colors after camera-specific shading."""
        return vmobject.get_stroke_rgbas(background)

    def get_fill_rgbas(self, vmobject: VMobject) -> FloatRGBA_Array:
        """Return fill colors after camera-specific shading."""
        return vmobject.get_fill_rgbas()

    def transform_points_pre_display(
        self,
        mobject: Mobject,
        points: Point3D_Array,
    ) -> Point3D_Array:
        """Project the mobject's points for display."""
        if not np.all(np.isfinite(points)):
            return np.zeros((1, 3))
        if getattr(self, "_euler_angles", None) is not None:
            return self.project_points(points)
        return points

    def generate_rotation_matrix(self) -> MatrixMN:
        phi = self.get_phi()
        theta = self.get_theta()
        gamma = self.get_gamma()
        return rotation_matrix_zxz(gamma, -phi, -theta - 90 * DEGREES)

    def project_points(self, points: Point3D_Array) -> Point3D_Array:
        """Apply the stored Euler orientation as a simple perspective view."""
        frame_center = (
            self.frame_center
            if isinstance(self.frame_center, np.ndarray)
            else np.array(self.frame_center)
        )
        focal_distance = 20.0
        zoom = 1.0
        rot_matrix = self.generate_rotation_matrix()
        points = points - frame_center
        points = np.dot(points, rot_matrix.T)
        zs = points[:, 2]
        factor = focal_distance / (focal_distance - zs)
        factor[(focal_distance - zs) < 0] = 10**6
        points[:, :2] *= (factor * zoom)[:, None]
        return points

    def set_euler_angles(self, theta=None, phi=None, gamma=None, **kwargs: Any) -> Self:
        """Accept the classic OpenGL camera's Euler-angle API.

        A plain ``Scene`` has no 3D perspective camera by default.  Store the
        requested orientation so old scenes can construct their objects; a
        ``ThreeDScene`` still uses the full perspective camera.
        """
        self._euler_angles = (theta, phi, gamma)
        return self

    def get_theta(self) -> float:
        angles = getattr(self, "_euler_angles", (None, None, None))
        return 0.0 if angles[0] is None else float(angles[0])

    def get_phi(self) -> float:
        angles = getattr(self, "_euler_angles", (None, None, None))
        return 0.0 if angles[1] is None else float(angles[1])

    def get_gamma(self) -> float:
        angles = getattr(self, "_euler_angles", (None, None, None))
        return 0.0 if angles[2] is None else float(angles[2])

    @property
    def animate(self) -> _CameraAnimate:
        return _CameraAnimate(self)

    @property
    def light_source(self) -> Point:
        if not hasattr(self, "_light_source"):
            self._light_source = Point(9 * DOWN + 7 * LEFT + 10 * OUT)
        return self._light_source

    @light_source.setter
    def light_source(self, value: Point) -> None:
        self._light_source = value

    def add_updater(self, update_function: Any, **kwargs: Any) -> Self:
        """Accept the classic OpenGL camera's updater API.

        A plain classic camera is not part of ``Scene.mobjects``; keep the
        callable for callers that inspect it, but do not invent a second update
        loop outside the scene manager.
        """
        self._camera_updaters = [
            *getattr(self, "_camera_updaters", []),
            update_function,
        ]
        return self

    def increment_theta(self, delta: float) -> Self:
        self.set_euler_angles(
            self.get_theta() + delta, self.get_phi(), self.get_gamma()
        )
        return self

    def increment_phi(self, delta: float) -> Self:
        self.set_euler_angles(
            self.get_theta(), self.get_phi() + delta, self.get_gamma()
        )
        return self

    def increment_gamma(self, delta: float) -> Self:
        self.set_euler_angles(
            self.get_theta(), self.get_phi(), self.get_gamma() + delta
        )
        return self


@deprecated(
    replacement="Camera",
    message="Every camera has an animatable frame and auto_zoom().",
)
class MovingCamera(Camera):
    """A :class:`Camera`: every camera has a movable frame now."""


class MultiCamera(Camera):
    """Describe a primary view with camera-backed image mobjects."""

    def __init__(
        self,
        image_mobjects_from_cameras: Iterable[ImageMobjectFromCamera] | None = None,
        **kwargs: Any,
    ) -> None:
        self.image_mobjects_from_cameras: list[ImageMobjectFromCamera] = []
        if image_mobjects_from_cameras is not None:
            for image_mobject in image_mobjects_from_cameras:
                self.add_image_mobject_from_camera(image_mobject)
        super().__init__(**kwargs)

    def add_image_mobject_from_camera(
        self,
        image_mobject_from_camera: ImageMobjectFromCamera,
    ) -> None:
        """Register a display whose image is rendered from another camera."""
        if not isinstance(image_mobject_from_camera.camera, Camera):
            raise TypeError("A nested view needs a Camera.")
        self.image_mobjects_from_cameras.append(image_mobject_from_camera)

    def get_mobjects_indicating_movement(self) -> list[Mobject]:
        """Return camera controls for the primary and nested views."""

        def collect(camera: Camera, visited: set[int]) -> list[Mobject]:
            if id(camera) in visited:
                return []
            visited.add(id(camera))
            if not isinstance(camera, MultiCamera):
                return camera.get_mobjects_indicating_movement()

            indicators = Camera.get_mobjects_indicating_movement(camera)
            for image_mobject in camera.image_mobjects_from_cameras:
                indicators.extend(collect(image_mobject.camera, visited))
            return indicators

        return collect(self, set())


class ThreeDCamera(Camera):
    def __init__(
        self,
        focal_distance: float = 20.0,
        shading_factor: float = 0.2,
        default_distance: float = 5.0,
        light_source_start_point: Point3DLike = 9 * DOWN + 7 * LEFT + 10 * OUT,
        should_apply_shading: bool = True,
        exponential_projection: bool = False,
        phi: float = 0,
        theta: float = -90 * DEGREES,
        gamma: float = 0,
        zoom: float = 1,
        **kwargs: Any,
    ):
        """Initializes the ThreeDCamera

        Parameters
        ----------
        *kwargs
            Any keyword argument of Camera.
        """
        super().__init__(**kwargs)
        self.shading_factor = shading_factor
        self.default_distance = default_distance
        self.light_source_start_point = light_source_start_point
        self.light_source = Point(self.light_source_start_point)
        self.should_apply_shading = should_apply_shading
        self.exponential_projection = exponential_projection
        self.phi_tracker = ValueTracker(phi)
        self.theta_tracker = ValueTracker(theta)
        self.focal_distance_tracker = ValueTracker(focal_distance)
        self.gamma_tracker = ValueTracker(gamma)
        self.zoom_tracker = ValueTracker(zoom)
        self.fixed_orientation_mobjects: dict[Mobject, Callable[[], Point3D]] = {}
        self.fixed_in_frame_mobjects: set[Mobject] = set()
        self._view: tuple[Point3D, float, float, MatrixMN] | None = None

    @contextlib.contextmanager
    def _drawing(self) -> Iterator[None]:
        # Every mobject of a frame is projected with the same view: compute it
        # once instead of in each projection (the frame center alone is a
        # bounding box of the frame mobject).
        self._view = self._current_view()
        try:
            yield
        finally:
            self._view = None

    def _current_view(self) -> tuple[Point3D, float, float, MatrixMN]:
        return (
            self.frame_center,
            self.get_focal_distance(),
            self.get_zoom(),
            self.get_rotation_matrix(),
        )

    # Read-only: the trackers hold the orientation. Change it with set_phi() and
    # the like, set_camera_orientation(), or by animating the trackers.
    @property
    def phi(self) -> float:
        """The polar angle (the angle off the z axis), in radians."""
        return self.get_phi()

    @property
    def theta(self) -> float:
        """The azimuthal angle, in radians."""
        return self.get_theta()

    @property
    def gamma(self) -> float:
        """The rotation about the viewing axis, in radians."""
        return self.get_gamma()

    @property
    def focal_distance(self) -> float:
        """The distance of the camera from the center of the frame."""
        return self.get_focal_distance()

    @property
    def zoom(self) -> float:
        """The zoom factor."""
        return self.get_zoom()

    def get_view_transform_center(self) -> Point3D:
        # project_points() already translates by frame_center.
        return ORIGIN.copy()

    def get_mobjects_indicating_movement(self) -> list[Mobject]:
        return [self.frame, self.light_source, *self.get_value_trackers()]

    def get_value_trackers(self) -> list[ValueTracker]:
        """A list of :class:`ValueTrackers <.ValueTracker>` of phi, theta, focal_distance,
        gamma and zoom.

        Returns
        -------
        list
            list of ValueTracker objects
        """
        return [
            self.phi_tracker,
            self.theta_tracker,
            self.focal_distance_tracker,
            self.gamma_tracker,
            self.zoom_tracker,
        ]

    def modified_rgbas(
        self, vmobject: VMobject, rgbas: FloatRGBA_Array
    ) -> FloatRGBA_Array:
        if not self.should_apply_shading:
            return rgbas
        if vmobject.shade_in_3d and (vmobject.get_num_points() > 0):
            light_source_point = self.light_source.points[0]
            if len(rgbas) < 2:
                shaded_rgbas = rgbas.repeat(2, axis=0)
            else:
                shaded_rgbas = np.array(rgbas[:2])
            shaded_rgbas[0, :3] = get_shaded_rgb(
                shaded_rgbas[0, :3],
                get_3d_vmob_start_corner(vmobject),
                get_3d_vmob_start_corner_unit_normal(vmobject),
                light_source_point,
            )
            shaded_rgbas[1, :3] = get_shaded_rgb(
                shaded_rgbas[1, :3],
                get_3d_vmob_end_corner(vmobject),
                get_3d_vmob_end_corner_unit_normal(vmobject),
                light_source_point,
            )
            return shaded_rgbas
        return rgbas

    def get_stroke_rgbas(
        self,
        vmobject: VMobject,
        background: bool = False,
    ) -> FloatRGBA_Array:  # NOTE : DocStrings From parent
        return self.modified_rgbas(vmobject, vmobject.get_stroke_rgbas(background))

    def get_fill_rgbas(
        self, vmobject: VMobject
    ) -> FloatRGBA_Array:  # NOTE : DocStrings From parent
        return self.modified_rgbas(vmobject, vmobject.get_fill_rgbas())

    def get_mobjects_to_display(
        self, *args: Any, **kwargs: Any
    ) -> list[Mobject]:  # NOTE : DocStrings From parent
        mobjects = super().get_mobjects_to_display(*args, **kwargs)
        rot_matrix = self.get_rotation_matrix()

        def z_key(mob: Mobject) -> float:
            if not (mob.depth_test or getattr(mob, "shade_in_3d", False)):
                return np.inf  # type: ignore[no-any-return]
            if mob.depth_test and isinstance(mob, VMobject):
                # get_draw_order orders a depth-tested VMobject by its
                # pieces; sorting it here would cost a pass over its many
                # points for nothing.  Such mobjects keep their family order.
                return 0.0
            if mob.depth_test and len(mob.points):
                return float(np.dot(mob.points.mean(axis=0), rot_matrix[2]))
            # Assign a number to a three dimensional mobjects
            # based on how close it is to the camera
            distance: float = np.dot(mob.get_z_index_reference_point(), rot_matrix.T)[2]
            return distance

        return sorted(mobjects, key=z_key)

    #: Number of depth slices used when depth pieces are present; see
    #: :meth:`get_draw_order`.  A run is tested against mesh depth with the
    #: key of its nearest piece, so a piece at most one slice behind a mesh
    #: surface may show in front of it.
    depth_slices = 128

    def get_draw_order(
        self,
        mobjects: list[Mobject],
        roots: dict[int, int] | None = None,
    ) -> DrawOrder:
        """Order depth pieces and 3D mobjects back to front.

        Every piece (see :meth:`.Mobject.get_depth_pieces`) and every
        ``shade_in_3d`` or depth-tested mobject without pieces is put in one
        of :attr:`depth_slices` slices of the frame's depth range.  Slices
        draw far to near; inside a slice, top-level scene mobjects (``roots``
        maps ``id(mobject)`` to its top-level index) draw in scene order;
        inside one of them, its mobjects without pieces far to near, then the
        pieces of each mobject with pieces, in family order, far to near.
        The pieces of one mobject in one slice form one run, drawn as one
        path.  Mobjects that are not 3D are drawn last, in
        display order.  Each 3D run carries the depth key of its nearest
        point, with which the renderer tests it against the depth of mesh
        surfaces per pixel.
        """
        if not mobjects:
            return DrawOrder.whole(mobjects)
        depth_axis = self.get_rotation_matrix()[2]
        focal_distance = self.get_focal_distance()

        def keys(mobject: Mobject, points: np.ndarray) -> np.ndarray:
            z = self.transform_points_pre_display(mobject, points)[:, 2]
            return 1 / np.maximum(focal_distance - z, 1e-6)

        roots = roots or {}
        members = len(mobjects) + 1
        owners: list[np.ndarray] = []
        parts: list[np.ndarray] = []
        depths: list[np.ndarray] = []
        groups: list[np.ndarray] = []
        piece_keys: list[np.ndarray] = []
        pieced = False
        for index, mobject in enumerate(mobjects):
            root = roots.get(id(mobject), index)
            centers = mobject.get_depth_pieces() if mobject.depth_test else None
            if centers is not None and len(centers):
                centers = np.asarray(centers, dtype=float)
                count = len(centers)
                owners.append(np.full(count, index))
                parts.append(np.arange(count))
                depths.append(centers @ depth_axis)
                groups.append(np.full(count, root * members + index + 1))
                piece_keys.append(keys(mobject, centers))
                pieced = True
                continue
            is_3d = mobject.depth_test or getattr(mobject, "shade_in_3d", False)
            owners.append(np.array([index]))
            parts.append(np.array([-1]))
            if is_3d:
                depths.append(
                    np.array([np.dot(mobject.get_z_index_reference_point(), depth_axis)])
                )
                points = mobject.points
                piece_keys.append(
                    np.array([keys(mobject, points).max() if len(points) else np.nan])
                )
            else:
                depths.append(np.array([np.inf]))
                piece_keys.append(np.array([np.nan]))
            # Flat mobjects keep their display order among themselves.
            groups.append(np.array([root * members if is_3d else index]))
        owner = np.concatenate(owners)
        part = np.concatenate(parts)
        key = np.concatenate(piece_keys)
        depth_slice = np.zeros(len(owner), dtype=np.int64)
        if pieced:
            order, depth_slice = manim_core.depth_order(
                np.concatenate(depths), np.concatenate(groups), self.depth_slices
            )
            owner, part, key = owner[order], part[order], key[order]

        # A run ends where the mobject or the slice changes: its key is that
        # of its nearest piece, so it must not span the depth range.
        starts = np.flatnonzero(
            (np.diff(owner, prepend=-1) != 0) | (np.diff(depth_slice, prepend=-1) != 0)
        )
        ends = np.append(starts[1:], len(owner))
        has_pieces = part[starts] >= 0
        return DrawOrder(
            mobjects,
            owner[starts].astype(np.int64),
            np.where(has_pieces, starts, 0).astype(np.int64),
            np.where(has_pieces, ends, 0).astype(np.int64),
            part.astype(np.int64),
            np.maximum.reduceat(key, starts) if len(starts) else key,
        )

    def get_phi(self) -> float:
        """Returns the Polar angle (the angle off Z_AXIS) phi.

        Returns
        -------
        float
            The Polar angle in radians.
        """
        return self.phi_tracker.get_value()

    def get_theta(self) -> float:
        """Returns the Azimuthal i.e the angle that spins the camera around the Z_AXIS.

        Returns
        -------
        float
            The Azimuthal angle in radians.
        """
        return self.theta_tracker.get_value()

    def get_focal_distance(self) -> float:
        """Returns focal_distance of the Camera.

        Returns
        -------
        float
            The focal_distance of the Camera in MUnits.
        """
        return self.focal_distance_tracker.get_value()

    def get_gamma(self) -> float:
        """Returns the rotation of the camera about the vector from the ORIGIN to the Camera.

        Returns
        -------
        float
            The angle of rotation of the camera about the vector
            from the ORIGIN to the Camera in radians
        """
        return self.gamma_tracker.get_value()

    def get_zoom(self) -> float:
        """Returns the zoom amount of the camera.

        Returns
        -------
        float
            The zoom amount of the camera.
        """
        return self.zoom_tracker.get_value()

    def set_phi(self, value: float) -> None:
        """Sets the polar angle i.e the angle between Z_AXIS and Camera through ORIGIN in radians.

        Parameters
        ----------
        value
            The new value of the polar angle in radians.
        """
        self.phi_tracker.set_value(value)

    def set_theta(self, value: float) -> None:
        """Sets the azimuthal angle i.e the angle that spins the camera around Z_AXIS in radians.

        Parameters
        ----------
        value
            The new value of the azimuthal angle in radians.
        """
        self.theta_tracker.set_value(value)

    def set_focal_distance(self, value: float) -> None:
        """Sets the focal_distance of the Camera.

        Parameters
        ----------
        value
            The focal_distance of the Camera.
        """
        self.focal_distance_tracker.set_value(value)

    def set_gamma(self, value: float) -> None:
        """Sets the angle of rotation of the camera about the vector from the ORIGIN to the Camera.

        Parameters
        ----------
        value
            The new angle of rotation of the camera.
        """
        self.gamma_tracker.set_value(value)

    def set_zoom(self, value: float) -> None:
        """Sets the zoom amount of the camera.

        Parameters
        ----------
        value
            The zoom amount of the camera.
        """
        self.zoom_tracker.set_value(value)

    def reset_rotation_matrix(self) -> None:
        """Do nothing: the rotation matrix is computed from the angles when read.

        Kept for code that called it after changing the angles.
        """

    def get_rotation_matrix(self) -> MatrixMN:
        """Return the rotation matrix for the camera's current angles.

        Returns
        -------
        numpy.ndarray
            The rotation matrix, computed from the angle trackers when called.
        """
        return self.generate_rotation_matrix()

    @property
    def rotation_matrix(self) -> MatrixMN:
        """The rotation matrix for the camera's current angles (read-only).

        Change the angle trackers, or use setters such as :meth:`set_theta`, to
        rotate the camera.
        """
        return self.get_rotation_matrix()

    def generate_rotation_matrix(self) -> MatrixMN:
        """Compute the rotation from world axes to camera axes.

        For column vectors, the matrix is
        ``Rz(gamma) @ Rx(-phi) @ Rz(-theta - pi / 2)`` using the current
        angle trackers. :meth:`project_points` centers points before applying
        this rotation, and applies perspective and zoom afterwards.

        Returns
        -------
        numpy.ndarray
            The 3-by-3 rotation matrix, without translation or scaling.
        """
        phi = self.get_phi()
        theta = self.get_theta()
        gamma = self.get_gamma()
        return rotation_matrix_zxz(gamma, -phi, -theta - 90 * DEGREES)

    def project_points(self, points: Point3D_Array) -> Point3D_Array:
        """Applies the current rotation_matrix as a projection
        matrix to the passed array of points.

        Parameters
        ----------
        points
            The list of points to project.

        Returns
        -------
        np.array
            The points after projecting.
        """
        view = self._view
        frame_center, focal_distance, zoom, rot_matrix = (
            view if view is not None else self._current_view()
        )

        # x and y scale by d / (d - z); with exponential_projection by
        # exp(z / d) for positive z, which avoids artifacts for high z.
        return manim_core.project_points(
            np.asarray(points, dtype=np.float64).reshape(-1, 3),
            tuple(np.asarray(frame_center, dtype=np.float64).reshape(3)),
            np.asarray(rot_matrix, dtype=np.float64).tolist(),
            float(focal_distance),
            float(zoom),
            bool(self.exponential_projection),
        )

    def project_point(self, point: Point3D) -> Point3D:
        """Applies the current rotation_matrix as a projection
        matrix to the passed point.

        Parameters
        ----------
        point
            The point to project.

        Returns
        -------
        np.array
            The point after projection.
        """
        return self.project_points(point.reshape((1, 3)))[0, :]

    def transform_points_pre_display(
        self,
        mobject: Mobject,
        points: Point3D_Array,
    ) -> Point3D_Array:  # TODO: Write Docstrings for this Method.
        points = super().transform_points_pre_display(mobject, points)
        fixed_orientation = mobject in self.fixed_orientation_mobjects
        fixed_in_frame = mobject in self.fixed_in_frame_mobjects

        if fixed_in_frame:
            return points
        if fixed_orientation:
            center_func = self.fixed_orientation_mobjects[mobject]
            center = center_func()
            new_center = self.project_point(center)
            return points + (new_center - center)
        else:
            return self.project_points(points)

    def add_fixed_orientation_mobjects(
        self,
        *mobjects: Mobject,
        use_static_center_func: bool = False,
        center_func: Callable[[], Point3D] | None = None,
    ) -> None:
        """This method allows the mobject to have a fixed orientation,
        even when the camera moves around.
        E.G If it was passed through this method, facing the camera, it
        will continue to face the camera even as the camera moves.
        Highly useful when adding labels to graphs and the like.

        Parameters
        ----------
        *mobjects
            The mobject whose orientation must be fixed.
        use_static_center_func
            Whether or not to use the function that takes the mobject's
            center as centerpoint, by default False
        center_func
            The function which returns the centerpoint
            with respect to which the mobject will be oriented, by default None
        """

        # This prevents the computation of mobject.get_center
        # every single time a projection happens
        def get_static_center_func(mobject: Mobject) -> Callable[[], Point3D]:
            point = mobject.get_center()
            return lambda: point

        for mobject in mobjects:
            if center_func:
                func = center_func
            elif use_static_center_func:
                func = get_static_center_func(mobject)
            else:
                func = mobject.get_center
            for submob in mobject.get_family():
                self.fixed_orientation_mobjects[submob] = func

    def add_fixed_in_frame_mobjects(self, *mobjects: Mobject) -> None:
        """This method allows the mobject to have a fixed position,
        even when the camera moves around.
        E.G If it was passed through this method, at the top of the frame, it
        will continue to be displayed at the top of the frame.

        Highly useful when displaying Titles or formulae or the like.

        Parameters
        ----------
        **mobjects
            The mobject to fix in frame.
        """
        for mobject in extract_mobject_family_members(mobjects):
            self.fixed_in_frame_mobjects.add(mobject)

    def remove_fixed_orientation_mobjects(self, *mobjects: Mobject) -> None:
        """If a mobject was fixed in its orientation by passing it through
        :meth:`.add_fixed_orientation_mobjects`, then this undoes that fixing.
        The Mobject will no longer have a fixed orientation.

        Parameters
        ----------
        mobjects
            The mobjects whose orientation need not be fixed any longer.
        """
        for mobject in extract_mobject_family_members(mobjects):
            if mobject in self.fixed_orientation_mobjects:
                del self.fixed_orientation_mobjects[mobject]

    def remove_fixed_in_frame_mobjects(self, *mobjects: Mobject) -> None:
        """If a mobject was fixed in frame by passing it through
        :meth:`.add_fixed_in_frame_mobjects`, then this undoes that fixing.
        The Mobject will no longer be fixed in frame.

        Parameters
        ----------
        mobjects
            The mobjects which need not be fixed in frame any longer.
        """
        for mobject in extract_mobject_family_members(mobjects):
            if mobject in self.fixed_in_frame_mobjects:
                self.fixed_in_frame_mobjects.remove(mobject)


class _CameraAnimate:
    """Small ``animate`` builder for the camera angle API."""

    def __init__(self, camera: Camera) -> None:
        self.camera = camera

    def _angle_animation(self, **target: float) -> _CameraEulerAnimation:
        return _CameraEulerAnimation(self.camera, **target)

    def increment_phi(self, delta: float, **kwargs: Any) -> _CameraEulerAnimation:
        return self._angle_animation(phi_delta=delta, **kwargs)

    def increment_theta(self, delta: float, **kwargs: Any) -> _CameraEulerAnimation:
        return self._angle_animation(theta_delta=delta, **kwargs)

    def increment_gamma(self, delta: float, **kwargs: Any) -> _CameraEulerAnimation:
        return self._angle_animation(gamma_delta=delta, **kwargs)


class _CameraEulerAnimation(Animation):
    """Interpolate a classic camera's stored Euler angles.

    It is intentionally a small animation: the camera is not a ``Mobject``,
    but the scene's animation loop only needs begin/interpolate/finish and a
    run time.
    """

    def __init__(
        self,
        camera: Camera,
        *,
        theta_delta: float = 0.0,
        phi_delta: float = 0.0,
        gamma_delta: float = 0.0,
        run_time: float = 1.0,
    ) -> None:
        super().__init__(None, run_time=run_time)
        self.camera = camera
        self.theta_delta = theta_delta
        self.phi_delta = phi_delta
        self.gamma_delta = gamma_delta
        self._start = (camera.get_theta(), camera.get_phi(), camera.get_gamma())

    def begin(self) -> None:
        self._start = (
            self.camera.get_theta(),
            self.camera.get_phi(),
            self.camera.get_gamma(),
        )
        self.interpolate(0.0)

    def interpolate(self, alpha: float) -> None:
        theta, phi, gamma = self._start
        self.camera.set_euler_angles(
            theta + self.theta_delta * alpha,
            phi + self.phi_delta * alpha,
            gamma + self.gamma_delta * alpha,
        )

    def finish(self) -> None:
        self.interpolate(1.0)

    def clean_up_from_scene(self, scene: Any) -> None:
        pass

    def update_mobjects(self, dt: float) -> None:
        pass

    def is_introducer(self) -> bool:
        return False

    def is_remover(self) -> bool:
        return False
