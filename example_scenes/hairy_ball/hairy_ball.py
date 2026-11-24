"""The hairy ball from 3Blue1Brown's "The Hairy Ball Theorem" (2026), on ManimVK.

Ported from 3b1b/videos, _2026/hairy_ball/spheres.py (ManimGL) to ManimVK
(Manim Community API on the Vulkan renderer):

* SingleNullPointHairyBall / SingleNullPointHairyBallRevealed:
  50 000 hairs combed by the pull-back of a constant flow under stereographic
  projection, with a single bald point at the north pole.
* PairsOfNullPoints: fields with two null points, changed by rotating the
  plane field, then flowing stream lines.
* HairyEarth: the end of IntroduceVectorField; a field growing over the
  sphere, a denser field, stream lines and the Earth under them.

Render with the Vulkan renderer, for example::

    manim -pqh --renderer vulkan hairy_ball.py SingleNullPointHairyBall

HairyEarth needs two equirectangular Earth images named EarthTextureMap and
NightEarthTextureMap (.jpg or .png) next to this file or in the assets
directory, for instance NASA's Blue Marble and Black Marble.
"""

from __future__ import annotations

import numpy as np

from manim import *

DEG = DEGREES
# ManimGL's default Sphere resolution.
SPHERE_RESOLUTION = (101, 51)


# ---------------------------------------------------------------------------
# Spherical vector fields (same functions as the original)
# ---------------------------------------------------------------------------


def fibonacci_sphere(samples: int = 1000) -> np.ndarray:
    """Nearly uniform points on the unit sphere along a golden-angle spiral."""
    phi = np.pi * (np.sqrt(5) - 1)
    index = np.arange(samples)
    y = 1 - (index / (samples - 1)) * 2
    radius = np.sqrt(1 - y * y)
    theta = phi * index
    return np.column_stack([np.cos(theta) * radius, y, np.sin(theta) * radius])


def stereographic_proj(points3d: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    x, y, z = points3d.T
    denom = 1 - z
    denom[np.abs(denom) < epsilon] = np.inf
    return np.array([x / denom, y / denom, 0 * z]).T


def right_func(points: np.ndarray) -> np.ndarray:
    return np.repeat([[1.0, 0.0]], len(points), axis=0)


def stereographic_vector_field(points3d: np.ndarray, vector_field_2d) -> np.ndarray:
    """Pull a plane field back to the sphere through the inverse stereographic
    projection: the image of the field under its Jacobian."""
    points2d = stereographic_proj(points3d)[:, :2]
    vect_u, vect_v = vector_field_2d(points2d).T
    u, v = points2d.T
    denom_squared = (1 + u**2 + v**2) ** 2
    dx_du = 2 * (1 + v**2 - u**2) / denom_squared
    dx_dv = -4 * u * v / denom_squared
    dy_du = -4 * u * v / denom_squared
    dy_dv = 2 * (1 + u**2 - v**2) / denom_squared
    dz_du = 4 * u / denom_squared
    dz_dv = 4 * v / denom_squared
    return np.array([
        dx_du * vect_u + dx_dv * vect_v,
        dy_du * vect_u + dy_dv * vect_v,
        dz_du * vect_u + dz_dv * vect_v,
    ]).T


def rotation_field(points3d: np.ndarray, axis=IN) -> np.ndarray:
    return np.cross(points3d, axis)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


# ---------------------------------------------------------------------------
# Mobjects
# ---------------------------------------------------------------------------


class SphereVectorField(VGroup):
    """ManimGL's VectorField for points on a sphere, as two batched paths.

    ``shafts`` holds one straight subpath per sample, ``tips`` one small
    filled triangle per sample, turned towards the camera like ManimGL's
    unflattened strokes.  Both are depth tested, so the half of the field
    behind the sphere is hidden.  Vectors are drawn ``max_len * tanh(norm /
    max_len)`` long, with ``max_len`` the distance between the first two
    samples times ``max_vect_len_to_step_size``, as in ManimGL.
    """

    def __init__(
        self,
        func,
        sample_coords: np.ndarray,
        radius: float = 1.0,
        max_vect_len_to_step_size: float = 1.0,
        color: ParsableManimColor = BLUE,
        stroke_width: float = 1.0,
        stroke_opacity: float = 0.8,
        tip_width_ratio: float = 4.0,
        tip_len_to_width: float = 0.01,
        show_tips: bool = True,
        camera: Camera | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.func = func
        self.sample_coords = np.asarray(sample_coords, dtype=float)
        self.radius = radius
        self.sample_points = radius * self.sample_coords
        step = np.linalg.norm(self.sample_points[1] - self.sample_points[0])
        self.max_len = max_vect_len_to_step_size * step
        self.base_stroke_width = stroke_width
        self.tip_width_ratio = tip_width_ratio
        self.tip_len_to_width = tip_len_to_width
        self.show_tips = show_tips
        self.camera = camera
        self.shafts = VMobject(stroke_color=color, stroke_width=stroke_width,
                               stroke_opacity=stroke_opacity, fill_opacity=0)
        self.tips = VMobject(fill_color=color, fill_opacity=stroke_opacity, stroke_width=0)
        self.add(self.shafts, self.tips)
        self.update_vectors()
        self.apply_depth_test()
        if show_tips and camera is not None:
            # Keep the tips turned towards the moving camera.
            self.add_updater(lambda m: m.update_vectors())

    def set_vector_color(self, color, opacity=None) -> SphereVectorField:
        self.shafts.set_stroke(color, opacity=opacity)
        self.tips.set_fill(color, opacity=opacity)
        return self

    def set_stroke_width_scale(self, factor: float) -> SphereVectorField:
        self.shafts.set_stroke(width=factor * self.base_stroke_width)
        return self

    def update_vectors(self) -> SphereVectorField:
        vects = self.radius * self.func(self.sample_coords)
        norms = np.linalg.norm(vects, axis=1, keepdims=True)
        units = vects / np.where(norms == 0, 1, norms)
        drawn = self.max_len * np.tanh(norms / self.max_len) if np.isfinite(self.max_len) else norms
        # Stroke widths are 1/100 of a scene unit, as in ManimGL.
        tip_width = 0.01 * self.tip_width_ratio * self.base_stroke_width
        tip_len = np.minimum(self.tip_len_to_width * self.tip_width_ratio * self.base_stroke_width, drawn)
        if not self.show_tips:
            tip_len = np.zeros_like(drawn)
        start = self.sample_points
        end = start + drawn * units
        head_base = end - tip_len * units
        self.shafts.set_points(_lines_to_cubics(start, head_base))
        if not self.show_tips:
            return self
        if self.camera is not None and hasattr(self.camera, "get_rotation_matrix"):
            view = self.camera.get_rotation_matrix()[2]
            side = normalize_rows(np.cross(units, view))
        else:
            side = normalize_rows(np.cross(units, normalize_rows(start)))
        half = 0.5 * tip_width * side
        self.tips.set_points(_triangles_to_cubics(head_base - half, end, head_base + half))
        return self


def _lines_to_cubics(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """One straight cubic Bézier per row: a separate subpath each."""
    delta = end - start
    return np.stack([start, start + delta / 3, start + 2 * delta / 3, end], axis=1).reshape(-1, 3)


def _triangles_to_cubics(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """One closed triangular subpath per row."""
    edges = [_lines_to_cubics(p, q).reshape(-1, 4, 3) for p, q in ((a, b), (b, c), (c, a))]
    return np.concatenate(edges, axis=1).reshape(-1, 3)


class SphereFlow(VGroup):
    """Stream lines on a sphere with flashes running along them.

    The analogue of ManimGL's ``AnimatedStreamLines(SphereStreamLines(...))``:
    every line is integrated once (RK4, projected back onto the sphere after
    each step), and each frame shows the moving window of a passing flash,
    tapered from tail to head, as three batched depth-tested paths.
    """

    def __init__(
        self,
        func,
        seeds: np.ndarray,
        radius: float = 1.0,
        solution_time: float = 1.0,
        dt: float = 0.05,
        lag_range: float = 4.0,
        rate_multiple: float = 1.0,
        color: ParsableManimColor = WHITE,
        stroke_width: float = 1.0,
        stroke_opacity: float = 0.5,
        samples_per_part: int = 4,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        rng = np.random.default_rng(seed)
        steps = max(1, int(round(solution_time / dt)))
        points = normalize_rows(np.asarray(seeds, dtype=float))
        path = [points]
        for _ in range(steps):
            k1 = func(points)
            k2 = func(normalize_rows(points + 0.5 * dt * k1))
            k3 = func(normalize_rows(points + 0.5 * dt * k2))
            k4 = func(normalize_rows(points + dt * k3))
            points = normalize_rows(points + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
            path.append(points)
        self.trajectories = radius * np.stack(path, axis=1)  # (lines, samples, 3)
        self.period = solution_time / rate_multiple
        self.times = -lag_range * rng.random(len(points))
        self.samples_per_part = samples_per_part
        self.flow_opacity = 1.0
        self.base_opacity = stroke_opacity
        self.parts = VGroup(*(
            VMobject(stroke_color=color, stroke_width=width * stroke_width, fill_opacity=0)
            for width in (0.35, 0.7, 1.0)
        ))
        self.add(self.parts)
        self.apply_depth_test()
        self.rebuild()
        self.add_updater(lambda m, dt: m.advance(dt))

    def set_flow_opacity(self, opacity: float) -> SphereFlow:
        self.flow_opacity = opacity
        for part, weight in zip(self.parts, (0.5, 0.8, 1.0)):
            part.set_stroke(opacity=weight * self.base_opacity * opacity)
        return self

    def advance(self, dt: float) -> SphereFlow:
        self.times += dt
        return self.rebuild()

    def rebuild(self) -> SphereFlow:
        alpha = (np.maximum(self.times, 0) % self.period) / self.period
        upper = np.clip(2 * alpha, 0, 1)
        lower = np.clip(2 * alpha - 1, 0, 1)
        last = self.trajectories.shape[1] - 1
        lines = np.arange(len(self.trajectories))[:, None]
        m = self.samples_per_part
        for k, part in enumerate(self.parts):
            a = lower + (upper - lower) * k / 3
            b = lower + (upper - lower) * (k + 1) / 3
            s = a[:, None] + (b - a)[:, None] * np.linspace(0, 1, m + 1)[None, :]
            position = s * last
            i0 = np.minimum(position.astype(int), last - 1)
            frac = (position - i0)[..., None]
            pts = (1 - frac) * self.trajectories[lines, i0] + frac * self.trajectories[lines, i0 + 1]
            segment_start = pts[:, :-1].reshape(-1, 3)
            segment_end = pts[:, 1:].reshape(-1, 3)
            part.set_points(_lines_to_cubics(segment_start, segment_end))
        self.set_flow_opacity(self.flow_opacity)
        return self


def sphere_mesh_lines(
    radius: float, resolution=(51, 25), samples: int = 64, normal_nudge: float = 1e-2
) -> VMobject:
    """Latitude and longitude lines on a sphere (ManimGL's SurfaceMesh of it),
    nudged out along the normals by ``normal_nudge`` as SurfaceMesh does."""
    nu, nv = resolution
    radius = radius + normal_nudge
    t = np.linspace(0, 1, samples)
    segments = []
    for u in np.linspace(0, TAU, nu)[:-1]:
        v = PI * t
        pts = radius * np.column_stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), -np.cos(v)])
        segments.append(pts)
    for v in np.linspace(0, PI, nv)[1:-1]:
        u = TAU * t
        pts = radius * np.column_stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), -np.cos(v) * np.ones_like(u)])
        segments.append(pts)
    mesh = VMobject(fill_opacity=0)
    mesh.set_points(np.concatenate([_lines_to_cubics(p[:-1], p[1:]) for p in segments]))
    mesh.apply_depth_test()
    return mesh


# ---------------------------------------------------------------------------
# Camera: ManimGL's frame.reorient(theta, phi, gamma, center, height)
# ---------------------------------------------------------------------------


class GLCameraScene(ThreeDScene):
    """ThreeDScene with ManimGL's frame orientation conventions.

    ManimGL's theta is ManimCE's theta + 90 degrees; its frame height h maps to
    ``zoom = 8 / h`` with the camera ``2 h`` from the frame center, which is
    ManimGL's default focal distance.
    """

    def frame_values(self, theta, phi, gamma=0, center=None, height=None):
        values = [
            (self.camera.theta_tracker, (theta - 90) * DEG),
            (self.camera.phi_tracker, phi * DEG),
            (self.camera.gamma_tracker, gamma * DEG),
        ]
        if height is not None:
            values += [
                (self.camera.zoom_tracker, config.frame_height / height),
                (self.camera.focal_distance_tracker, 2 * height),
            ]
        return values

    def reorient(self, theta, phi, gamma=0, center=None, height=None) -> None:
        for tracker, value in self.frame_values(theta, phi, gamma, center, height):
            tracker.set_value(value)
        if center is not None:
            self.camera.frame_center = np.array(center, dtype=float)

    def reorient_anims(self, theta, phi, gamma=0, center=None, height=None, **anim_args) -> list[Animation]:
        anims = [
            tracker.animate(**anim_args).set_value(value)
            for tracker, value in self.frame_values(theta, phi, gamma, center, height)
        ]
        if center is not None:
            anims.append(self.camera.frame.animate(**anim_args).move_to(np.array(center, dtype=float)))
        return anims

    def gl_height(self) -> float:
        return config.frame_height / self.camera.get_zoom()

    def add_ambient_rotation(self, rate: float) -> None:
        """ManimGL's frame.add_ambient_rotation: ``rate`` radians per second."""
        self.begin_ambient_camera_rotation(rate=rate)

    def clear_ambient_rotation(self) -> None:
        self.stop_ambient_camera_rotation()

    def scale_stroke_with_zoom(self, field: SphereVectorField) -> None:
        """ManimGL's set_scale_stroke_with_zoom(True)."""
        field.add_updater(lambda m: m.set_stroke_width_scale(self.camera.get_zoom()))


def in_time_span(animation: Animation, start: float, end: float, total: float) -> Animation:
    """ManimGL's ``time_span``: run ``animation`` from ``start`` to ``end``
    seconds into a play lasting ``total`` seconds."""
    rate = animation.rate_func
    animation.run_time = total

    def spanned(t: float) -> float:
        return rate(float(np.clip((t * total - start) / (end - start), 0, 1)))

    animation.rate_func = spanned
    return animation


def flow_fade(flow: SphereFlow, fade_in: bool, run_time: float = 1.0) -> Animation:
    return UpdateFromAlphaFunc(
        flow,
        lambda m, a: m.set_flow_opacity(a if fade_in else 1 - a),
        run_time=run_time,
    )


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------


class SingleNullPointHairyBall(GLCameraScene):
    hide_top = True

    def construct(self):
        radius = 3
        sphere = MeshSurface(
            Sphere(radius=radius, resolution=SPHERE_RESOLUTION),
            color=GREY_E, opacity=1, shading=(0.1, 0.1, 0.3),
        )
        self.camera.light_source.move_to(3 * LEFT + 12 * UP + 3 * OUT)
        self.reorient(-3, 161, 0)
        self.add(sphere)

        # Vector field
        def v_func(points3d):
            vects = stereographic_vector_field(points3d, right_func)
            return 0.2 * normalize_rows(vects)

        def out_func(points3d):
            return 0.2 * points3d

        n_points = 50_000
        rng = np.random.default_rng(1)
        v_range = (-0.95, 1) if self.hide_top else (-1, 1)
        u = TAU * rng.random(n_points)
        v = np.arccos(rng.uniform(*v_range, n_points))
        points = np.column_stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), -np.cos(v)])
        pre_field, field = [
            SphereVectorField(func, 1.01 * points, radius=radius, max_vect_len_to_step_size=3,
                              color=BLUE_E, stroke_width=0.5, stroke_opacity=0.75, show_tips=False)
            for func in [out_func, v_func]
        ]

        # The bald spot at the north pole, covered at first
        q_marks = Text("???", font_size=72)
        q_marks.rotate(-90 * DEG)
        q_marks.move_to(radius * OUT)
        disk = MeshSurface(
            Sphere(radius=radius, v_range=(0.9 * PI, PI), resolution=(64, 4)),
            color=BLACK, opacity=1, shading=(0, 0, 0),
        )
        disk.deactivate_depth_test()
        top_q = Group(disk, q_marks)
        if not self.hide_top:
            top_q.set_opacity(0)

        self.add(pre_field)
        self.play(
            ReplacementTransform(pre_field, field, run_time=2),
            *self.reorient_anims(-92, 11, 0, run_time=7),
            in_time_span(FadeIn(top_q), 3.5, 5, 7),
        )
        self.play(
            *self.reorient_anims(-168, 127, 0, run_time=10),
            in_time_span(FadeOut(top_q), 4.2, 5, 10),
        )


class SingleNullPointHairyBallRevealed(SingleNullPointHairyBall):
    hide_top = False


class PairsOfNullPoints(GLCameraScene):
    def construct(self):
        radius = 3
        sphere = MeshSurface(Sphere(radius=0.99 * radius, resolution=SPHERE_RESOLUTION), color=GREY_D, opacity=0.5)
        axes = ThreeDAxes(
            x_range=(-4, 4, 1), y_range=(-4, 4, 1), z_range=(-4, 4, 1),
            x_length=8 * radius, y_length=8 * radius, z_length=8 * radius,
        )
        axes.apply_depth_test()

        self.reorient(-155, 74, 0)
        self.add(axes, sphere)

        def source_and_sink(points2d, offset=-3):
            x, y = points2d.T
            return np.array([x - offset, y]).T

        def twirl_func(points2d, offset=-3):
            x, y = points2d.T
            return np.array([-y, x - offset]).T

        rotation = np.identity(3)[:, [0, 2, 1]]
        field = self.get_spherical_field(twirl_func, rotation=rotation)
        offset_tracker = ValueTracker(-1)

        self.add(field)
        self.reorient(-136, 77, 0, (0.91, 0.31, 0.79), 12.58)
        self.play(*self.reorient_anims(-177, 80, 0, (0.0, 0.0, 0.0), 8, run_time=4))
        self.add_ambient_rotation(1 * DEG)

        # Change the field around
        new_params = [
            (rotation_matrix(90 * DEG, UP).T, -5),
            (rotation_matrix(30 * DEG, DOWN).T, -2),
            (rotation_matrix(120 * DEG, LEFT).T, 3),
            (rotation, -1),
        ]
        for new_rot, offset in new_params:
            new_field = self.get_spherical_field(lambda ps, o=offset: twirl_func(ps, o), rotation=new_rot)
            self.play(Transform(field, new_field, run_time=2))

        self.play(
            offset_tracker.animate.set_value(-3),
            UpdateFromFunc(
                field,
                lambda m: m.become(self.get_spherical_field(
                    lambda ps: twirl_func(ps, offset_tracker.get_value()), rotation=rotation
                )),
            ),
            run_time=4,
        )

        # Show some flow
        streamlines = self.get_streamlines(field, radius)
        self.add(streamlines)
        self.wait(8)

        # New field
        field2 = self.get_spherical_field(source_and_sink)
        streamlines2 = self.get_streamlines(field2, radius, density=100)
        streamlines2.set_flow_opacity(0)

        self.play(flow_fade(streamlines, fade_in=False))
        self.remove(streamlines)
        self.play(Transform(field, field2))
        self.add(streamlines2)
        self.play(flow_fade(streamlines2, fade_in=True))
        self.wait(8)

    def get_streamlines(self, field, radius, density=100):
        return SphereFlow(
            field.func, fibonacci_sphere(int(4 * PI * density)), radius=radius,
            solution_time=1.0, dt=0.05, rate_multiple=0.2,
            color=WHITE, stroke_width=1, stroke_opacity=0.5,
        )

    def get_spherical_field(self, plane_func, n_sample_points=2000, stroke_width=2,
                            rotation=np.identity(3)):
        def v_func(points3d):
            rot_points = np.dot(points3d, rotation)
            vects = normalize_rows(stereographic_vector_field(rot_points, plane_func))
            return np.dot(vects, rotation.T)

        return SphereVectorField(
            v_func, 1.01 * fibonacci_sphere(n_sample_points), radius=3,
            color=BLUE, stroke_width=stroke_width, stroke_opacity=0.8, camera=self.camera,
        )


class HairyEarth(GLCameraScene):
    """The end of IntroduceVectorField: a combed sphere, its flow, the Earth."""

    def construct(self):
        self.camera.light_source.move_to([0, -10, 10])
        radius = 3
        sphere = MeshSurface(Sphere(radius=radius, resolution=SPHERE_RESOLUTION), color=BLUE_B, opacity=0.3)
        mesh = sphere_mesh_lines(radius)
        mesh.set_stroke(WHITE, 1, 0.15)

        self.reorient(0, 90, 0)
        self.play(
            *self.reorient_anims(30, 65, 0),
            FadeIn(sphere),
            Create(mesh, lag_ratio=0.01),
            run_time=3,
        )

        def v_func(points3d):
            v1 = stereographic_vector_field(points3d, right_func)
            v2 = normalize_rows(rotation_field(points3d, RIGHT))
            v3 = normalize_rows(rotation_field(points3d, RIGHT + IN))
            return (3 * v1 + v2 + v3) / 6

        start_point = radius * normalize(RIGHT + OUT)
        field = self.get_vector_field(v_func, 4000, start_point=start_point)
        self.scale_stroke_with_zoom(field)
        self.add_ambient_rotation(-2.5 * DEG)
        self.add(field)
        self.play(Create(field), run_time=5)
        self.wait(4)
        self.clear_ambient_rotation()
        self.play(*self.reorient_anims(0, 59, 0, (-0.03, 0.15, -0.08), 6.80, run_time=4))
        self.wait(2)

        # Denser field
        dense_field = self.get_vector_field(v_func, 50_000, mvltss=5.0)
        dense_field.set_vector_color(BLUE, opacity=0.35)
        self.scale_stroke_with_zoom(dense_field)
        self.play(
            FadeOut(field),
            Create(dense_field),
            *self.reorient_anims(-66, 52, 0, (-0.51, 0.27, 0.2), 2.81),
            run_time=4,
        )
        self.wait()
        self.play(
            FadeOut(dense_field),
            FadeIn(field),
            *self.reorient_anims(-43, 55, 0, (-2.42, 2.76, -0.55), 9.22),
            run_time=4,
        )

        # Stream lines and the Earth
        stream_lines = SphereFlow(
            lambda p: v_func(p), fibonacci_sphere(int(4 * PI * 400)), radius=radius,
            solution_time=1, dt=0.05, lag_range=10, rate_multiple=0.5,
            color=BLUE_B, stroke_width=3, stroke_opacity=0.8,
        )
        earth = TexturedSurface(
            Sphere(radius=radius, resolution=SPHERE_RESOLUTION), "EarthTextureMap", "NightEarthTextureMap"
        )
        earth.rotate(-90 * DEG, IN)
        earth.scale(1.001)
        earth.set_opacity(0)
        stream_lines.set_flow_opacity(0)
        self.add_ambient_rotation(2 * DEG)
        self.add(earth, stream_lines, mesh, field)
        self.play(
            earth.animate(run_time=3).set_opacity(1),
            flow_fade(stream_lines, fade_in=True, run_time=3),
            field.animate(run_time=2).set_vector_color(BLUE, opacity=0.25),
            *self.reorient_anims(-90, 74, 0, (-1.37, 0.04, 0.37), 5.68, run_time=3),
        )
        self.wait(10)

        # Zoom in to the null point
        self.clear_ambient_rotation()
        dense_field = self.get_vector_field(v_func, 10_000)
        dense_field.set_vector_color(BLUE, opacity=0.35)
        self.scale_stroke_with_zoom(dense_field)
        self.play(
            FadeOut(field, run_time=2),
            FadeIn(dense_field, run_time=2),
            *self.reorient_anims(-69, 54, 0, (-0.01, 0.19, -0.04), 4.81, run_time=5),
        )
        self.wait(3)
        self.play(*self.reorient_anims(129, 70, 0, (-0.18, 0.18, -0.1), 8.22, run_time=20))
        self.wait(10)

    def get_vector_field(self, v_func, n_points, start_point=None, mvltss=1.0):
        points = fibonacci_sphere(n_points)
        if start_point is not None:
            order = np.argsort(np.linalg.norm(points - normalize(start_point), axis=1))
            points = points[order]
        alpha = np.clip((n_points - 10_000) / (1000 - 10_000), 0, 1)
        stroke_width = interpolate(1, 3, alpha**2)
        return SphereVectorField(
            v_func, 1.01 * points, radius=3, max_vect_len_to_step_size=mvltss,
            color=BLUE, stroke_width=stroke_width, stroke_opacity=0.8, camera=self.camera,
        )
