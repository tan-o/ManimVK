"""Depth-tested mobjects and mesh surfaces in a ThreeDScene.

The camera orders depth pieces (subpaths of depth-tested VMobjects, triangles
of mesh surfaces) back to front in depth slices; the Skia drawing context sends
the runs of a frame's VMobjects and meshes in one native call.  The native
module is a recorder here, so these check calls, not pixels.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from PIL import Image

from manim import (
    DEGREES,
    MeshSurface,
    Sphere,
    Square,
    TexturedSurface,
    ThreeDScene,
    VMobject,
    tempconfig,
)
from manim.camera import ThreeDCamera
from manim.renderer.skia import renderer as skia_renderer


def _hairs(count: int, radius: float = 1.0) -> VMobject:
    """``count`` short radial lines on a sphere, one subpath each."""
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    start, end = 1.02 * radius * directions, 1.2 * radius * directions
    delta = end - start
    points = np.stack([start, start + delta / 3, start + 2 * delta / 3, end], axis=1)
    hairs = VMobject()
    hairs.set_points(points.reshape(-1, 3))
    return hairs.apply_depth_test()


def test_depth_pieces_are_subpath_centers():
    hairs = _hairs(10)
    centers = hairs.get_depth_pieces()
    assert centers.shape == (10, 3)
    np.testing.assert_allclose(centers, hairs.points.reshape(10, 4, 3).mean(axis=1))
    assert VMobject().get_depth_pieces() is None
    assert _hairs(10).deactivate_depth_test().get_depth_pieces() is None


def test_draw_order_runs_far_to_near_and_flat_last():
    camera = ThreeDCamera(phi=0, theta=-90 * DEGREES)  # looking down the z axis
    hairs = _hairs(200)
    flat = Square()
    order = camera.get_draw_order([flat, hairs], {id(flat): 0, id(hairs): 1})
    runs = list(order.runs())
    assert runs[-1][0] is flat
    assert runs[-1][1] is None
    pieces = np.concatenate([p for m, p in runs if m is hairs])
    assert sorted(pieces.tolist()) == list(range(200))
    # Runs come far to near: the z of their pieces increases.
    run_depth = [
        hairs.get_depth_pieces()[p][:, 2].mean() for m, p in runs if m is hairs
    ]
    assert run_depth == sorted(run_depth)
    assert len(run_depth) <= camera.depth_slices


def test_without_pieces_draw_order_keeps_display_order():
    camera = ThreeDCamera()
    mobjects = [Square(), Square().shift(2 * np.array([1.0, 0, 0]))]
    assert list(camera.get_draw_order(mobjects).runs()) == [
        (mobjects[0], None),
        (mobjects[1], None),
    ]


def test_mesh_surface_follows_the_transformed_surface():
    sphere = Sphere(radius=2, resolution=(8, 4)).shift(np.array([1.0, 0, 0]))
    mesh = MeshSurface(sphere)
    assert mesh.resolution == (8, 4)
    assert len(mesh.points) == 9 * 5
    assert mesh.triangles.shape == (2 * 8 * 4, 3)
    np.testing.assert_allclose(
        np.linalg.norm(mesh.points - [1, 0, 0], axis=1), 2, atol=1e-9
    )
    assert mesh.get_depth_pieces().shape == (64, 3, 3)


def test_mesh_normals_point_outward_along_du_cross_dv():
    sphere = MeshSurface(Sphere(radius=2, resolution=(12, 6)))
    normals = sphere.get_vertex_normals()
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-9)
    # On a sphere about the origin, du x dv points away from the centre.
    outward = np.einsum("ij,ij->i", normals, sphere.points / 2)
    assert np.all(outward > 0.9)


def test_textured_surface_images_are_read_once(tmp_path):
    day = tmp_path / "day.png"
    night = tmp_path / "night.png"
    Image.new("RGB", (8, 4), (10, 200, 30)).save(day)
    Image.new("RGB", (16, 8), (5, 5, 50)).save(night)
    earth = TexturedSurface(Sphere(resolution=(6, 3)), day, night)
    (day_pixels, day_key), (night_pixels, night_key) = earth.get_images()
    assert earth.get_images()[0][1] == day_key != night_key
    assert not day_pixels.flags.writeable
    assert day_pixels.shape == (4, 8, 4)
    assert night_pixels.shape == (8, 16, 4)
    assert tuple(day_pixels[0, 0]) == (10, 200, 30, 255)
    assert MeshSurface(Sphere(resolution=(6, 3))).get_images() == []
    earth.fade(1)
    assert earth.get_opacity() == 0


class _Target:
    def __init__(self, calls: Counter, width: int, height: int) -> None:
        self.calls, self.width, self.height = calls, width, height
        self.paths: list[int] = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls[name] += 1

        return record

    def draw_display_list(self, objects, meshes, runs, parts):
        self.calls["draw_display_list"] += 1
        assert runs.dtype == np.int64
        assert runs.shape[1] == 4
        assert parts.dtype == np.int64
        for kind, index, start, end in runs.tolist():
            if kind == 0:
                self.calls["vmobject_runs"] += 1
                whole = len(objects[index][1])
                self.calls["subpaths"] += end - start if end > start else whole
                continue
            (xy, points, normals, uvs, depth, triangles), _, _, style = meshes[index]
            assert xy.dtype == points.dtype == normals.dtype == np.float32
            assert len(xy) == len(points) == len(normals) == len(uvs) == len(depth)
            assert triangles.dtype == np.int32
            self.calls["mesh_runs"] += 1
            self.calls["mesh_triangles"] += (
                end - start if end > start else len(triangles)
            )
            if style[2] is not None and end > start:
                self.calls["occlude"] += 1

    def restore_static_layer(self):
        return False

    def commit_into(self, frame):
        frame[...] = 0


class _Device:
    calls: Counter = Counter()

    def __init__(self, *args, **kwargs) -> None:
        pass

    def device_name(self) -> str:
        return "recorder"

    def create_skia_target(self, width, height):
        return _Target(self.calls, width, height)


@pytest.fixture
def calls(monkeypatch):
    _Device.calls = Counter()
    monkeypatch.setattr(skia_renderer.manim_vk, "Device", _Device)
    return _Device.calls


class _HairyBall(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=30 * DEGREES)
        self.add(MeshSurface(Sphere(resolution=(12, 6)), opacity=0.5), _hairs(3000))


def test_hairs_and_mesh_draw_in_few_runs(calls, tmp_path):
    with tempconfig(
        {
            "pixel_width": 160,
            "pixel_height": 90,
            "media_dir": str(tmp_path),
            "save_last_frame": True,
            "write_to_movie": False,
            "disable_caching": True,
        }
    ):
        _HairyBall().render()
    # Every hair and every triangle is drawn once ...
    assert calls["subpaths"] == 3000
    assert calls["mesh_triangles"] == 2 * 12 * 6
    # ... in a few dozen runs, not one per piece, all in one call per frame.
    assert calls["draw_display_list"] == 1
    assert calls["vmobject_runs"] <= ThreeDCamera.depth_slices
    assert calls["mesh_runs"] <= ThreeDCamera.depth_slices
    # The translucent sphere hides what is behind it.
    assert calls["occlude"] == calls["mesh_runs"]
