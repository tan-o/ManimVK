"""manim_core kernels and the batch mobjects built on them."""

from __future__ import annotations

import manim_core
import numpy as np
import pytest

from manim import (
    DOWN,
    LEFT,
    OUT,
    RIGHT,
    UP,
    Arrow,
    ArrowBatch,
    Circle,
    CircleBatch,
    Create,
    DotBatch,
    Line,
    LineBatch,
    Square,
    VMobject,
    complex_to_R3_array,
)
from manim.utils.bezier import interpolate
from manim.utils.space_ops import rotation_matrix


def test_transform_points_matches_numpy():
    rng = np.random.default_rng(0)
    arrays = [rng.normal(size=(n, 3)) for n in (1, 4, 33)]
    matrix = rotation_matrix(0.7, np.array([1.0, 2.0, 3.0])) @ np.diag([2.0, 0.5, 1.0])
    about = np.array([0.3, -1.0, 2.0])
    shift = np.array([1.0, 2.0, 3.0])
    out = manim_core.transform_points(arrays, matrix, about, shift)
    for points, result in zip(arrays, out):
        np.testing.assert_allclose(result, (points - about) @ matrix.T + about + shift)
        assert result is not points
    moved = manim_core.transform_points(arrays, shift=shift)
    np.testing.assert_allclose(moved[2], arrays[2] + shift)
    with pytest.raises(ValueError):
        manim_core.transform_points([np.zeros((3, 2))])


def test_bounds_of_anchors_and_of_all_points():
    curve = np.array([[0, 0, 0], [5, 9, 0], [-4, 1, 0], [1, 1, 0]], dtype=float)
    assert manim_core.bounds([curve], 4) == ([0, 0, 0], [1, 1, 0])
    assert manim_core.bounds([curve]) == ([-4, 0, 0], [5, 9, 0])
    assert manim_core.bounds([]) is None
    assert manim_core.bounds([np.array([[2.0, 3.0, 4.0]])], 4) == ([2, 3, 4], [2, 3, 4])


def test_lerp_matches_interpolate():
    a, b = np.random.default_rng(1).normal(size=(2, 7, 4))
    np.testing.assert_array_equal(manim_core.lerp(a, b, 0.3), interpolate(a, b, 0.3))


def test_cubic_subpaths_match_vmobject_rules():
    square = Square()
    vm = VMobject()
    vm.set_points(np.concatenate([square.points, Line(LEFT, RIGHT).shift(UP * 3).points]))
    _, subpaths = manim_core.cubic_subpaths(vm.points, 4, 2, vm.tolerance_for_point_equality)
    assert subpaths == [(0, 16, True), (16, 20, False)]


def test_mobject_transforms_and_boxes_are_unchanged():
    square = Square().shift(RIGHT).scale(2).rotate(0.3).stretch(2, 0)
    reference = Square().points + RIGHT
    reference = (reference - [1, 0, 0]) * 2 + [1, 0, 0]
    reference = (reference - [1, 0, 0]) @ rotation_matrix(0.3, OUT).T + [1, 0, 0]
    center = reference.min(axis=0)[0] / 2 + reference.max(axis=0)[0] / 2
    reference[:, 0] = (reference[:, 0] - center) * 2 + center
    np.testing.assert_allclose(square.points, reference, atol=1e-12)
    np.testing.assert_allclose(square.get_center(), [1, 0, 0], atol=1e-12)
    assert square.width == pytest.approx(np.ptp(reference[:, 0]))


def test_circle_and_line_batches_match_single_mobjects():
    centers = np.array([[1.0, 2.0, 0.0], [-3.0, 0.5, 0.0]])
    radii = np.array([0.5, 2.0])
    batch = CircleBatch(2, centers, radii)
    single = np.concatenate(
        [Circle(radius=r).move_to(c).points for c, r in zip(centers, radii)]
    )
    np.testing.assert_allclose(batch.points, single, atol=1e-12)

    lines = LineBatch(2, [LEFT, UP], [RIGHT, DOWN])
    np.testing.assert_allclose(
        lines.points, np.concatenate([Line(LEFT, RIGHT).points, Line(UP, DOWN).points])
    )
    assert len(VMobject.get_subpath_split_indices_from_points(lines, lines.points)) == 2
    assert DotBatch(3).get_fill_opacity() == 1


def test_arrow_batch_matches_arrow_tip_geometry():
    batch = ArrowBatch(1, [LEFT], [RIGHT * 2], stroke_width=2)
    arrow = Arrow(LEFT, RIGHT * 2, buff=0)
    np.testing.assert_allclose(batch.tips.points[[0, 4, 8]], arrow.tip.points[[0, 4, 8]], atol=1e-9)
    np.testing.assert_allclose(batch.shafts.points[[0, -1]], arrow.points[[0, -1]], atol=1e-9)
    short = ArrowBatch(1, [np.zeros(3)], [RIGHT * 0.4])
    assert np.linalg.norm(short.tips.points[0] - short.tips.points[4 + 0]) <= 0.25 * 0.4 * 1.2


def test_batches_animate_like_vmobjects():
    a = CircleBatch(3, np.zeros((3, 3)), [1, 2, 3])
    b = CircleBatch(3, np.ones((3, 3)), [3, 2, 1])
    mid = a.copy().interpolate(a, b, 0.5)
    np.testing.assert_allclose(mid.points, (a.points + b.points) / 2)
    create = Create(a)
    create.begin()
    create.interpolate(0.5)
    assert 0 < len(a.points) <= len(b.points)


def test_complex_to_r3_array():
    np.testing.assert_array_equal(complex_to_R3_array([1 + 2j, -3j]), [[1, 2, 0], [0, -3, 0]])


def test_transforms_accept_broadcastable_vectors():
    square = Square()
    square.move_to(np.array([[1.0, 2.0, 0.0]]))  # e.g. a ValueTracker's points
    np.testing.assert_allclose(square.get_center(), [1, 2, 0])
    square.scale(2, about_point=np.array([[1.0, 2.0, 0.0]]))
    np.testing.assert_allclose(square.get_center(), [1, 2, 0])
    with pytest.raises(ValueError):
        square.shift(np.zeros(2))
