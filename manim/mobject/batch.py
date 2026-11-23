r"""Many simple shapes as one mobject, updated with arrays.

A scene with hundreds of moving arrows or circles spends its time in Python:
every :class:`.Arrow` and :class:`.Circle` runs its own updater, and every
one of their methods makes dozens of small NumPy calls.  A batch holds N
shapes as the subpaths of one :class:`.VMobject` instead.  Its geometry is
rebuilt from arrays in one native call and drawn in one GPU call, so one
updater moves all N shapes; this is what matplotlib's collections and
ManimGL's ``VectorField`` do for the same reason.

The shapes of a batch share one style (colour, width, opacity); use several
batches for several styles.  Animations that work on any :class:`.VMobject`
(:class:`.Create`, :class:`.Transform` between batches of equal size,
:class:`.FadeIn`, depth ordering in 3D) work on batches.

.. manim:: FourierArrows

    class FourierArrows(Scene):
        def construct(self):
            n = 40
            freqs = np.arange(n) - n // 2
            coefs = np.exp(-0.15 * freqs**2 + 1j * freqs) * 1.5
            arrows = ArrowBatch(n, stroke_width=2, tip_length=0.15)
            circles = CircleBatch(n, stroke_width=1, stroke_color=BLUE, stroke_opacity=0.5)
            t = ValueTracker(0)

            def update(group):
                z = coefs * np.exp(TAU * 1j * freqs * t.get_value())
                ends = np.cumsum(z)
                starts = ends - z
                arrows.put_start_and_end(complex_to_R3_array(starts), complex_to_R3_array(ends))
                circles.set_circles(complex_to_R3_array(starts), np.abs(z))

            group = VGroup(circles, arrows).add_updater(update)
            update(group)
            self.add(group)
            self.play(t.animate.set_value(1), run_time=4, rate_func=linear)
"""

from __future__ import annotations

__all__ = [
    "ArrowBatch",
    "CircleBatch",
    "DotBatch",
    "LineBatch",
    "complex_to_R3_array",
]

from typing import TYPE_CHECKING, Any

import manim_core
import numpy as np

from ..constants import DEFAULT_ARROW_TIP_LENGTH, DEFAULT_DOT_RADIUS
from .types.vectorized_mobject import VGroup, VMobject

if TYPE_CHECKING:
    from typing_extensions import Self

    from ..typing import Point3D_Array, Point3DLike_Array


def complex_to_R3_array(values: np.ndarray) -> Point3D_Array:
    """Complex numbers ``x + iy`` as rows ``(x, y, 0)``."""
    values = np.asarray(values)
    return np.column_stack([values.real, values.imag, np.zeros(values.shape)])


def _rows(points: Point3DLike_Array | None, count: int) -> np.ndarray:
    if points is None:
        return np.zeros((count, 3))
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) != count:
        raise ValueError(f"expected {count} points, got {len(points)}")
    return points


class LineBatch(VMobject):
    """``n`` straight segments in one path.

    Parameters
    ----------
    n
        The number of segments.
    starts, ends
        ``(n, 3)`` arrays; the segments start degenerate at the origin.
    kwargs
        Style, as for :class:`.VMobject`.
    """

    def __init__(
        self,
        n: int,
        starts: Point3DLike_Array | None = None,
        ends: Point3DLike_Array | None = None,
        **kwargs: Any,
    ) -> None:
        self.n = int(n)
        super().__init__(**kwargs)
        self.put_start_and_end(_rows(starts, self.n), _rows(ends, self.n))

    def put_start_and_end(self, starts: Point3DLike_Array, ends: Point3DLike_Array) -> Self:
        self.points = manim_core.line_points(_rows(starts, self.n), _rows(ends, self.n))
        return self


class CircleBatch(VMobject):
    """``n`` circles in the xy-plane in one path, laid out as :class:`.Circle`.

    Parameters
    ----------
    n
        The number of circles.
    centers
        ``(n, 3)`` array; circles start at the origin.
    radii
        ``(n,)`` array or one radius for all.
    kwargs
        Style, as for :class:`.VMobject`; circles are outlines by default.
    """

    segments = 8

    def __init__(
        self,
        n: int,
        centers: Point3DLike_Array | None = None,
        radii: float | np.ndarray = 1.0,
        **kwargs: Any,
    ) -> None:
        self.n = int(n)
        super().__init__(**kwargs)
        self.set_circles(_rows(centers, self.n), radii)

    def set_circles(self, centers: Point3DLike_Array, radii: float | np.ndarray) -> Self:
        radii = np.broadcast_to(np.asarray(radii, dtype=float), (self.n,))
        self.points = manim_core.circle_points(_rows(centers, self.n), radii, self.segments)
        return self


class DotBatch(CircleBatch):
    """``n`` filled dots in one path, sized like :class:`.Dot`."""

    def __init__(
        self,
        n: int,
        centers: Point3DLike_Array | None = None,
        radius: float | np.ndarray = DEFAULT_DOT_RADIUS,
        fill_opacity: float = 1.0,
        stroke_width: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n, centers, radius, fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs
        )


class ArrowBatch(VGroup):
    """``n`` arrows: one path of shafts and one path of filled triangle tips.

    Tips are ``tip_length`` long and ``tip_width`` wide (as
    :class:`.ArrowTriangleFilledTip`), shrunk to at most
    ``max_tip_length_to_length_ratio`` of their arrow, and keep their size as
    arrows grow, like :class:`.Arrow`.  Unlike :class:`.Arrow`, the stroke
    width is the same for all arrows.

    Parameters
    ----------
    n
        The number of arrows.
    starts, ends
        ``(n, 3)`` arrays.
    color
        Colour of shafts and tips.
    stroke_width, stroke_opacity
        Of the shafts; tips take the stroke opacity as their fill opacity.
    """

    def __init__(
        self,
        n: int,
        starts: Point3DLike_Array | None = None,
        ends: Point3DLike_Array | None = None,
        tip_length: float = DEFAULT_ARROW_TIP_LENGTH,
        tip_width: float | None = None,
        max_tip_length_to_length_ratio: float = 0.25,
        color: Any = None,
        stroke_width: float = 6,
        stroke_opacity: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n = int(n)
        self.tip_length = tip_length
        self.tip_width = tip_length if tip_width is None else tip_width
        self.max_tip_length_to_length_ratio = max_tip_length_to_length_ratio
        style = {} if color is None else {"color": color}
        self.shafts = VMobject(stroke_width=stroke_width, stroke_opacity=stroke_opacity, **style)
        self.tips = VMobject(fill_opacity=stroke_opacity, stroke_width=0, **style)
        if color is None:
            self.tips.set_fill(self.shafts.get_stroke_color())
        self.add(self.shafts, self.tips)
        self.put_start_and_end(_rows(starts, self.n), _rows(ends, self.n))

    def put_start_and_end(self, starts: Point3DLike_Array, ends: Point3DLike_Array) -> Self:
        shafts, tips = manim_core.arrow_points(
            _rows(starts, self.n),
            _rows(ends, self.n),
            self.tip_length,
            self.tip_width,
            self.max_tip_length_to_length_ratio,
        )
        self.shafts.points = shafts
        self.tips.points = tips
        return self
