from __future__ import annotations

from pathlib import Path

import numpy as np

import manim.opengl as ogl
from manim import (
    AnimationGroup,
    ApplyWave,
    Arc,
    Circumscribe,
    Create,
    DrawBorderThenFill,
    FadeIn,
    FadeOut,
    FadeTransform,
    Flash,
    GrowFromCenter,
    Indicate,
    LaggedStart,
    MoveAlongPath,
    Rotate,
    Scene,
    ShrinkToCenter,
    Succession,
    Transform,
    Wiggle,
    Write,
    config,
)
from manim import BLUE, DOWN, LEFT, ORIGIN, PI, RED, RIGHT, Sphere, TEAL, UP, WHITE, YELLOW


class OpenGLGeometryZoo(Scene):
    def construct(self):
        shapes = [
            ogl.OpenGLCircle(radius=0.5),
            ogl.OpenGLSquare(side_length=0.8),
            ogl.OpenGLTriangle(),
            ogl.OpenGLRectangle(width=0.9, height=0.5),
            ogl.OpenGLRegularPolygon(n=5, radius=0.5),
            ogl.OpenGLEllipse(width=0.8, height=0.4),
            ogl.OpenGLAnnulus(inner_radius=0.2, outer_radius=0.5),
            ogl.OpenGLSector(radius=0.5, angle=PI / 2),
            ogl.OpenGLRoundedRectangle(width=0.8, height=0.5, corner_radius=0.1),
            ogl.OpenGLDot(radius=0.06),
            ogl.OpenGLLine(LEFT, RIGHT),
            ogl.OpenGLArrow(LEFT, RIGHT, buff=0.1),
            ogl.OpenGLDashedLine(LEFT, RIGHT),
            ogl.OpenGLVector(RIGHT),
            ogl.OpenGLCubicBezier(LEFT, UP, DOWN, RIGHT),
            ogl.OpenGLArc(radius=0.5, angle=PI),
            ogl.OpenGLArcBetweenPoints(LEFT, RIGHT),
            ogl.OpenGLCurvedArrow(LEFT, RIGHT),
            ogl.OpenGLPolygon(UP, LEFT, RIGHT),
        ]
        group = ogl.OpenGLVGroup(*shapes).arrange_in_grid(rows=4, cols=5, buff=0.45).scale(0.85)
        self.play(LaggedStart(*[Create(shape) for shape in group], lag_ratio=0.04), run_time=3)
        self.play(
            Indicate(group[0]),
            Rotate(group[1], PI / 2),
            Flash(group[2], color=YELLOW),
        )
        self.wait(1)


class OpenGLTextZoo(Scene):
    def construct(self):
        text = ogl.OpenGLText("OpenGL Text", font_size=48)
        math = ogl.OpenGLMathTex(r"E = mc^2", font_size=56).next_to(text, DOWN, buff=0.5)
        tex = ogl.OpenGLTex("Vulkan", font_size=48).next_to(math, DOWN, buff=0.5)
        self.play(Write(text), run_time=1.5)
        self.play(FadeIn(math), Create(tex))
        self.play(Transform(math, ogl.OpenGLMathTex(r"a^2 + b^2 = c^2").move_to(math)))
        self.play(Wiggle(text), Circumscribe(tex), run_time=1.5)
        self.wait(1)


class OpenGLParticleZoo(Scene):
    def construct(self):
        cloud = ogl.PointCloudDot(radius=1.2, density=30, color=BLUE)
        group = ogl.OpenGLPGroup(
            ogl.OpenGLPMPoint(),
            ogl.OpenGLPoint(),
        )
        true_dot = ogl.TrueDot(ORIGIN)
        group.arrange(RIGHT, buff=0.4).shift(DOWN)
        self.play(Create(cloud), run_time=2)
        self.play(FadeIn(group, true_dot))
        self.play(ApplyWave(cloud), run_time=2)
        self.wait(1)


class OpenGLSurfaceZoo(Scene):
    def construct(self):
        surface = ogl.OpenGLSurface(
            lambda u, v: np.array([u, v, 0.2 * np.sin(3 * u) * np.cos(3 * v)]),
            u_range=(-2, 2),
            v_range=(-2, 2),
            resolution=24,
            color=TEAL,
        )
        mesh = ogl.OpenGLSurfaceMesh(surface, resolution=(12, 12))
        mesh.set_stroke(WHITE, 1, opacity=0.4)
        texture = ogl.OpenGLTexturedSurface(
            Sphere(radius=1, resolution=(24, 24)),
            Path("example_scenes/assets/1280px-Whole_world_-_land_and_oceans.jpg"),
        )
        texture.next_to(surface, RIGHT, buff=0.8)
        self.play(FadeIn(surface), Create(mesh), FadeIn(texture), run_time=2)
        self.play(Rotate(surface, PI / 4), Rotate(texture, PI / 4), run_time=2)
        self.wait(1)


class OpenGLAnimationZoo(Scene):
    def construct(self):
        dot = ogl.OpenGLDot(LEFT * 3, color=YELLOW)
        square = ogl.OpenGLSquare(color=BLUE).next_to(dot, RIGHT, buff=1)
        circle = ogl.OpenGLCircle(color=RED).next_to(square, RIGHT, buff=1)
        self.play(
            Succession(
                GrowFromCenter(dot),
                MoveAlongPath(dot, ogl.ArcBetweenPoints(LEFT * 3, RIGHT * 3, angle=PI / 2)),
                ShrinkToCenter(dot),
            ),
            AnimationGroup(
                DrawBorderThenFill(square),
                FadeTransform(square.copy(), circle),
                lag_ratio=0.5,
            ),
            run_time=4,
        )
        self.play(
            Rotate(circle, PI),
            Indicate(square),
            Circumscribe(circle),
            run_time=2,
        )
        self.wait(1)


class OpenGLImageZoo(Scene):
    def construct(self):
        image = ogl.OpenGLImageMobject("example_scenes/assets/1280px-Whole_world_-_land_and_oceans.jpg")
        image.scale_to_fit_height(4)
        text = ogl.OpenGLText("ImageMobject", font_size=36).next_to(image, DOWN)
        self.play(FadeIn(image), Write(text))
        self.play(image.animate.scale(0.5).rotate(PI / 6))
        self.wait(1)


