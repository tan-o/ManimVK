from manim import *


class Smoke(Scene):
    def construct(self):
        circle = Circle(radius=1.2, color=BLUE, fill_opacity=0.8)
        title = Text("Vulkan on Windows", font_size=44).shift(DOWN * 2)
        formula = MathTex("E = mc^2", font_size=48).shift(UP * 2)
        self.play(Create(circle), run_time=1.2)
        self.play(Write(title), run_time=1.2)
        self.play(Write(formula), run_time=1.0)
        self.wait(0.5)


class Surface3D(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)
        axes = ThreeDAxes()
        surface = Surface(
            lambda u, v: axes.c2p(u, v, 0.4 * np.sin(u) * np.cos(v)),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(24, 24),
            fill_opacity=0.75,
        )
        self.add(axes, surface)
        self.begin_ambient_camera_rotation(rate=0.08)
        self.wait(1.0)
