from manim import *
from manim.opengl import OpenGLSurface


class SurfaceFacadeProbe(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-30 * DEGREES)
        s = OpenGLSurface(
            lambda u, v: np.array([u, v, 0.2 * np.sin(3 * u) * np.cos(3 * v)])
        )
        s.set_fill(BLUE, opacity=0.7)
        self.add(s)
        self.wait(0.2)
