import numpy as np

from manim import *
from manim.opengl import Mesh, Shader


class MeshProbe(Scene):
    def construct(self):
        attributes = np.zeros(
            6,
            dtype=[("in_vert", np.float32, (4,)), ("in_color", np.float32, (4,))],
        )
        attributes["in_vert"] = np.array(
            [
                [-2, -1, 0, 1],
                [2, -1, 0, 1],
                [2, 1, 0, 1],
                [-2, -1, 0, 1],
                [2, 1, 0, 1],
                [-2, 1, 0, 1],
            ]
        )
        attributes["in_color"] = np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
                [1, 0, 0, 1],
                [0, 0, 1, 1],
                [1, 1, 0, 1],
            ]
        )
        mesh = Mesh(Shader(None, name="vertex_colors"), attributes)
        self.add(mesh)
