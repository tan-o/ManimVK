import numpy as np

from manim import *
from manim.opengl import Mesh, Shader


def get_plane_mesh(context):
    shader = Shader(context, name="vertex_colors")
    attributes = np.zeros(
        18,
        dtype=[("in_vert", np.float32, (4,)), ("in_color", np.float32, (4,))],
    )
    attributes["in_vert"] = np.array(
        [
            [-1, -1, 0, 1], [-1, 1, 0, 1], [1, 1, 0, 1],
            [-1, -1, 0, 1], [1, -1, 0, 1], [1, 1, 0, 1],
            [0, -1, -1, 1], [0, -1, 1, 1], [0, 1, 1, 1],
            [0, -1, -1, 1], [0, 1, -1, 1], [0, 1, 1, 1],
            [-1, 0, -1, 1], [-1, 0, 1, 1], [1, 0, 1, 1],
            [-1, 0, -1, 1], [1, 0, -1, 1], [1, 0, 1, 1],
        ]
    )
    attributes["in_color"] = np.array(
        [
            [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1],
            [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1],
            [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1],
            [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1],
            [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1],
            [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1],
        ]
    )
    return Mesh(shader, attributes)


class Mesh3DProbe(Scene):
    def construct(self):
        s = Square(fill_opacity=0.5).shift(2 * RIGHT)
        self.add(s)
        sp = Sphere().shift(2 * LEFT)
        self.add(sp)
        mesh = get_plane_mesh(self.renderer.context)
        mesh.model_matrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        self.add(mesh)
        self.wait(0.5)
