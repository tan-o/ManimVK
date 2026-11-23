from manim import Circle, Scene, Text
import numpy as np


class S(Scene):
    def construct(self):
        self.add(Circle(), Text("ok"))


s = S()
r = s.renderer
r.update_frame(s)
a = r.get_frame()
b = r.get_frame()
print(a.shape, b.shape, np.array_equal(a, b), a.mean(), b.mean())
