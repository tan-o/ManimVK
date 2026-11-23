from manim import *
from manim.opengl import *


class AliasProbe(Scene):
    def construct(self):
        c = OpenGLCircle(color=BLUE, fill_opacity=0.5)
        s = OpenGLSquare(color=RED).shift(RIGHT * 2)
        a = OpenGLArrow(LEFT * 2, RIGHT * 0.5, color=GREEN)
        d = OpenGLDot(color=YELLOW).shift(UP)
        self.add(c, s, a, d)
        self.play(Create(OpenGLLine(LEFT * 3, RIGHT * 3, color=WHITE)))
