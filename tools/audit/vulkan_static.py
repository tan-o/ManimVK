from manim import *


class StaticScene(Scene):
    def construct(self):
        circle = Circle(radius=1.2, color=BLUE, fill_opacity=0.8)
        title = Text("Vulkan on Windows", font_size=44).shift(DOWN * 2)
        formula = MathTex("E = mc^2", font_size=48).shift(UP * 2)
        self.add(circle, title, formula)
        self.wait(0.2)
