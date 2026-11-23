from manim import *


class Live(Scene):
    def construct(self):
        self.add(
            Circle(radius=1.2, color=BLUE, fill_opacity=0.8),
            Text("Vulkan on Windows", font_size=44).shift(DOWN * 2),
        )
        self.wait(5)
