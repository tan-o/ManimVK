from manim import *


class VulkanDemo(Scene):
    def construct(self):
        title = Text("Manim on Vulkan", font_size=56).to_edge(UP)
        subtitle = Text("RTX 3050 Ti · native Vulkan surface", font_size=28, color=GREY_B)
        subtitle.next_to(title, DOWN, buff=0.3)

        circle = Circle(radius=1.0, color=BLUE, fill_opacity=0.8).shift(LEFT * 2.4)
        square = Square(side_length=2.0, color=GREEN, fill_opacity=0.7).shift(RIGHT * 2.4)

        formula = MathTex("E = mc^2", font_size=64).shift(DOWN * 2.3)

        self.play(Write(title), FadeIn(subtitle, shift=DOWN * 0.2), run_time=2.0)
        self.play(Create(circle), Create(square), run_time=2.0)
        self.play(
            Transform(square, circle.copy().shift(RIGHT * 2.4)),
            Indicate(title, color=YELLOW),
            run_time=2.0,
        )
        self.play(Write(formula), run_time=2.0)
        self.play(
            Rotate(VGroup(circle, square), angle=PI / 2),
            FadeOut(subtitle),
            run_time=2.0,
        )
        self.wait(1.0)
        self.play(FadeOut(VGroup(title, circle, square, formula)), run_time=1.0)
