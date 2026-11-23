from manim import FadeIn, Scene, Square


class ParallelEncodingCacheScene(Scene):
    def construct(self):
        for index in range(6):
            square = Square(side_length=0.2 + index / 100)
            self.play(FadeIn(square), run_time=0.1)
            self.clear()

        self.play(FadeIn(Square(side_length=0.75)), run_time=0.1)
        self.clear()
        self.play(FadeIn(Square(side_length=0.75)), run_time=0.1)
