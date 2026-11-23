from manim import BLUE, Create, FadeIn, PointCloudDot, Scene


class PointCloudCreateProbe(Scene):
    def construct(self):
        cloud = PointCloudDot(radius=1.2, density=30, color=BLUE)
        self.play(Create(cloud), run_time=2)
        self.wait(1)


class PointCloudFadeProbe(Scene):
    def construct(self):
        cloud = PointCloudDot(radius=1.2, density=30, color=BLUE)
        self.play(FadeIn(cloud), run_time=2)
        self.wait(1)


class PointCloudWaveProbe(Scene):
    def construct(self):
        cloud = PointCloudDot(radius=1.2, density=30, color=BLUE)
        self.add(cloud)
        self.play(Create(cloud), run_time=1)
        self.play(__import__("manim").ApplyWave(cloud), run_time=2)
        self.wait(1)
