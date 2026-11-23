import time

from manim import *
from manim.renderer.skia.renderer import SkiaRenderer

config.renderer = "vulkan"
config.live_preview = True
config.quality = "low_quality"
config.pixel_width = 854
config.pixel_height = 480
config.frame_rate = 15


class WindowScene(Scene):
    def construct(self):
        self.add(
            Circle(radius=1.2, color=BLUE, fill_opacity=0.8),
            Text("Vulkan on Windows", font_size=44).shift(DOWN * 2),
            MathTex("E = mc^2", font_size=48).shift(UP * 2),
        )


scene = WindowScene(renderer=SkiaRenderer())
scene.construct()
scene.renderer.update_frame(scene)
scene.renderer.window.swap_buffers()
print("WINDOW_READY", flush=True)
time.sleep(8)
scene.renderer.window.close()

