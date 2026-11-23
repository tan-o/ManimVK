from __future__ import annotations

import cProfile
import os
import pstats
import time

from manim import Circle, Rotate, Scene, PI, tempconfig, VGroup

class PerfScene(Scene):
    def construct(self) -> None:
        colors = ["#5B8FF9", "#61DDAA", "#F6BD16", "#E8684A"]
        group = VGroup(
            *[
                Circle(radius=0.035, color=colors[index % len(colors)]).shift(
                    [-3.2 + 0.16 * (index % 41), -2.0 + 0.16 * (index // 41), 0.0]
                )
                for index in range(41 * 25)
            ]
        )
        self.add(group)
        self.play(Rotate(group, PI / 2), run_time=2.5)

with tempconfig(
    {
        "quality": "low_quality",
        "renderer": "vulkan",
        "format": "mp4",
        "media_dir": os.path.join("media", "perf_profile"),
        "disable_caching": True,
        "progress_bar": "none",
    }
):
    profile = cProfile.Profile()
    started = time.perf_counter()
    profile.enable()
    PerfScene().render()
    profile.disable()
    profile.dump_stats(os.path.join("media", "perf_profile.prof"))
    elapsed = time.perf_counter() - started

print(f"elapsed={elapsed:.3f}s")
pstats.Stats(profile).sort_stats("cumtime").print_stats(35)
