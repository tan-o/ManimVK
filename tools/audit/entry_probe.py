from manim import Circle, Scene, config, tempconfig


class EntryProbe(Scene):
    def construct(self):
        self.add(Circle())


if __name__ == "__main__":
    with tempconfig(
        {
            "renderer": "opengl",
            "quality": "low_quality",
            "media_dir": "media/entry_probe",
            "format": "png",
            "progress_bar": "none",
        }
    ):
        scene = EntryProbe()
        print(type(scene.renderer).__name__, type(scene.camera).__name__, config.renderer)
        scene.render()
