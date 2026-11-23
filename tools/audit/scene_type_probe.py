from manim import Animation, Scene, Square, tempconfig


with tempconfig(
    {
        "renderer": "opengl",
        "format": "none",
        "live_preview": False,
        "pixel_width": 32,
        "pixel_height": 16,
        "frame_rate": 4,
        "disable_caching": True,
        "progress_bar": "none",
    }
):
    scene = Scene(skip_animations=False)
    scene.renderer._skips_all_plays = False
    scene.add(Square())
    print("mobjects", [type(m) for m in scene.mobjects])
    print("meshes", [type(m) for m in scene.meshes])
    print("renderer", type(scene.renderer))
