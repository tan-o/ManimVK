from manim import Circle, Square, Text, MathTex, VMobject

for obj in [Circle(), Square(), Text("hello"), MathTex("E=mc^2")]:
    print(
        type(obj).__name__,
        "fill",
        obj.get_fill_rgbas().shape,
        float(obj.get_fill_rgbas()[0, 3]) if len(obj.get_fill_rgbas()) else None,
        "stroke",
        obj.get_stroke_width(),
        obj.get_stroke_width(background=True),
        obj.get_stroke_rgbas().shape,
    )
