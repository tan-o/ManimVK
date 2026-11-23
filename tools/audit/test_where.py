import manim


def test_where():
    print("MANIM_FILE", manim.__file__)
    print("SQUARE", manim.Square, manim.Square.__module__)
    with manim.tempconfig({"renderer": "opengl"}):
        s = manim.Square()
        print("SQUARE_INST", type(s), isinstance(s, manim.Mobject))
