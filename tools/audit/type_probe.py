from manim import Scene, Square, tempconfig

with tempconfig({"renderer": "opengl"}):
    square = Square()
    print(type(square), square.__class__.__mro__)
    scene = Scene()
    scene.add(square)
    print("after add", type(square))
