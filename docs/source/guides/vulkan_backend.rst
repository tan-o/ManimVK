Vulkan / Skia backend
====================

ManimVK renders the classic Manim object model through Skia's Ganesh backend on
top of Vulkan, its only renderer.  Scene scripts written for Manim Community run
unchanged.

Default behaviour
-----------------

The OpenGL renderer, the ``OpenGL*`` mobject family and the GLSL
``Shader``/``Mesh`` API are removed; custom effects use :class:`.SkSLMobject`.
``config.renderer`` and ``--renderer`` still accept ``"cairo"`` and
``"vulkan"``, as Manim Community scripts and configs use them; both draw with
Skia/Vulkan.

The renderer keeps one Vulkan device per process and one Skia Ganesh context per
device.  Vector paths, text, images, point clouds, background-filled
shapes and ``MultiCamera`` sub-views all draw and composite on the GPU.  A frame
is copied to host memory only when a video writer, still image, or preview
needs pixels.

Selecting a GPU
---------------

Set ``MANIM_VK_DEVICE`` to a case-insensitive substring of the Vulkan adapter
name, for example::

    set MANIM_VK_DEVICE=RTX 3050
    manim -qm scene.py Scene

Without the variable, a discrete GPU is preferred, then an integrated GPU, then a
CPU device such as lavapipe.  If no Vulkan physical device exists the renderer
raises ``RuntimeError`` with installation guidance.

Depth in 3D scenes
------------------

In a :class:`.ThreeDScene`, mobjects with ``shade_in_3d`` and mobjects marked
with :meth:`.Mobject.apply_depth_test` are drawn back to front.  A depth-tested
:class:`.VMobject` is ordered per subpath, so one path holding thousands of
short strokes, such as the hairs of a vector field on a sphere, is partly
hidden behind the sphere and partly in front of it, while still drawing in a
few dozen GPU calls.

:class:`.MeshSurface` draws a :class:`.Surface` as one smoothly lit triangle
mesh and :class:`.TexturedSurface` stretches an image over it, optionally with
a second image on the side facing away from the light.  Their triangles take
part in the same depth ordering; a translucent mesh surface hides depth-tested
mobjects behind it, as a depth buffer would.  These are drawn by the Vulkan
renderer only.

Custom shaders
--------------

Legacy GLSL ``Shader`` objects are not compiled.  Use
:class:`~manim.mobject.sksl.SkSLMobject` with a SkSL source string instead::

    from manim import Scene, SkSLMobject

    class ShaderScene(Scene):
        def construct(self):
            self.add(SkSLMobject("""
                uniform float2 u_resolution;
                half4 main(float2 fragCoord) {
                    float2 uv = fragCoord / u_resolution;
                    return half4(uv.x, uv.y, 0.5, 1.0);
                }
            """))

The backend injects ``u_resolution`` when it is not supplied.  Invalid SkSL
raises ``RuntimeError`` with the compiler diagnostic.

Known limitations
-----------------

- The preview window is read-only.  ``interactive_embed()`` warns and returns;
  mouse and keyboard events are not forwarded.
- ``fullscreen``, ``window_position`` and ``window_monitor`` are applied by the
  native preview window; HiDPI scaling may differ from Manim Community's.
- ``ImageMobject.pixel_array`` can be written in place, as in Manim Community:
  writes by index or with in-place operators upload the new pixels.
  ``np.copyto`` and ``out=`` arguments are not noticed; assign
  ``pixel_array`` after using them.
- Prebuilt native wheels currently target Windows x64.  Building on Linux
  requires a Vulkan loader and the Skia binary cache; macOS additionally needs
  MoltenVK.  See ``native/manim_vk/README.md`` for build commands.
