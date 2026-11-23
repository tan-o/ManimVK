from IPython.testing.globalipapp import get_ipython

ip = get_ipython()
ip.run_cell("from manim import *", silent=False)
ip.run_cell(
    "%%manim -ql --format=mp4 --disable_caching JupyterVideoProbe\n"
    "class JupyterVideoProbe(Scene):\n"
    "    def construct(self):\n"
    "        c = Circle()\n"
    "        self.play(Create(c), run_time=1)\n"
    "        self.wait(0.2)\n",
    silent=False,
)
print("JUPYTER_VIDEO_PROBE_DONE")
