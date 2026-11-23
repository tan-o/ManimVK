from IPython.testing.globalipapp import get_ipython

ip = get_ipython()
ip.run_cell("from manim import *", silent=False)
ip.run_cell(
    "%%manim -ql -s --format=png --disable_caching JupyterProbe\n"
    "class JupyterProbe(Scene):\n"
    "    def construct(self):\n"
    "        self.add(Circle(), Text('Jupyter Vulkan'))\n",
    silent=False,
)
print("JUPYTER_PROBE_DONE")
