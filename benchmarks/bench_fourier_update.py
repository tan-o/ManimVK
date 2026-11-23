"""Per-frame updater cost of the 100-vector Fourier scene, without rendering.

Usage: python benchmarks/bench_fourier_update.py [n_vectors] [frames]
The 30 fps live-preview budget is 33 ms per frame for update + draw + present.
"""
import sys, time
import numpy as np
from manim import Vector, Circle, VGroup, TracedPath, ORIGIN, WHITE, BLUE, config

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 60
rng = np.random.default_rng(0)
freqs = np.array([0] + [k for i in range(1, N) for k in ((i + 1) // 2 * (1 if i % 2 else -1),)])[:N]
coefs = (rng.normal(size=N) + 1j * rng.normal(size=N)) / (1 + np.abs(freqs)) ** 1.5
slow = 0.1
t_state = {"t": 0.0}

vectors, circles = [], []
last = None
for k, (f, c) in enumerate(zip(freqs, coefs)):
    v = Vector(RIGHT := np.array([1.0, 0, 0]), buff=0, stroke_width=2, tip_length=0.15,
               max_tip_length_to_length_ratio=0.2, color=WHITE)
    v.freq, v.coef = f, c
    v.center_func = (lambda prev=last: ORIGIN if prev is None else prev.get_end())
    def update_vector(v, dt=0):
        time_ = t_state["t"]
        z = v.coef * np.exp(1j * 2 * np.pi * v.freq * time_ * slow)
        v.set_length(max(abs(z), 1e-9))
        v.set_angle(np.angle(z))
        v.shift(v.center_func() - v.get_start())
    v.add_updater(update_vector)
    vectors.append(v); last = v
for v in vectors:
    c = Circle(stroke_width=1, color=BLUE)
    c.vector = v
    def update_circle(c, dt=0):
        c.width = 2 * max(c.vector.get_length(), 1e-6)
        c.move_to(c.vector.get_start())
    c.add_updater(update_circle)
    circles.append(c)
path = TracedPath(vectors[-1].get_end)
mobs = vectors + circles + [path]

def step():
    for m in mobs:
        m.update(1 / 60)

step()
t0 = time.perf_counter()
for i in range(FRAMES):
    t_state["t"] += 1 / 60
    step()
dt = (time.perf_counter() - t0) / FRAMES
print(f"N={N} update per frame: {dt*1e3:.1f} ms  ({1/dt:.1f} fps upper bound)")
