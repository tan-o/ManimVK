"""The compiled modules, checked against the interface this code expects.

``manim_core`` and ``manim_vk`` are built separately from the Python package
(``maturin develop -r`` in ``native/<name>``), so an old build can sit next to
new Python code.  Each module reports ``API_VERSION``; a mismatch stops the
import here with the command that fixes it, instead of failing later on a
missing method or a changed signature.
"""

from __future__ import annotations

import manim_core
import manim_vk

__all__: list[str] = []

# Bump with the API_VERSION constant in native/<name>/src/lib.rs.
_EXPECTED = {manim_core: 1, manim_vk: 1}

for _module, _expected in _EXPECTED.items():
    _found = getattr(_module, "API_VERSION", None)
    if _found != _expected:
        raise ImportError(
            f"{_module.__name__} has interface version {_found}, this manim needs "
            f"{_expected}: rebuild it with `maturin develop -r` in "
            f"native/{_module.__name__}."
        )
