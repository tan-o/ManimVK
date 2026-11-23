"""The moderngl/OpenGL backend is retired in this fork.

``config.renderer = "opengl"`` and ``--renderer opengl`` now route to the
Vulkan/Skia renderer, so these tests assert the old OpenGL class hierarchy and
cannot describe the current runtime.  Vulkan regression coverage lives in
``tests/test_vulkan``.  Keep the collection skipped rather than deleting the
historical tests.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    reason = "legacy moderngl backend retired; see tests/test_vulkan"
    for item in items:
        item.add_marker(pytest.mark.skip(reason=reason))
