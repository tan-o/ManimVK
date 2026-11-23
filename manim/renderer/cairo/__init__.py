"""Cairo rendering backend (reference renderer for the golden comparison)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .renderer import CairoRenderer

__all__ = ["CairoRenderer"]


def __getattr__(name: str) -> Any:
    if name != "CairoRenderer":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Import on first use: the renderer imports much of manim.
    from .renderer import CairoRenderer

    globals()[name] = CairoRenderer
    return CairoRenderer
