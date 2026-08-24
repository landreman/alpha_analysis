"""Infrastructure for the J-connectivity calculation (DESIGN.md §14).

The numerical core deliberately imports only the base dependencies. Optional mesh,
visualization, graph, and I/O packages are requested at their use sites.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from .config import ConnectivityConfig
from .types import (
    FloodFillStatus,
    QuadratureStatus,
    RunMetadata,
    SurfaceStatus,
    TraceStatus,
    TransitionStatus,
)

__all__ = [
    "ConnectivityConfig",
    "FloodFillStatus",
    "QuadratureStatus",
    "RunMetadata",
    "SurfaceStatus",
    "TraceStatus",
    "TransitionStatus",
    "optional_import",
]


def optional_import(module_name: str, *, extra: str) -> ModuleType:
    """Import an optional module or explain which project extra supplies it.

    This preserves the base-only import contract in DESIGN.md §§19.2 and 23,
    while making an optional-feature failure actionable.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        raise ImportError(
            f"Optional dependency '{module_name}' is required for this feature. "
            f"Install it with: python -m pip install 'alpha-analysis[{extra}]'."
        ) from error
