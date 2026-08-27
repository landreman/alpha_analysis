"""Infrastructure for the J-connectivity calculation (DESIGN.md §14).

The numerical core deliberately imports only the base dependencies. Optional mesh,
visualization, graph, and I/O packages are requested at their use sites.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from .background_mesh import (
    BackgroundMeshConfig,
    GmshBackgroundMeshBackend,
    GmshBackgroundMeshConfig,
    StructuredPrismMeshBackend,
    signed_tetrahedron_volumes,
    tetrahedron_quality,
)
from .config import ConnectivityConfig
from .denominator import (
    BoundsConfig,
    DenominatorConfig,
    DenominatorConvergence,
    DenominatorEstimate,
    GlobalBBounds,
    SourceProfile,
    UniformSourceProfile,
    compute_denominator,
    denominator_convergence,
    find_global_B_bounds,
)
from .field import BoozerFieldLike
from .surface_extract import (
    MarchingTetrahedraExtractor,
    PyVistaSurfaceExtractor,
    SurfaceCurveMesh,
    SurfaceExtraction,
    SurfaceExtractionConfig,
    SurfaceExtractionError,
    SurfaceMesh,
    surface_flux,
)
from .types import (
    BackgroundMesh,
    FloodFillStatus,
    QuadratureStatus,
    RunMetadata,
    SurfaceStatus,
    TraceStatus,
    TransitionStatus,
    WellTrace,
)
from .well_trace import (
    WellProfile,
    WellTraceConfig,
    sample_well_profile,
    trace_regular_well,
)

__all__ = [
    "BackgroundMesh",
    "BackgroundMeshConfig",
    "ConnectivityConfig",
    "BoundsConfig",
    "BoozerFieldLike",
    "DenominatorConfig",
    "DenominatorConvergence",
    "DenominatorEstimate",
    "FloodFillStatus",
    "GlobalBBounds",
    "GmshBackgroundMeshBackend",
    "GmshBackgroundMeshConfig",
    "MarchingTetrahedraExtractor",
    "PyVistaSurfaceExtractor",
    "QuadratureStatus",
    "RunMetadata",
    "SourceProfile",
    "StructuredPrismMeshBackend",
    "SurfaceCurveMesh",
    "SurfaceExtraction",
    "SurfaceExtractionConfig",
    "SurfaceExtractionError",
    "SurfaceMesh",
    "SurfaceStatus",
    "TraceStatus",
    "TransitionStatus",
    "WellTrace",
    "WellTraceConfig",
    "WellProfile",
    "UniformSourceProfile",
    "compute_denominator",
    "denominator_convergence",
    "find_global_B_bounds",
    "optional_import",
    "signed_tetrahedron_volumes",
    "surface_flux",
    "tetrahedron_quality",
    "sample_well_profile",
    "trace_regular_well",
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
