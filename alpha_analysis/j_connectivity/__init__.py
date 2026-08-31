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
from .critical_curves import (
    CriticalCurveConfig,
    CriticalCurveError,
    CriticalCurveReport,
    CriticalCurveStatus,
    CriticalCurves,
    CriticalKind,
    CriticalPolyline,
    extract_critical_curves,
)
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
from .mesh_cut import (
    ConstrainedCutConfig,
    ConstrainedCutError,
    CutSurface,
    CutEvent,
    CutTransitionPort,
    cut_surface_at_transitions,
    cut_surface_at_transition_arcs,
    load_cut_surface,
    save_cut_surface,
)
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
from .surface_refine import (
    SurfaceDownsamplingConfig,
    SurfaceDownsamplingError,
    SurfaceDownsamplingReport,
    SurfaceDownsamplingResult,
    downsample_surface,
)
from .surface_data import (
    SurfaceData,
    SurfaceEdgeIndicators,
    SurfaceRefinementError,
    SurfaceRefinementConfig,
    SurfaceRefinementLevel,
    SurfaceRefinementReport,
    SurfaceRefinementResult,
    evaluate_surface_data,
    itineraries_are_continuous,
    refine_surface_data,
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
from .transitions import (
    TransitionCurve,
    TransitionMappingConfig,
    TransitionPort,
    map_transitions,
    map_transitions_budget_sweep,
)
from .well_trace import (
    WellProfile,
    WellTraceConfig,
    sample_well_profile,
    trace_regular_well,
)
from .transition_events import (
    ContactLocalizationConfig,
    ContactBracket,
    TransitionEvent,
    TransitionArc,
    LocalizedTransitions,
    localize_transition_contacts,
    build_transition_arcs,
)

__all__ = [
    "ContactLocalizationConfig",
    "ContactBracket",
    "TransitionEvent",
    "TransitionArc",
    "LocalizedTransitions",
    "localize_transition_contacts",
    "build_transition_arcs",
    "CutEvent",
    "cut_surface_at_transition_arcs",
    "BackgroundMesh",
    "BackgroundMeshConfig",
    "ConnectivityConfig",
    "ConstrainedCutConfig",
    "ConstrainedCutError",
    "CutSurface",
    "CutTransitionPort",
    "CriticalCurveConfig",
    "CriticalCurveError",
    "CriticalCurveReport",
    "CriticalCurveStatus",
    "CriticalCurves",
    "CriticalKind",
    "CriticalPolyline",
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
    "SurfaceDownsamplingConfig",
    "SurfaceDownsamplingError",
    "SurfaceDownsamplingReport",
    "SurfaceDownsamplingResult",
    "SurfaceExtraction",
    "SurfaceExtractionConfig",
    "SurfaceExtractionError",
    "SurfaceMesh",
    "SurfaceData",
    "SurfaceEdgeIndicators",
    "SurfaceRefinementConfig",
    "SurfaceRefinementError",
    "SurfaceRefinementLevel",
    "SurfaceRefinementReport",
    "SurfaceRefinementResult",
    "SurfaceStatus",
    "TraceStatus",
    "TransitionStatus",
    "TransitionCurve",
    "TransitionMappingConfig",
    "TransitionPort",
    "WellTrace",
    "WellTraceConfig",
    "WellProfile",
    "UniformSourceProfile",
    "compute_denominator",
    "cut_surface_at_transitions",
    "downsample_surface",
    "evaluate_surface_data",
    "extract_critical_curves",
    "denominator_convergence",
    "find_global_B_bounds",
    "itineraries_are_continuous",
    "load_cut_surface",
    "refine_surface_data",
    "optional_import",
    "signed_tetrahedron_volumes",
    "surface_flux",
    "tetrahedron_quality",
    "sample_well_profile",
    "save_cut_surface",
    "trace_regular_well",
    "map_transitions",
    "map_transitions_budget_sweep",
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
