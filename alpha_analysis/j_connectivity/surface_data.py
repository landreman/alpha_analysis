"""Surface-wide well data and adaptive refinement (DESIGN.md §§8.4 and 9.5).

This module is the NumPy-only layer between level-surface extraction and later
critical-curve construction. It retains every trace status, compares authoritative
unquantized return itineraries, and refines a conforming surface without crossing or
hiding a candidate return-map discontinuity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .field import BoozerFieldLike
from .surface_extract import (
    SurfaceExtractionConfig,
    SurfaceExtractionError,
    SurfaceMesh,
    _canonicalize_point,
    _evaluate_B,
    _physical_g,
    _project_to_level_near,
)
from .surface_refine import _periodic_edge_length, _periodic_midpoint
from .types import FloatArray, IntArray, TraceStatus, WellTrace
from .well_trace import WellTraceConfig, trace_regular_well


class SurfaceRefinementError(RuntimeError):
    """A pitch surface could not be refined without violating §8.4."""


@dataclass(frozen=True)
class SurfaceRefinementConfig:
    """Controls action and return-itinerary refinement (DESIGN.md §8.4).

    The action and bounce-time absolute tolerances have the length units of
    ``A`` and ``K`` from §4.2; their relative tolerances are dimensionless.
    ``itinerary_tolerance`` is dimensionless: angles and lifted extrema
    positions are normalized by their periods and extrema heights by ``b``.
    ``B_tolerance`` has magnetic-field units and ``g_tolerance`` has the units
    of physical ``b dot grad B``. New vertices may move at most
    ``max_projection_distance_ratio`` times their parent edge length while
    being projected back to ``B=b``.
    """

    max_levels: int = 3
    action_interpolation_rtol: float = 1.0e-3
    action_interpolation_atol: float = 1.0e-8
    bounce_time_interpolation_rtol: float = 1.0e-3
    bounce_time_interpolation_atol: float = 1.0e-8
    itinerary_tolerance: float = 0.25
    B_tolerance: float = 1.0e-10
    g_tolerance: float = 1.0e-10
    merge_tolerance: float = 1.0e-10
    max_projection_distance_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.max_levels < 0:
            raise ValueError("max_levels must be nonnegative")
        for name in (
            "action_interpolation_rtol",
            "action_interpolation_atol",
            "bounce_time_interpolation_rtol",
            "bounce_time_interpolation_atol",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "itinerary_tolerance",
            "B_tolerance",
            "g_tolerance",
            "merge_tolerance",
            "max_projection_distance_ratio",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class SurfaceData:
    """Authoritative per-vertex well data on one incoming pitch surface.

    Angles are radians. ``action_length`` and ``bounce_time_length`` contain
    the half-bounce ``A`` and ``K`` of DESIGN.md §4.2 in length units.
    Non-regular traces remain present with their explicit ``TraceStatus`` and
    ``NaN`` numerical values; no failed vertex is assigned zero action or
    silently removed (DESIGN.md §21.2).
    """

    surface: SurfaceMesh
    traces: tuple[WellTrace, ...]
    action_length: FloatArray
    bounce_time_length: FloatArray
    zeta_out_unwrapped: FloatArray
    field_period_count: IntArray
    n_internal_maxima: IntArray
    itinerary_hash: np.ndarray
    trace_status: IntArray

    def __post_init__(self) -> None:
        n_points = len(self.surface.points)
        if len(self.traces) != n_points:
            raise ValueError("surface data need one trace per point")
        arrays = {
            "action_length": np.asarray(self.action_length, dtype=np.float64),
            "bounce_time_length": np.asarray(self.bounce_time_length, dtype=np.float64),
            "zeta_out_unwrapped": np.asarray(self.zeta_out_unwrapped, dtype=np.float64),
            "field_period_count": np.asarray(self.field_period_count, dtype=np.int64),
            "n_internal_maxima": np.asarray(self.n_internal_maxima, dtype=np.int64),
            "itinerary_hash": np.asarray(self.itinerary_hash, dtype=np.uint64),
            "trace_status": np.asarray(self.trace_status, dtype=np.int64),
        }
        if any(values.shape != (n_points,) for values in arrays.values()):
            raise ValueError("surface scalar arrays must have one value per point")
        for name, values in arrays.items():
            object.__setattr__(self, name, values)

    @property
    def regular(self) -> np.ndarray:
        """Point mask for successful first-return traces."""
        return self.trace_status == TraceStatus.REGULAR.value

    def to_pyvista(self):
        """Return an optional surface view with named point data (§§8.1, 17.1)."""
        view = self.surface.to_pyvista()
        view.point_data["A action length [length]"] = self.action_length
        view.point_data["K bounce-time length [length]"] = self.bounce_time_length
        view.point_data["unwrapped exit zeta [rad]"] = self.zeta_out_unwrapped
        view.point_data["field-period count [integer]"] = self.field_period_count
        view.point_data["internal maximum count [integer]"] = self.n_internal_maxima
        view.point_data["itinerary hash [uint64]"] = self.itinerary_hash
        view.point_data["trace status [TraceStatus code]"] = self.trace_status
        return view


@dataclass(frozen=True)
class SurfaceRefinementLevel:
    """Machine-readable convergence diagnostics for one §8.4 mesh level."""

    level: int
    point_count: int
    triangle_count: int
    regular_trace_count: int
    failed_trace_count: int
    refined_edge_count: int
    candidate_edge_count: int
    max_action_interpolation_error: float
    max_bounce_time_interpolation_error: float
    max_candidate_s_width: float
    max_B_residual: float


@dataclass(frozen=True)
class SurfaceRefinementReport:
    """Surface-refinement convergence history required by DESIGN.md §21.3."""

    levels: tuple[SurfaceRefinementLevel, ...]


@dataclass(frozen=True)
class SurfaceEdgeIndicators:
    """Final edge-local interpolation and return-sheet indicators (§8.4).

    ``edges`` contains surface point IDs. Action and bounce-time errors have
    the length units of ``A`` and ``K``; they are ``NaN`` when any of the two
    endpoint traces or the projected midpoint trace is non-regular. In that
    case ``itinerary_candidate`` remains true, so failure cannot erase a
    possible sheet boundary (DESIGN.md §21.2).
    """

    edges: IntArray
    action_interpolation_error: FloatArray
    bounce_time_interpolation_error: FloatArray
    itinerary_candidate: np.ndarray

    def __post_init__(self) -> None:
        edges = np.asarray(self.edges, dtype=np.int64)
        action = np.asarray(self.action_interpolation_error, dtype=np.float64)
        bounce_time = np.asarray(self.bounce_time_interpolation_error, dtype=np.float64)
        candidates = np.asarray(self.itinerary_candidate, dtype=bool)
        if edges.ndim != 2 or edges.shape[1:] != (2,):
            raise ValueError("edge indicators require edges with shape (n_edges, 2)")
        n_edges = len(edges)
        if any(
            values.shape != (n_edges,) for values in (action, bounce_time, candidates)
        ):
            raise ValueError("edge indicator arrays must have one value per edge")
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "action_interpolation_error", action)
        object.__setattr__(self, "bounce_time_interpolation_error", bounce_time)
        object.__setattr__(self, "itinerary_candidate", candidates)


@dataclass(frozen=True)
class SurfaceRefinementResult:
    """Refined pitch surface, its vertex data, and convergence diagnostics."""

    surface: SurfaceMesh
    data: SurfaceData
    edge_indicators: SurfaceEdgeIndicators
    report: SurfaceRefinementReport


@dataclass(frozen=True)
class _EdgeSample:
    point: FloatArray
    B: float
    g: float
    trace: WellTrace
    action_error: float
    bounce_time_error: float
    candidate: bool


def _logical_coordinates(
    points: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    s = np.sum(points[:, :2] ** 2, axis=1)
    theta = np.arctan2(points[:, 1], points[:, 0])
    theta[s == 0.0] = 0.0
    return s, theta, points[:, 2]


def evaluate_surface_data(
    surface: SurfaceMesh,
    field: BoozerFieldLike,
    trace_config: WellTraceConfig | None = None,
) -> SurfaceData:
    """Trace every pitch-surface vertex and retain failures (DESIGN.md §6 step 4).

    ``surface`` is the incoming ``g<=0`` half of ``B=b``. Every point is
    converted from axis-regular ``(x,y,zeta)`` to reduced
    ``(s,theta,zeta)`` and passed through the production first-return tracer.
    Boundary or unresolved points therefore receive explicit non-regular
    statuses and ``NaN`` action rather than being skipped.
    """
    cfg = WellTraceConfig() if trace_config is None else trace_config
    s, theta, zeta = _logical_coordinates(surface.points)
    traces = tuple(
        trace_regular_well(field, surface.level, np.array(point), cfg)
        for point in zip(s, theta, zeta)
    )
    return SurfaceData(
        surface=surface,
        traces=traces,
        action_length=np.array([trace.action_length for trace in traces]),
        bounce_time_length=np.array([trace.bounce_time_length for trace in traces]),
        zeta_out_unwrapped=np.array([trace.zeta_out_unwrapped for trace in traces]),
        field_period_count=np.array(
            [trace.field_period_count for trace in traces], dtype=np.int64
        ),
        n_internal_maxima=np.array(
            [trace.n_internal_maxima for trace in traces], dtype=np.int64
        ),
        itinerary_hash=np.array(
            [trace.itinerary_hash for trace in traces], dtype=np.uint64
        ),
        trace_status=np.array([trace.status.value for trace in traces], dtype=np.int64),
    )


def _periodic_difference(first: float, second: float, period: float) -> float:
    return float((first - second + 0.5 * period) % period - 0.5 * period)


def itineraries_are_continuous(
    first: WellTrace,
    second: WellTrace,
    period: float,
    tolerance: float,
) -> bool:
    """Compare unquantized return itineraries as required by DESIGN.md §9.5.

    The quick hash is deliberately ignored. A pair is continuous only when
    both traces are regular and their lifted period count, reduced exit,
    ordered extrema kinds, extrema locations relative to the exit, and extrema
    heights agree within the dimensionless tolerance. Non-regular data remain
    refinement candidates rather than being treated as a continuous sheet.
    """
    if (
        first.status is not TraceStatus.REGULAR
        or second.status is not TraceStatus.REGULAR
    ):
        return False
    if first.field_period_count != second.field_period_count:
        return False
    if not np.array_equal(first.extrema_kind, second.extrema_kind):
        return False
    theta_difference = abs(
        _periodic_difference(
            first.q_out_reduced[1], second.q_out_reduced[1], 2.0 * np.pi
        )
        / (2.0 * np.pi)
    )
    zeta_difference = abs(
        _periodic_difference(first.q_out_reduced[2], second.q_out_reduced[2], period)
        / period
    )
    lifted_exit_difference = (
        abs(
            (first.zeta_out_unwrapped - first.q_in[2])
            - (second.zeta_out_unwrapped - second.q_in[2])
        )
        / period
    )
    if max(theta_difference, zeta_difference, lifted_exit_difference) > tolerance:
        return False
    first_positions = (first.extrema_zeta_unwrapped - first.zeta_out_unwrapped) / period
    second_positions = (
        second.extrema_zeta_unwrapped - second.zeta_out_unwrapped
    ) / period
    if first_positions.shape != second_positions.shape:
        return False
    if (
        first_positions.size
        and np.max(abs(first_positions - second_positions)) > tolerance
    ):
        return False
    scale = max(abs(first.b), abs(second.b), np.finfo(float).tiny)
    if (
        first.extrema_B.size
        and np.max(abs(first.extrema_B - second.extrema_B)) / scale > tolerance
    ):
        return False
    return True


def _surface_edges(triangles: IntArray) -> list[tuple[int, int]]:
    return sorted(
        {
            tuple(sorted((int(first), int(second))))
            for triangle in triangles
            for first, second in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            )
        }
    )


def _projected_midpoint(
    surface: SurfaceMesh,
    field: BoozerFieldLike,
    edge: tuple[int, int],
    config: SurfaceRefinementConfig,
) -> tuple[FloatArray, float, float]:
    first, second = edge
    seed = _periodic_midpoint(
        surface.points[first], surface.points[second], surface.period
    )
    edge_length = _periodic_edge_length(
        surface.points[first], surface.points[second], surface.period
    )
    extraction_config = SurfaceExtractionConfig(
        B_tolerance=config.B_tolerance,
        g_tolerance=config.g_tolerance,
        merge_tolerance=config.merge_tolerance,
    )
    point = _project_to_level_near(
        seed,
        field,
        surface.level,
        extraction_config,
        max(
            config.max_projection_distance_ratio * edge_length,
            config.merge_tolerance,
        ),
    )
    if point is None:
        raise SurfaceRefinementError(
            f"could not project midpoint of surface edge {edge} to B=b"
        )
    point = _canonicalize_point(point, surface.period, config.merge_tolerance)
    if np.sum(point[:2] ** 2) > 1.0 + config.merge_tolerance:
        raise SurfaceRefinementError(f"projected midpoint of edge {edge} left s<=1")
    B_value = float(_evaluate_B(field, point[np.newaxis, :])[0])
    if abs(B_value - surface.level) > config.B_tolerance:
        raise SurfaceRefinementError(
            f"projected midpoint of edge {edge} violates the B=b tolerance"
        )
    try:
        g_value = float(_physical_g(field, point[np.newaxis, :])[0])
    except SurfaceExtractionError as error:
        raise SurfaceRefinementError(
            f"could not evaluate physical g at midpoint of edge {edge}"
        ) from error
    if g_value > config.g_tolerance:
        raise SurfaceRefinementError(
            f"projected midpoint of incoming edge {edge} crossed to g>0"
        )
    return point, B_value, g_value


def _trace_point(
    field: BoozerFieldLike,
    b: float,
    point: FloatArray,
    trace_config: WellTraceConfig,
) -> WellTrace:
    s = float(point[0] ** 2 + point[1] ** 2)
    theta = float(np.arctan2(point[1], point[0])) if s > 0.0 else 0.0
    return trace_regular_well(
        field, b, np.array([s, theta, float(point[2])]), trace_config
    )


def _edge_samples(
    data: SurfaceData,
    field: BoozerFieldLike,
    config: SurfaceRefinementConfig,
    trace_config: WellTraceConfig,
) -> dict[tuple[int, int], _EdgeSample]:
    result = {}
    for edge in _surface_edges(data.surface.triangles):
        first, second = edge
        point, B_value, g_value = _projected_midpoint(data.surface, field, edge, config)
        midpoint_trace = _trace_point(field, data.surface.level, point, trace_config)
        endpoint_traces = (data.traces[first], data.traces[second])
        regular_triplet = all(
            trace.status is TraceStatus.REGULAR
            for trace in endpoint_traces + (midpoint_trace,)
        )
        if regular_triplet:
            action_error = abs(
                midpoint_trace.action_length
                - 0.5
                * (endpoint_traces[0].action_length + endpoint_traces[1].action_length)
            )
            bounce_time_error = abs(
                midpoint_trace.bounce_time_length
                - 0.5
                * (
                    endpoint_traces[0].bounce_time_length
                    + endpoint_traces[1].bounce_time_length
                )
            )
        else:
            action_error = np.nan
            bounce_time_error = np.nan
        candidate = not (
            itineraries_are_continuous(
                endpoint_traces[0],
                endpoint_traces[1],
                data.surface.period,
                config.itinerary_tolerance,
            )
            and itineraries_are_continuous(
                endpoint_traces[0],
                midpoint_trace,
                data.surface.period,
                config.itinerary_tolerance,
            )
            and itineraries_are_continuous(
                midpoint_trace,
                endpoint_traces[1],
                data.surface.period,
                config.itinerary_tolerance,
            )
        )
        result[edge] = _EdgeSample(
            point=point,
            B=B_value,
            g=g_value,
            trace=midpoint_trace,
            action_error=float(action_error),
            bounce_time_error=float(bounce_time_error),
            candidate=bool(candidate),
        )
    return result


def _exceeds(
    error: float,
    midpoint: float,
    first: float,
    second: float,
    *,
    rtol: float,
    atol: float,
) -> bool:
    if not np.isfinite(error):
        return False
    scale = max(abs(midpoint), abs(first), abs(second))
    return bool(error > atol + rtol * scale)


def _selected_edges(
    data: SurfaceData,
    samples: dict[tuple[int, int], _EdgeSample],
    config: SurfaceRefinementConfig,
) -> set[tuple[int, int]]:
    selected = set()
    for edge, sample in samples.items():
        first, second = edge
        if (
            sample.candidate
            or _exceeds(
                sample.action_error,
                sample.trace.action_length,
                data.action_length[first],
                data.action_length[second],
                rtol=config.action_interpolation_rtol,
                atol=config.action_interpolation_atol,
            )
            or _exceeds(
                sample.bounce_time_error,
                sample.trace.bounce_time_length,
                data.bounce_time_length[first],
                data.bounce_time_length[second],
                rtol=config.bounce_time_interpolation_rtol,
                atol=config.bounce_time_interpolation_atol,
            )
        ):
            selected.add(edge)
    return selected


def _maximum_finite(values) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if finite.size else np.nan


def _make_level(
    level: int,
    data: SurfaceData,
    samples: dict[tuple[int, int], _EdgeSample],
    selected: set[tuple[int, int]],
) -> SurfaceRefinementLevel:
    candidates = [edge for edge, sample in samples.items() if sample.candidate]
    s = np.sum(data.surface.points[:, :2] ** 2, axis=1)
    candidate_widths = [abs(s[first] - s[second]) for first, second in candidates]
    residuals = [
        abs(trace.B_residual_in)
        for trace in data.traces
        if np.isfinite(trace.B_residual_in)
    ]
    return SurfaceRefinementLevel(
        level=level,
        point_count=len(data.surface.points),
        triangle_count=len(data.surface.triangles),
        regular_trace_count=int(np.count_nonzero(data.regular)),
        failed_trace_count=int(np.count_nonzero(~data.regular)),
        refined_edge_count=len(selected),
        candidate_edge_count=len(candidates),
        max_action_interpolation_error=_maximum_finite(
            [sample.action_error for sample in samples.values()]
        ),
        max_bounce_time_interpolation_error=_maximum_finite(
            [sample.bounce_time_error for sample in samples.values()]
        ),
        max_candidate_s_width=float(max(candidate_widths, default=0.0)),
        max_B_residual=float(max(residuals, default=0.0)),
    )


def _make_edge_indicators(
    samples: dict[tuple[int, int], _EdgeSample],
) -> SurfaceEdgeIndicators:
    edges = list(samples)
    return SurfaceEdgeIndicators(
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        action_interpolation_error=np.asarray(
            [samples[edge].action_error for edge in edges], dtype=float
        ),
        bounce_time_interpolation_error=np.asarray(
            [samples[edge].bounce_time_error for edge in edges], dtype=float
        ),
        itinerary_candidate=np.asarray(
            [samples[edge].candidate for edge in edges], dtype=bool
        ),
    )


def _children_for_triangle(
    triangle: IntArray,
    midpoint_ids: dict[tuple[int, int], int],
) -> list[tuple[int, int, int]]:
    a, b, c = map(int, triangle)
    ab = midpoint_ids.get(tuple(sorted((a, b))))
    bc = midpoint_ids.get(tuple(sorted((b, c))))
    ca = midpoint_ids.get(tuple(sorted((c, a))))
    count = sum(value is not None for value in (ab, bc, ca))
    if count == 0:
        return [(a, b, c)]
    if count == 1:
        if ab is not None:
            return [(a, ab, c), (ab, b, c)]
        if bc is not None:
            return [(b, bc, a), (bc, c, a)]
        return [(c, ca, b), (ca, a, b)]
    if count == 2:
        if ab is not None and bc is not None:
            return [(b, bc, ab), (a, ab, c), (ab, bc, c)]
        if bc is not None and ca is not None:
            return [(c, ca, bc), (b, bc, a), (bc, ca, a)]
        return [(a, ab, ca), (c, ca, b), (ca, ab, b)]
    return [
        (a, ab, ca),
        (ab, b, bc),
        (ca, bc, c),
        (ab, bc, ca),
    ]


def _unwrap_vertices(vertices: FloatArray, period: float) -> FloatArray:
    result = np.asarray(vertices, dtype=float).copy()
    for index in range(1, len(result)):
        difference = result[index, 2] - result[0, 2]
        result[index, 2] -= period * np.round(difference / period)
    return result


def _refine_edges(
    surface: SurfaceMesh,
    samples: dict[tuple[int, int], _EdgeSample],
    selected: set[tuple[int, int]],
) -> SurfaceMesh:
    if not selected:
        return surface
    points = list(np.asarray(surface.points, dtype=float))
    B_values = list(np.asarray(surface.B, dtype=float))
    g_values = list(np.asarray(surface.g, dtype=float))
    tags = list(np.asarray(surface.boundary_tags, dtype=np.int64))
    parent_edges = list(np.asarray(surface.point_parent_edges, dtype=np.int64))
    midpoint_ids = {}
    for edge in sorted(selected):
        sample = samples[edge]
        midpoint_ids[edge] = len(points)
        points.append(sample.point)
        B_values.append(sample.B)
        g_values.append(sample.g)
        tags.append(
            int(surface.boundary_tags[edge[0]] & surface.boundary_tags[edge[1]])
        )
        parent_edges.append(np.array([-1, -1], dtype=np.int64))

    result_points = np.asarray(points, dtype=float)
    triangles = []
    parents = []
    components = []
    for triangle_id, triangle in enumerate(surface.triangles):
        children = _children_for_triangle(triangle, midpoint_ids)
        parent_vertices = _unwrap_vertices(surface.points[triangle], surface.period)
        parent_normal = np.cross(
            parent_vertices[1] - parent_vertices[0],
            parent_vertices[2] - parent_vertices[0],
        )
        for child in children:
            child_vertices = _unwrap_vertices(
                result_points[np.asarray(child)], surface.period
            )
            child_normal = np.cross(
                child_vertices[1] - child_vertices[0],
                child_vertices[2] - child_vertices[0],
            )
            if (
                np.linalg.norm(child_normal) <= np.finfo(float).eps
                or np.dot(parent_normal, child_normal) <= 0.0
            ):
                raise SurfaceRefinementError(
                    f"projected refinement inverted triangle {triangle_id}"
                )
            triangles.append(child)
            parents.append(
                surface.triangle_parent_tetrahedra[triangle_id]
                if len(children) == 1
                else -1
            )
            components.append(surface.component_ids[triangle_id])
    return SurfaceMesh(
        level=surface.level,
        period=surface.period,
        points=result_points,
        triangles=np.asarray(triangles, dtype=np.int64),
        B=np.asarray(B_values, dtype=float),
        g=np.asarray(g_values, dtype=float),
        boundary_tags=np.asarray(tags, dtype=np.int64),
        point_parent_edges=np.asarray(parent_edges, dtype=np.int64),
        triangle_parent_tetrahedra=np.asarray(parents, dtype=np.int64),
        component_ids=np.asarray(components, dtype=np.int64),
    )


def refine_surface_data(
    surface: SurfaceMesh,
    field: BoozerFieldLike,
    config: SurfaceRefinementConfig | None = None,
    trace_config: WellTraceConfig | None = None,
) -> SurfaceRefinementResult:
    """Refine nonlinear action data and return-sheet candidates (§§6, 8.4).

    Every unique edge is sampled at a locally projected ``B=b`` midpoint.
    Midpoint deviations from linear interpolation estimate errors in ``A``
    and ``K``. Unquantized lifted return data identify candidate sheet
    discontinuities. Marked edges are bisected conformingly, all changed
    provenance is invalidated, and every new vertex is checked to remain on
    the incoming half. The report includes the final unrefined indicator level,
    so convergence and the remaining candidate width are machine-readable.
    """
    cfg = SurfaceRefinementConfig() if config is None else config
    trace_cfg = WellTraceConfig() if trace_config is None else trace_config
    if np.any(surface.g > cfg.g_tolerance):
        raise ValueError("surface refinement requires the incoming g<=0 half")
    current = surface
    history = []
    data = evaluate_surface_data(current, field, trace_cfg)
    for level in range(cfg.max_levels + 1):
        samples = _edge_samples(data, field, cfg, trace_cfg)
        selected = _selected_edges(data, samples, cfg)
        will_refine = selected if level < cfg.max_levels else set()
        history.append(_make_level(level, data, samples, will_refine))
        if level == cfg.max_levels or not selected:
            break
        current = _refine_edges(current, samples, selected)
        data = evaluate_surface_data(current, field, trace_cfg)
    return SurfaceRefinementResult(
        surface=current,
        data=data,
        edge_indicators=_make_edge_indicators(samples),
        report=SurfaceRefinementReport(tuple(history)),
    )
