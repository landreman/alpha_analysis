"""Marginal-curve classification on fixed-pitch surfaces.

This module implements DESIGN.md §§5.1–5.4 and 10.1.  It consumes the
authoritative NumPy ``B=b, g=0`` curve mesh produced by surface extraction,
stitches only provenance-marked copies across the one-field-period seam,
classifies points and segments with analytic ``D_parallel^2 B``, and refines
segments that straddle a change of critical type.  Angles are radians and
``D2_B`` has the field's units per radian squared.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto

import numpy as np
from scipy.optimize import root

from .field import BoozerFieldLike
from .surface_extract import SurfaceCurveMesh, SurfaceExtraction, SurfaceMesh
from .types import FloatArray, IntArray, SurfaceStatus


class CriticalCurveError(RuntimeError):
    """A critical curve could not be returned without hiding ambiguity."""


class CriticalKind(IntEnum):
    """Classification by the sign of ``D_parallel^2 B`` (§5.2)."""

    GAMMA_MIN = auto()
    GAMMA_MAX = auto()
    DEGENERATE = auto()


class CriticalCurveStatus(IntEnum):
    """Overall result status; unresolved segments remain explicit."""

    REGULAR = auto()
    DEGENERATE = auto()
    UNRESOLVED = auto()


@dataclass(frozen=True)
class CriticalCurveConfig:
    """Numerical safeguards for critical-curve extraction.

    ``B_tolerance`` is in field units, ``g_tolerance`` is in physical
    ``b dot grad B`` units, and ``D2_tolerance`` is in field units per radian
    squared.  ``merge_tolerance`` and logical coordinates are dimensionless.
    Zero ``max_refinement_levels`` disables junction refinement; a positive
    value permits the direct degenerate-point solve. A failed solve is reported
    as an unresolved segment rather than guessed or discarded.
    """

    B_tolerance: float = 1.0e-9
    g_tolerance: float = 1.0e-9
    D2_tolerance: float = 1.0e-8
    merge_tolerance: float = 1.0e-9
    max_refinement_levels: int = 6
    max_midpoint_displacement_ratio: float = 2.0

    def __post_init__(self) -> None:
        for name in (
            "B_tolerance",
            "g_tolerance",
            "D2_tolerance",
            "merge_tolerance",
            "max_midpoint_displacement_ratio",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_refinement_levels < 0:
            raise ValueError("max_refinement_levels must be nonnegative")


@dataclass(frozen=True)
class CriticalPolyline:
    """One connected, single-class critical polyline on the quotient.

    ``vertex_ids`` and ``segment_ids`` index the owning ``CriticalCurves``.
    ``u`` is cumulative logical arc length from the first vertex.  Closed
    polylines do not repeat their first vertex; ``total_length`` includes the
    closing segment.
    """

    kind: CriticalKind
    vertex_ids: IntArray
    segment_ids: IntArray
    u: FloatArray
    total_length: float
    closed: bool


@dataclass(frozen=True)
class CriticalCurveReport:
    """Machine-readable classification-refinement diagnostics (§21.3)."""

    refined_segment_count: int
    unresolved_segment_count: int
    degenerate_solve_failure_count: int
    unresolved_endpoint_count: int
    source_unresolved_split_count: int
    ambiguous_segment_length_history: tuple[float, ...]


@dataclass(frozen=True)
class CriticalCurves:
    """Classified marginal curves represented only by NumPy arrays.

    Points use dimensionless logical ``(x,y,zeta)`` coordinates with zeta in
    radians on ``0 <= zeta < period``.  Segment and point kinds are explicit;
    A segment left ambiguous because refinement is disabled or its local solve
    fails is ``DEGENERATE`` and the overall status is ``UNRESOLVED`` rather
    than a guessed minimum or maximum.
    """

    period: float
    b: float
    points: FloatArray
    segments: IntArray
    D2_B: FloatArray
    point_kind: np.ndarray
    segment_kind: np.ndarray
    boundary_tags: IntArray
    polylines: tuple[CriticalPolyline, ...]
    status: CriticalCurveStatus
    report: CriticalCurveReport

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        segments = np.asarray(self.segments, dtype=np.int64)
        D2 = np.asarray(self.D2_B, dtype=np.float64)
        point_kind = np.asarray(self.point_kind, dtype=np.int64)
        segment_kind = np.asarray(self.segment_kind, dtype=np.int64)
        tags = np.asarray(self.boundary_tags, dtype=np.int64)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("critical-curve points must have shape (n_points, 3)")
        if segments.ndim != 2 or segments.shape[1:] != (2,):
            raise ValueError("critical-curve segments must have shape (n_segments, 2)")
        if D2.shape != (len(points),) or point_kind.shape != (len(points),):
            raise ValueError("point data must have one value per critical point")
        if tags.shape != (len(points),):
            raise ValueError("boundary tags must have one value per critical point")
        if segment_kind.shape != (len(segments),):
            raise ValueError("segment kinds must have one value per segment")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(D2)):
            raise ValueError("critical-curve point data must be finite")
        if segments.size and (segments.min() < 0 or segments.max() >= len(points)):
            raise ValueError("critical-curve segment index is outside the point array")
        valid = np.array([kind.value for kind in CriticalKind])
        if np.any(~np.isin(point_kind, valid)) or np.any(~np.isin(segment_kind, valid)):
            raise ValueError("unknown critical-curve classification")
        for name, values in (
            ("points", points),
            ("segments", segments),
            ("D2_B", D2),
            ("point_kind", point_kind),
            ("segment_kind", segment_kind),
            ("boundary_tags", tags),
        ):
            object.__setattr__(self, name, values)


def _coordinates(points: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    s = np.sum(points[:, :2] ** 2, axis=1)
    theta = np.arctan2(points[:, 1], points[:, 0])
    theta[s == 0.0] = 0.0
    return s, theta, points[:, 2]


def _field_values(
    field: BoozerFieldLike, points: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    s, theta, zeta = _coordinates(points)
    B = np.asarray(field.B(s, theta, zeta), dtype=np.float64)
    D_B = np.asarray(field.D_B(s, theta, zeta), dtype=np.float64)
    D2_B = np.asarray(field.D2_B(s, theta, zeta), dtype=np.float64)
    C = np.asarray(field.G(s), dtype=np.float64) + np.asarray(
        field.iota(s), dtype=np.float64
    ) * np.asarray(field.I(s), dtype=np.float64)
    if np.any(~np.isfinite(C)) or np.any(C == 0.0):
        raise CriticalCurveError("G + iota I must be finite and nonzero")
    g = B * D_B / C
    return B, g, D2_B


def _point_kinds(D2_B: FloatArray, tolerance: float) -> np.ndarray:
    kinds = np.full(len(D2_B), CriticalKind.DEGENERATE.value, dtype=np.int64)
    kinds[D2_B > tolerance] = CriticalKind.GAMMA_MIN.value
    kinds[D2_B < -tolerance] = CriticalKind.GAMMA_MAX.value
    return kinds


def _periodic_delta(first: FloatArray, second: FloatArray, period: float) -> FloatArray:
    delta = np.asarray(second - first, dtype=np.float64)
    delta[2] -= period * np.round(delta[2] / period)
    return delta


def _segment_length(
    points: FloatArray, segment: tuple[int, int] | np.ndarray, period: float
) -> float:
    first, second = map(int, segment)
    return float(np.linalg.norm(_periodic_delta(points[first], points[second], period)))


def _stitch_periodic_endpoints(
    curve: SurfaceCurveMesh, tolerance: float
) -> tuple[FloatArray, IntArray, IntArray]:
    """Identify only explicit lower/upper seam endpoint copies.

    Euclidean proximity alone is never sufficient: candidates must carry
    ``PERIODIC_SEAM`` provenance, have graph degree one, lie on opposite seam
    sides, and form a unique one-to-one match in ``(x,y)``.  This prevents the
    forbidden merging of nearby disconnected surface components (§21.2).
    """
    points = np.asarray(curve.points, dtype=np.float64).copy()
    segments = np.asarray(curve.segments, dtype=np.int64).copy()
    tags = np.asarray(curve.boundary_tags, dtype=np.int64).copy()
    if not len(points):
        return points, segments, tags
    degree = np.bincount(segments.ravel(), minlength=len(points))
    seam_end = ((tags & SurfaceMesh.PERIODIC_SEAM) != 0) & (degree == 1)
    lower = np.flatnonzero(seam_end & (np.abs(points[:, 2]) <= tolerance))
    upper = np.flatnonzero(
        seam_end & (np.abs(points[:, 2] - curve.period) <= tolerance)
    )
    if not len(lower) and not len(upper):
        points[:, 2] = np.mod(points[:, 2], curve.period)
        return points, segments, tags
    if len(lower) != len(upper):
        raise CriticalCurveError(
            "periodic critical-curve endpoints do not have equal lower/upper counts"
        )
    candidates = np.linalg.norm(
        points[lower, np.newaxis, :2] - points[upper, :2], axis=2
    )
    remap = np.arange(len(points), dtype=np.int64)
    used_upper: set[int] = set()
    for lower_id, row in zip(lower, candidates):
        possible = np.flatnonzero(row <= tolerance)
        possible = np.array(
            [index for index in possible if int(upper[index]) not in used_upper],
            dtype=np.int64,
        )
        if len(possible) != 1:
            raise CriticalCurveError(
                "periodic critical-curve endpoint matching is not uniquely one-to-one"
            )
        upper_id = int(upper[int(possible[0])])
        used_upper.add(upper_id)
        remap[upper_id] = int(lower_id)
        tags[lower_id] |= tags[upper_id]
    segments = remap[segments]
    segments = segments[segments[:, 0] != segments[:, 1]]
    segment_keys = np.sort(segments, axis=1)
    if len(segment_keys):
        _, keep = np.unique(segment_keys, axis=0, return_index=True)
        segments = segments[np.sort(keep)]
    used = np.unique(segments) if len(segments) else np.empty(0, dtype=np.int64)
    compact = np.full(len(points), -1, dtype=np.int64)
    compact[used] = np.arange(len(used))
    points = points[used]
    points[:, 2] = np.mod(points[:, 2], curve.period)
    return points, compact[segments], tags[used]


def _interpolated_coordinates(
    first: FloatArray, second: FloatArray, fraction: float, period: float
) -> np.ndarray:
    s0 = float(np.dot(first[:2], first[:2]))
    s1 = float(np.dot(second[:2], second[:2]))
    theta0 = float(np.arctan2(first[1], first[0]))
    theta1 = float(np.arctan2(second[1], second[0]))
    theta1 = theta0 + (theta1 - theta0 + np.pi) % (2.0 * np.pi) - np.pi
    zeta0 = float(first[2])
    zeta1 = zeta0 + (float(second[2]) - zeta0 + 0.5 * period) % period - 0.5 * period
    first_coordinates = np.array([s0, theta0, zeta0])
    second_coordinates = np.array([s1, theta1, zeta1])
    return first_coordinates + fraction * (second_coordinates - first_coordinates)


def _midpoint_coordinates(
    first: FloatArray, second: FloatArray, period: float
) -> np.ndarray:
    return _interpolated_coordinates(first, second, 0.5, period)


def _point_from_coordinates(coordinates: np.ndarray, period: float) -> FloatArray:
    s, theta, zeta = map(float, coordinates)
    rho = np.sqrt(max(s, 0.0))
    return np.array([rho * np.cos(theta), rho * np.sin(theta), zeta % period])


def _degenerate_point(
    first: FloatArray,
    second: FloatArray,
    field: BoozerFieldLike,
    b: float,
    period: float,
    config: CriticalCurveConfig,
) -> FloatArray:
    def residual(coordinates: np.ndarray) -> np.ndarray:
        point = _point_from_coordinates(coordinates, period)
        B, g, D2_B = _field_values(field, point[np.newaxis, :])
        return np.array(
            [
                float(B[0] - b),
                float(g[0]),
                float(D2_B[0]),
            ]
        )

    edge_length = float(np.linalg.norm(_periodic_delta(first, second, period)))
    endpoint_D2 = np.array(
        [
            _field_values(field, endpoint[np.newaxis, :])[2][0]
            for endpoint in (first, second)
        ]
    )
    degenerate_endpoints = [
        endpoint
        for endpoint, D2_B in zip((first, second), endpoint_D2)
        if abs(float(D2_B)) <= config.D2_tolerance
    ]
    # HYBR can choose a different basin after tiny platform-dependent changes.
    # Fixed interior fallback starts avoid making that basin choice decisive;
    # the physical residual, domain, and locality gates below remain
    # authoritative instead of the solver's success flag.
    for fraction in (0.5, 0.375, 0.625):
        seed = _interpolated_coordinates(first, second, fraction, period)
        initial_residual = residual(seed)
        if (
            abs(initial_residual[0]) <= config.B_tolerance
            and abs(initial_residual[1]) <= config.g_tolerance
            and abs(initial_residual[2]) <= config.D2_tolerance
        ):
            solution = seed
        else:
            solved = root(residual, seed, options={"xtol": 1.0e-12})
            solution = np.asarray(solved.x, dtype=np.float64)
        if solution.shape != (3,) or not np.all(np.isfinite(solution)):
            continue
        point = _point_from_coordinates(solution, period)
        B, g, D2_B = _field_values(field, point[np.newaxis, :])
        seed_point = _point_from_coordinates(seed, period)
        if (
            not (0.0 <= solution[0] <= 1.0 + config.merge_tolerance)
            or abs(float(B[0] - b)) > config.B_tolerance
            or abs(float(g[0])) > config.g_tolerance
            or abs(float(D2_B[0])) > config.D2_tolerance
            or np.linalg.norm(_periodic_delta(seed_point, point, period))
            > config.max_midpoint_displacement_ratio * edge_length
            or any(
                np.linalg.norm(_periodic_delta(endpoint, point, period))
                <= config.merge_tolerance
                for endpoint in degenerate_endpoints
            )
        ):
            continue
        return point
    raise CriticalCurveError("degenerate point violates B, g, D2_B, or locality bounds")


def _segment_midpoint_data(
    points: FloatArray,
    segments: IntArray,
    field: BoozerFieldLike,
    period: float,
    tolerance: float,
) -> tuple[FloatArray, np.ndarray]:
    midpoints = np.array(
        [
            _point_from_coordinates(
                _midpoint_coordinates(points[first], points[second], period), period
            )
            for first, second in segments
        ],
        dtype=np.float64,
    ).reshape(-1, 3)
    if not len(midpoints):
        return midpoints, np.empty(0, dtype=np.int64)
    _, _, midpoint_D2 = _field_values(field, midpoints)
    return midpoints, _point_kinds(midpoint_D2, tolerance)


def _refinement_bracket(
    first: FloatArray,
    second: FloatArray,
    first_kind: int,
    second_kind: int,
    midpoint: FloatArray,
    midpoint_kind: int,
) -> tuple[FloatArray, FloatArray]:
    """Choose a half-segment that brackets one unresolved type change."""
    degenerate = CriticalKind.DEGENERATE.value
    if (
        first_kind != degenerate
        and midpoint_kind != degenerate
        and first_kind != midpoint_kind
    ):
        return first, midpoint
    if (
        midpoint_kind != degenerate
        and second_kind != degenerate
        and midpoint_kind != second_kind
    ):
        return midpoint, second
    return first, second


def _ambiguous_segments(
    segments: IntArray, point_kind: np.ndarray, midpoint_kind: np.ndarray
) -> np.ndarray:
    first = point_kind[segments[:, 0]]
    second = point_kind[segments[:, 1]]
    degenerate = CriticalKind.DEGENERATE.value
    regular_first = first != degenerate
    regular_second = second != degenerate
    regular_midpoint = midpoint_kind != degenerate
    sign_change = (
        (regular_first & regular_second & (first != second))
        | (regular_first & regular_midpoint & (first != midpoint_kind))
        | (regular_second & regular_midpoint & (second != midpoint_kind))
    )
    near_zero_inside = (midpoint_kind == degenerate) & (regular_first | regular_second)
    return sign_change | near_zero_inside


def _classify_segments(
    segments: IntArray,
    point_kind: np.ndarray,
    midpoint_kind: np.ndarray,
    unresolved: np.ndarray,
) -> np.ndarray:
    result = np.full(len(segments), CriticalKind.DEGENERATE.value, dtype=np.int64)
    for segment_id, (first, second) in enumerate(segments):
        if unresolved[segment_id]:
            continue
        endpoint_kinds = {
            int(point_kind[first]),
            int(point_kind[second]),
            int(midpoint_kind[segment_id]),
        } - {CriticalKind.DEGENERATE.value}
        if len(endpoint_kinds) == 1:
            result[segment_id] = endpoint_kinds.pop()
        elif not endpoint_kinds:
            result[segment_id] = CriticalKind.DEGENERATE.value
        else:
            result[segment_id] = CriticalKind.DEGENERATE.value
    return result


def _walk_polylines(
    points: FloatArray,
    segments: IntArray,
    segment_kind: np.ndarray,
    period: float,
) -> tuple[CriticalPolyline, ...]:
    polylines: list[CriticalPolyline] = []
    for kind in CriticalKind:
        selected = np.flatnonzero(segment_kind == kind.value)
        if not len(selected):
            continue
        adjacency: dict[int, list[int]] = {}
        for segment_id in selected:
            first, second = map(int, segments[segment_id])
            adjacency.setdefault(first, []).append(int(segment_id))
            adjacency.setdefault(second, []).append(int(segment_id))
        unvisited = set(map(int, selected))

        def walk(start: int, initial_segment: int) -> CriticalPolyline:
            vertex_ids = [start]
            segment_ids: list[int] = []
            current = start
            segment_id = initial_segment
            closed = False
            while segment_id in unvisited:
                unvisited.remove(segment_id)
                segment_ids.append(segment_id)
                first, second = map(int, segments[segment_id])
                following = second if first == current else first
                if following == start:
                    closed = True
                    break
                vertex_ids.append(following)
                current = following
                available = [
                    candidate
                    for candidate in adjacency[current]
                    if candidate in unvisited
                ]
                if len(adjacency[current]) != 2 or not available:
                    break
                segment_id = min(available)
            u = np.zeros(len(vertex_ids), dtype=np.float64)
            for index in range(1, len(vertex_ids)):
                u[index] = u[index - 1] + _segment_length(
                    points, (vertex_ids[index - 1], vertex_ids[index]), period
                )
            total = float(u[-1]) if len(u) else 0.0
            if closed:
                total += _segment_length(points, (vertex_ids[-1], start), period)
            return CriticalPolyline(
                kind=kind,
                vertex_ids=np.asarray(vertex_ids, dtype=np.int64),
                segment_ids=np.asarray(segment_ids, dtype=np.int64),
                u=u,
                total_length=total,
                closed=closed,
            )

        for start in sorted(
            vertex for vertex, edges in adjacency.items() if len(edges) != 2
        ):
            for segment_id in sorted(adjacency[start]):
                if segment_id in unvisited:
                    polylines.append(walk(start, segment_id))
        while unvisited:
            initial_segment = min(unvisited)
            start = int(min(segments[initial_segment]))
            polylines.append(walk(start, initial_segment))
    return tuple(polylines)


def extract_critical_curves(
    curve: SurfaceCurveMesh | SurfaceExtraction,
    field: BoozerFieldLike,
    b: float,
    config: CriticalCurveConfig | None = None,
) -> CriticalCurves:
    """Extract and classify ``Gamma_min``, ``Gamma_max``, and degeneracies.

    The preferred input is the complete ``SurfaceExtraction`` so its status
    and unresolved split count cannot be lost; a standalone recorded
    ``B=b, g=0`` boundary is accepted for synthetic and staged callers.
    Classification uses the analytic second parallel derivative
    from DESIGN.md §5.1: positive is ``GAMMA_MIN``, negative is
    ``GAMMA_MAX``, and magnitude at or below ``D2_tolerance`` is
    ``DEGENERATE``.  Oppositely classified endpoints trigger a local solve of
    ``B-b = g = D_parallel^2 B = 0`` and insertion of that degenerate point.
    If refinement is disabled or the local solve fails its residual/locality
    gates, the original segment remains explicitly degenerate and the result
    status is ``UNRESOLVED``.
    """
    cfg = config or CriticalCurveConfig()
    if not np.isfinite(b):
        raise ValueError("b must be finite")
    source_status = SurfaceStatus.REGULAR
    source_unresolved_splits = 0
    if isinstance(curve, SurfaceExtraction):
        source_status = curve.status
        source_unresolved_splits = int(curve.n_unresolved_splits)
        curve = curve.g_zero
    if not np.isclose(curve.period, 2.0 * np.pi / field.nfp):
        raise CriticalCurveError("curve period disagrees with the field period")
    points, segments, tags = _stitch_periodic_endpoints(curve, cfg.merge_tolerance)
    if not len(points):
        empty = np.empty(0, dtype=np.float64)
        return CriticalCurves(
            period=curve.period,
            b=b,
            points=points,
            segments=segments,
            D2_B=empty,
            point_kind=np.empty(0, dtype=np.int64),
            segment_kind=np.empty(0, dtype=np.int64),
            boundary_tags=tags,
            polylines=(),
            status=(
                CriticalCurveStatus.REGULAR
                if source_status is SurfaceStatus.REGULAR
                and source_unresolved_splits == 0
                else CriticalCurveStatus.UNRESOLVED
            ),
            report=CriticalCurveReport(
                refined_segment_count=0,
                unresolved_segment_count=0,
                degenerate_solve_failure_count=0,
                unresolved_endpoint_count=0,
                source_unresolved_split_count=source_unresolved_splits,
                ambiguous_segment_length_history=(0.0,),
            ),
        )
    B, g, D2_B = _field_values(field, points)
    if np.max(np.abs(B - b)) > cfg.B_tolerance:
        raise CriticalCurveError("critical-curve point violates B=b tolerance")
    if np.max(np.abs(g)) > cfg.g_tolerance:
        raise CriticalCurveError("critical-curve point violates g=0 tolerance")
    point_kind = _point_kinds(D2_B, cfg.D2_tolerance)
    points_list = list(points)
    tags_list = list(tags)
    segments_list = [tuple(map(int, segment)) for segment in segments]
    D2_list = list(D2_B)
    kind_list = list(point_kind)
    refined_count = 0
    solve_failure_count = 0
    history: list[float] = []
    failed_segment_keys: set[tuple[int, int]] = set()

    for level in range(cfg.max_refinement_levels + 1):
        segment_array = np.asarray(segments_list, dtype=np.int64).reshape(-1, 2)
        kind_array = np.asarray(kind_list, dtype=np.int64)
        point_array = np.asarray(points_list, dtype=np.float64)
        midpoint_points, midpoint_kind = _segment_midpoint_data(
            point_array, segment_array, field, curve.period, cfg.D2_tolerance
        )
        ambiguous = _ambiguous_segments(segment_array, kind_array, midpoint_kind)
        retry = ambiguous.copy()
        for segment_id, segment in enumerate(segment_array):
            if tuple(sorted(map(int, segment))) in failed_segment_keys:
                retry[segment_id] = False
        lengths = [
            _segment_length(point_array, segment, curve.period)
            for segment in segment_array[ambiguous]
        ]
        history.append(max(lengths, default=0.0))
        if not np.any(retry) or level == cfg.max_refinement_levels:
            break
        replacement: list[tuple[int, int]] = []
        for segment_id, (first, second) in enumerate(segments_list):
            if not retry[segment_id]:
                replacement.append((first, second))
                continue
            bracket_first, bracket_second = _refinement_bracket(
                point_array[first],
                point_array[second],
                int(kind_array[first]),
                int(kind_array[second]),
                midpoint_points[segment_id],
                int(midpoint_kind[segment_id]),
            )
            try:
                midpoint = _degenerate_point(
                    bracket_first,
                    bracket_second,
                    field,
                    b,
                    curve.period,
                    cfg,
                )
            except CriticalCurveError:
                replacement.append((first, second))
                solve_failure_count += 1
                failed_segment_keys.add(tuple(sorted((first, second))))
                continue
            _, _, midpoint_D2 = _field_values(field, midpoint[np.newaxis, :])
            midpoint_id = len(points_list)
            points_list.append(midpoint)
            tags_list.append(SurfaceMesh.G_ZERO)
            D2_list.append(float(midpoint_D2[0]))
            kind_list.append(int(_point_kinds(midpoint_D2, cfg.D2_tolerance)[0]))
            replacement.extend(((first, midpoint_id), (midpoint_id, second)))
            refined_count += 1
        segments_list = replacement

    points = np.asarray(points_list, dtype=np.float64).reshape(-1, 3)
    segments = np.asarray(segments_list, dtype=np.int64).reshape(-1, 2)
    D2_B = np.asarray(D2_list, dtype=np.float64)
    point_kind = np.asarray(kind_list, dtype=np.int64)
    _, midpoint_kind = _segment_midpoint_data(
        points, segments, field, curve.period, cfg.D2_tolerance
    )
    unresolved = _ambiguous_segments(segments, point_kind, midpoint_kind)
    segment_kind = _classify_segments(segments, point_kind, midpoint_kind, unresolved)
    polylines = _walk_polylines(points, segments, segment_kind, curve.period)
    unresolved_count = int(np.count_nonzero(unresolved))
    degree = np.bincount(segments.ravel(), minlength=len(points))
    tag_array = np.asarray(tags_list, dtype=np.int64)
    physical_endpoint = (
        tag_array & (SurfaceMesh.EDGE | SurfaceMesh.AXIS | SurfaceMesh.PERIODIC_SEAM)
    ) != 0
    unresolved_endpoint_count = int(
        np.count_nonzero((degree == 1) & ~physical_endpoint)
    )
    source_unresolved = (
        source_status is not SurfaceStatus.REGULAR or source_unresolved_splits > 0
    )
    has_degenerate = np.any(point_kind == CriticalKind.DEGENERATE.value) or np.any(
        segment_kind == CriticalKind.DEGENERATE.value
    )
    status = (
        CriticalCurveStatus.UNRESOLVED
        if unresolved_count or unresolved_endpoint_count or source_unresolved
        else (
            CriticalCurveStatus.DEGENERATE
            if has_degenerate
            else CriticalCurveStatus.REGULAR
        )
    )
    return CriticalCurves(
        period=curve.period,
        b=b,
        points=points,
        segments=segments,
        D2_B=D2_B,
        point_kind=point_kind,
        segment_kind=segment_kind,
        boundary_tags=tag_array,
        polylines=polylines,
        status=status,
        report=CriticalCurveReport(
            refined_segment_count=refined_count,
            unresolved_segment_count=unresolved_count,
            degenerate_solve_failure_count=solve_failure_count,
            unresolved_endpoint_count=unresolved_endpoint_count,
            source_unresolved_split_count=source_unresolved_splits,
            ambiguous_segment_length_history=tuple(history),
        ),
    )
