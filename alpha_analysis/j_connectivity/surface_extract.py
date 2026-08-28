"""Fixed-``B`` level-surface extraction (DESIGN.md §8.3)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import ClassVar

import numpy as np
from scipy.optimize import brentq, linear_sum_assignment, root
from scipy.spatial import cKDTree

from .field import BoozerFieldLike
from .types import BackgroundMesh, SurfaceStatus

_PYVISTA_OUTER_INDICATOR = "background OUTER boundary indicator"


class SurfaceExtractionError(RuntimeError):
    """A level surface could not be returned without hiding a failure (§21.2)."""

    def __init__(self, status: SurfaceStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class SurfaceExtractionConfig:
    """Absolute tolerances for marching tetrahedra and root polishing.

    ``B_tolerance`` is in the field's magnetic-field units, ``g_tolerance`` is
    in the units of ``B D_parallel B / (G + iota I)``, and the parameter and
    merge tolerances are dimensionless. These implement DESIGN.md §§5.1 and
    8.3; they are numerical safeguards, not certified accuracy settings.
    """

    B_tolerance: float = 1.0e-10
    g_tolerance: float = 1.0e-10
    parameter_tolerance: float = 1.0e-13
    merge_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        for name, value in (
            ("B_tolerance", self.B_tolerance),
            ("g_tolerance", self.g_tolerance),
            ("parameter_tolerance", self.parameter_tolerance),
            ("merge_tolerance", self.merge_tolerance),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class SurfaceMesh:
    """Authoritative triangular mesh on the one-field-period quotient (§8.1).

    Points use dimensionless logical ``(x, y, zeta)`` coordinates. On
    construction, zeta is reduced in radians to ``0 <= zeta < period`` and
    values within roundoff of either seam are snapped to zero. ``B`` retains the
    field's units and ``g`` stores the physical signed
    ``B D_parallel B / (G + iota I)`` from DESIGN.md §5.1. Point parent edges
    are canonical background-node IDs; a ``(-1, -1)`` entry denotes a point
    introduced by the second marching-triangles split. Parent tetrahedra and
    component IDs are cell-located integer arrays.
    """

    EDGE: ClassVar[int] = 1
    AXIS: ClassVar[int] = 2
    PERIODIC_SEAM: ClassVar[int] = 4
    G_ZERO: ClassVar[int] = 8
    # A split vertex placed at the g sign discontinuity of a marching
    # triangle that bridges two sheets of an under-resolved surface: no
    # local g=0 point exists there, so the vertex satisfies B=b but not
    # g=0, is excluded from the g=0 curve, and marks the extraction
    # UNRESOLVED (ADR 0001).
    G_JUMP: ClassVar[int] = 16

    level: float
    period: float
    points: np.ndarray
    triangles: np.ndarray
    B: np.ndarray
    g: np.ndarray
    boundary_tags: np.ndarray
    point_parent_edges: np.ndarray
    triangle_parent_tetrahedra: np.ndarray
    component_ids: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.period) or self.period <= 0.0:
            raise ValueError("surface period must be finite and positive")
        points = np.asarray(self.points, dtype=np.float64).copy()
        if points.ndim == 2 and points.shape[1:] == (3,):
            points[:, 2] = np.mod(points[:, 2], self.period)
            near_seam = np.minimum(points[:, 2], self.period - points[:, 2]) <= (
                16.0 * np.finfo(float).eps * self.period
            )
            points[near_seam, 2] = 0.0
        triangles = np.asarray(self.triangles, dtype=np.int64)
        B = np.asarray(self.B, dtype=np.float64)
        g = np.asarray(self.g, dtype=np.float64)
        tags = np.asarray(self.boundary_tags, dtype=np.int64)
        parent_edges = np.asarray(self.point_parent_edges, dtype=np.int64)
        parent_tetrahedra = np.asarray(self.triangle_parent_tetrahedra, dtype=np.int64)
        components = np.asarray(self.component_ids, dtype=np.int64)
        n_points = len(points)
        n_triangles = len(triangles)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("surface points must have shape (n_points, 3)")
        if triangles.ndim != 2 or triangles.shape[1:] != (3,):
            raise ValueError("surface triangles must have shape (n_triangles, 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("surface points must be finite")
        if B.shape != (n_points,) or g.shape != (n_points,):
            raise ValueError("surface B and g must have one value per point")
        if not np.all(np.isfinite(B)) or not np.all(np.isfinite(g)):
            raise ValueError("surface B and g must be finite")
        if tags.shape != (n_points,):
            raise ValueError("surface boundary_tags must have one value per point")
        if parent_edges.shape != (n_points, 2):
            raise ValueError("point_parent_edges must have shape (n_points, 2)")
        if parent_tetrahedra.shape != (n_triangles,):
            raise ValueError("each surface triangle needs one parent tetrahedron")
        if components.shape != (n_triangles,):
            raise ValueError("each surface triangle needs one component ID")
        if triangles.size and (triangles.min() < 0 or triangles.max() >= n_points):
            raise ValueError("surface triangle point index is outside the mesh")
        if not np.isfinite(self.level):
            raise ValueError("surface level must be finite")
        for name, values in (
            ("points", points),
            ("triangles", triangles),
            ("B", B),
            ("g", g),
            ("boundary_tags", tags),
            ("point_parent_edges", parent_edges),
            ("triangle_parent_tetrahedra", parent_tetrahedra),
            ("component_ids", components),
        ):
            object.__setattr__(self, name, values)

    def to_pyvista(self):
        """Return an optional PyVista view; authoritative data stay arrays."""
        from . import optional_import

        pyvista = optional_import("pyvista", extra="connectivity")
        faces = np.column_stack(
            (np.full(len(self.triangles), 3, dtype=np.int64), self.triangles)
        ).ravel()
        surface = pyvista.PolyData(self.points, faces)
        surface.point_data["B [field units]"] = self.B
        surface.point_data["g = b dot grad B [physical units]"] = self.g
        surface.point_data["boundary tag [bit mask]"] = self.boundary_tags
        surface.cell_data["component ID"] = self.component_ids
        surface.cell_data["parent tetrahedron ID"] = self.triangle_parent_tetrahedra
        return surface


@dataclass(frozen=True)
class SurfaceCurveMesh:
    """Polyline representation of the recorded ``g=0`` split boundary (§8.3)."""

    period: float
    points: np.ndarray
    segments: np.ndarray
    B: np.ndarray
    g: np.ndarray
    boundary_tags: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        segments = np.asarray(self.segments, dtype=np.int64)
        B = np.asarray(self.B, dtype=np.float64)
        g = np.asarray(self.g, dtype=np.float64)
        tags = np.asarray(self.boundary_tags, dtype=np.int64)
        n_points = len(points)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("curve points must have shape (n_points, 3)")
        if segments.ndim != 2 or segments.shape[1:] != (2,):
            raise ValueError("curve segments must have shape (n_segments, 2)")
        if B.shape != (n_points,) or g.shape != (n_points,):
            raise ValueError("curve B and g must have one value per point")
        if tags.shape != (n_points,):
            raise ValueError("curve tags must have one value per point")
        if segments.size and (segments.min() < 0 or segments.max() >= n_points):
            raise ValueError("curve segment point index is outside the mesh")
        for name, values in (
            ("points", points),
            ("segments", segments),
            ("B", B),
            ("g", g),
            ("boundary_tags", tags),
        ):
            object.__setattr__(self, name, values)

    def to_pyvista(self):
        """Return an optional PyVista polyline view (DESIGN.md §17.1)."""
        from . import optional_import

        pyvista = optional_import("pyvista", extra="connectivity")
        lines = np.column_stack(
            (np.full(len(self.segments), 2, dtype=np.int64), self.segments)
        ).ravel()
        curve = pyvista.PolyData(self.points, lines=lines)
        curve.point_data["B [field units]"] = self.B
        curve.point_data["g = b dot grad B [physical units]"] = self.g
        curve.point_data["boundary tag [bit mask]"] = self.boundary_tags
        return curve


@dataclass(frozen=True)
class SurfaceExtraction:
    """Full ``B=b`` surface, signed halves, and preserved ``g=0`` curves.

    ``n_unresolved_splits`` counts split vertices placed at ``g`` sign
    discontinuities of sheet-bridging marching triangles (``G_JUMP``); when
    it is nonzero the status is ``UNRESOLVED`` and the ``g=0`` curve omits
    those vertices rather than pretending they satisfy ``g=0``.
    """

    b: float
    full: SurfaceMesh
    incoming: SurfaceMesh
    outgoing: SurfaceMesh
    g_zero: SurfaceCurveMesh
    status: SurfaceStatus = SurfaceStatus.REGULAR
    n_unresolved_splits: int = 0


class MarchingTetrahedraExtractor:
    """Custom production extractor with polished roots and provenance (§8.3)."""

    def __init__(self, config: SurfaceExtractionConfig | None = None) -> None:
        self.config = config or SurfaceExtractionConfig()

    def extract(
        self, background: BackgroundMesh, field: BoozerFieldLike, b: float
    ) -> SurfaceExtraction:
        """Extract all ``B=b`` components on the periodic logical cylinder.

        Edge intersections are bracketed by the background samples and polished
        against ``field.B``. Upper/lower seam edge copies receive one canonical
        vertex ID. A second marching-triangles pass uses the physical sign
        ``g=B D_parallel B/(G+iota I)`` and records the shared ``g=0`` boundary.
        Coordinates are dimensionless except zeta, which is in radians.
        """
        if not np.isfinite(b):
            raise ValueError("b must be finite")
        period = 2.0 * np.pi / field.nfp
        full = _march_tetrahedra(background, field, float(b), period, self.config)
        return _split_by_physical_g(full, field, self.config)


class PyVistaSurfaceExtractor:
    """Optional VTK contour prototype converted immediately to plain arrays."""

    def __init__(self, config: SurfaceExtractionConfig | None = None) -> None:
        self.config = config or SurfaceExtractionConfig()

    def extract(
        self, background: BackgroundMesh, field: BoozerFieldLike, b: float
    ) -> SurfaceExtraction:
        """Prototype ``B=b`` extraction; PyVista objects do not cross this call.

        VTK does not expose parent-edge or parent-tetrahedron IDs, so those
        provenance arrays contain ``-1``. Boundary provenance is captured
        before projection and carried through coordinate merging. Contour
        points are projected to ``B=b`` without leaving the logical plasma
        before the common physical-sign splitting stage.
        """
        if not np.isfinite(b):
            raise ValueError("b must be finite")
        period = 2.0 * np.pi / field.nfp
        grid = background.to_pyvista()
        grid.point_data[_PYVISTA_OUTER_INDICATOR] = (
            (background.boundary_tags & BackgroundMesh.OUTER) != 0
        ).astype(np.float64)
        polydata = grid.contour([float(b)], scalars="B [field units]").triangulate()
        points = np.asarray(polydata.points, dtype=np.float64)
        faces = np.asarray(polydata.faces, dtype=np.int64)
        if len(points) and _PYVISTA_OUTER_INDICATOR not in polydata.point_data:
            raise SurfaceExtractionError(
                SurfaceStatus.DEGENERATE,
                "PyVista contour lost the OUTER boundary indicator",
            )
        outer_indicator = (
            np.asarray(polydata.point_data[_PYVISTA_OUTER_INDICATOR], dtype=np.float64)
            if len(points)
            else np.empty(0, dtype=np.float64)
        )
        if len(faces):
            faces = faces.reshape(-1, 4)
            if np.any(faces[:, 0] != 3):
                raise SurfaceExtractionError(
                    SurfaceStatus.DEGENERATE, "PyVista returned a non-triangle face"
                )
            triangles = faces[:, 1:]
        else:
            triangles = np.empty((0, 3), dtype=np.int64)
        seam_sides = np.zeros(len(points), dtype=np.int8)
        seam_sides[points[:, 2] <= self.config.merge_tolerance] = -1
        seam_sides[period - points[:, 2] <= self.config.merge_tolerance] = 1
        source_boundary_tags = _pyvista_boundary_tags(
            points,
            triangles,
            outer_indicator,
            seam_sides,
            period,
            self.config.merge_tolerance,
        )
        points = _polish_pyvista_background_edge_roots(
            points,
            background,
            field,
            float(b),
            period,
            self.config,
        )
        points = _match_pyvista_periodic_seam(
            points, seam_sides, self.config.merge_tolerance
        )
        points, triangles, point_remap = _merge_coordinate_copies(
            points, triangles, period, self.config.merge_tolerance
        )
        B_values = _evaluate_B(field, points)
        g = _physical_g(field, points)
        tags = _coordinate_boundary_tags(points, period, self.config.merge_tolerance)
        np.bitwise_or.at(tags, point_remap, source_boundary_tags)
        tags[np.abs(g) <= self.config.g_tolerance] |= SurfaceMesh.G_ZERO
        triangles = _orient_triangles(points, triangles, field, period)
        full = SurfaceMesh(
            level=float(b),
            period=period,
            points=points,
            triangles=triangles,
            B=B_values,
            g=g,
            boundary_tags=tags,
            point_parent_edges=np.full((len(points), 2), -1, dtype=np.int64),
            triangle_parent_tetrahedra=np.full(len(triangles), -1, dtype=np.int64),
            component_ids=_component_ids(triangles),
        )
        return _split_by_physical_g(full, field, self.config)


def surface_flux(surface: SurfaceMesh, field: BoozerFieldLike) -> float:
    """Integrate ``|ds wedge d alpha|`` by triangle midpoint quadrature.

    This is the axis-regular two-form in DESIGN.md §4.4. Logical ``x`` and
    ``y`` are dimensionless and zeta is in radians, so the returned normalized
    flux measure is dimensionless. Periodic seam triangles are locally
    unwrapped before their tangent vectors are formed.
    """
    total = 0.0
    for triangle in surface.triangles:
        vertices = _unwrap_triangle(surface.points[triangle], surface.period)
        first = vertices[1] - vertices[0]
        second = vertices[2] - vertices[0]
        x, y, _ = np.mean(vertices, axis=0)
        s = x * x + y * y
        iota = float(field.iota(s))
        omega = 2.0 * (first[0] * second[1] - first[1] * second[0])
        omega -= iota * (
            (2.0 * x * first[0] + 2.0 * y * first[1]) * second[2]
            - (2.0 * x * second[0] + 2.0 * y * second[1]) * first[2]
        )
        total += 0.5 * abs(omega)
    return float(total)


def _march_tetrahedra(
    background: BackgroundMesh,
    field: BoozerFieldLike,
    b: float,
    period: float,
    config: SurfaceExtractionConfig,
) -> SurfaceMesh:
    representative = np.arange(len(background.points), dtype=np.int64)
    for lower, upper in background.periodic_node_pairs:
        representative[upper] = lower

    parent_edges: list[tuple[int, int]] = []
    point_tags: list[int] = []
    raw_edges: list[tuple[int, int]] = []
    duplicate_checks: list[tuple[int, tuple[int, int]]] = []
    duplicate_seen: set[tuple[int, int]] = set()
    triangles: list[tuple[int, int, int]] = []
    parent_tetrahedra: list[int] = []
    edge_cache: dict[tuple[int, int], int] = {}
    edge_sources: dict[tuple[int, int], tuple[int, int]] = {}

    def edge_vertex(first: int, second: int) -> int:
        """Register the root-carrying edge; polishing happens batched below."""
        raw_edge = tuple(sorted((int(first), int(second))))
        canonical_edge = tuple(
            sorted((int(representative[first]), int(representative[second])))
        )
        if canonical_edge[0] == canonical_edge[1]:
            raise SurfaceExtractionError(
                SurfaceStatus.DEGENERATE,
                "a level root lies on an edge collapsed by the periodic quotient",
            )
        if canonical_edge in edge_cache:
            point_id = edge_cache[canonical_edge]
            if edge_sources[canonical_edge] != raw_edge:
                point_tags[point_id] |= _edge_boundary_tag(background, first, second)
                if raw_edge not in duplicate_seen:
                    duplicate_seen.add(raw_edge)
                    duplicate_checks.append((point_id, raw_edge))
            return point_id

        point_id = len(raw_edges)
        edge_cache[canonical_edge] = point_id
        edge_sources[canonical_edge] = raw_edge
        raw_edges.append(raw_edge)
        parent_edges.append(canonical_edge)
        point_tags.append(_edge_boundary_tag(background, first, second))
        return point_id

    for tetrahedron_id, tetrahedron in enumerate(background.tetrahedra):
        values = background.B[tetrahedron] - b
        if np.any(np.abs(values) <= config.B_tolerance):
            raise SurfaceExtractionError(
                SurfaceStatus.DEGENERATE,
                "B=b passes through a background vertex; refine or perturb the slice",
            )
        positive = [index for index, value in enumerate(values) if value > 0.0]
        negative = [index for index, value in enumerate(values) if value < 0.0]
        if not positive or not negative:
            continue

        if len(positive) in (1, 3):
            isolated_group = positive if len(positive) == 1 else negative
            other_group = negative if len(positive) == 1 else positive
            isolated = isolated_group[0]
            vertices = [
                edge_vertex(tetrahedron[isolated], tetrahedron[other])
                for other in other_group
            ]
            triangles.append(tuple(vertices))
            parent_tetrahedra.append(tetrahedron_id)
        elif len(positive) == 2:
            positive = sorted(positive, key=lambda index: tetrahedron[index])
            negative = sorted(negative, key=lambda index: tetrahedron[index])
            a = edge_vertex(tetrahedron[positive[0]], tetrahedron[negative[0]])
            b_vertex = edge_vertex(tetrahedron[positive[0]], tetrahedron[negative[1]])
            c = edge_vertex(tetrahedron[positive[1]], tetrahedron[negative[0]])
            d = edge_vertex(tetrahedron[positive[1]], tetrahedron[negative[1]])
            if tuple(sorted((a, d))) <= tuple(sorted((b_vertex, c))):
                new_triangles = ((a, b_vertex, d), (a, d, c))
            else:
                new_triangles = ((a, b_vertex, c), (b_vertex, d, c))
            triangles.extend(new_triangles)
            parent_tetrahedra.extend((tetrahedron_id, tetrahedron_id))
        else:  # pragma: no cover
            raise AssertionError("invalid marching-tetrahedra sign partition")

    point_array = _polish_background_edges(
        background, raw_edges, field, b, period, config
    )
    if duplicate_checks:
        # Seam partner copies of already-polished edges must land on the
        # same canonical point.
        duplicate_points = _polish_background_edges(
            background,
            [raw_edge for _, raw_edge in duplicate_checks],
            field,
            b,
            period,
            config,
        )
        for (point_id, _), duplicate in zip(duplicate_checks, duplicate_points):
            if (
                _periodic_distance(point_array[point_id], duplicate, period)
                > config.merge_tolerance
            ):
                raise SurfaceExtractionError(
                    SurfaceStatus.PERIODIC_MISMATCH,
                    "periodic seam roots do not coincide within merge_tolerance",
                )
    triangle_array = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if len(triangle_array):
        keys = [tuple(sorted(map(int, triangle))) for triangle in triangle_array]
        if len(set(keys)) != len(keys):
            raise SurfaceExtractionError(
                SurfaceStatus.DEGENERATE,
                "periodic identification produced duplicate surface triangles",
            )
    triangle_array = _orient_triangles(point_array, triangle_array, field, period)
    B_values = _evaluate_B(field, point_array)
    if len(B_values) and np.max(np.abs(B_values - b)) > config.B_tolerance:
        raise SurfaceExtractionError(
            SurfaceStatus.ROOT_FAILURE,
            "a polished surface vertex exceeds B_tolerance",
        )
    g = _physical_g(field, point_array)
    tags = np.asarray(point_tags, dtype=np.int64)
    tags[np.abs(g) <= config.g_tolerance] |= SurfaceMesh.G_ZERO
    return SurfaceMesh(
        level=b,
        period=period,
        points=point_array,
        triangles=triangle_array,
        B=B_values,
        g=g,
        boundary_tags=tags,
        point_parent_edges=np.asarray(parent_edges, dtype=np.int64).reshape(-1, 2),
        triangle_parent_tetrahedra=np.asarray(parent_tetrahedra, dtype=np.int64),
        component_ids=_component_ids(triangle_array),
    )


def _polish_background_edges(
    background: BackgroundMesh,
    edges: list[tuple[int, int]],
    field: BoozerFieldLike,
    b: float,
    period: float,
    config: SurfaceExtractionConfig,
) -> np.ndarray:
    """Polish the bracketed ``B=b`` root on every background edge at once.

    Each edge is solved on its own chord by the vectorized Chandrupatla
    iteration, to the same parameter tolerance as the previous per-edge
    ``brentq`` calls, so one batched field evaluation per iteration serves
    all still-unconverged edges. Returns canonicalized points, one per edge.
    """
    if not len(edges):
        return np.empty((0, 3), dtype=np.float64)
    edge_array = np.asarray(edges, dtype=np.int64)
    firsts = np.asarray(background.points[edge_array[:, 0]], dtype=np.float64)
    seconds = np.asarray(background.points[edge_array[:, 1]], dtype=np.float64).copy()
    difference = seconds[:, 2] - firsts[:, 2]
    seconds[:, 2] -= period * np.round(difference / period)
    directions = seconds - firsts

    def residuals(parameters: np.ndarray, active: np.ndarray) -> np.ndarray:
        points = firsts[active] + parameters[:, np.newaxis] * directions[active]
        return _evaluate_B(field, points) - b

    endpoint_values = _evaluate_B(field, np.vstack((firsts, seconds))) - b
    lower_values = endpoint_values[: len(edge_array)]
    upper_values = endpoint_values[len(edge_array) :]
    if np.any(lower_values * upper_values >= 0.0):
        raise SurfaceExtractionError(
            SurfaceStatus.ROOT_FAILURE,
            "background samples bracket a root that the analytic field does not",
        )
    parameters, unconverged = _chandrupatla_roots(
        residuals, lower_values, upper_values, config
    )
    if np.any(unconverged):
        raise SurfaceExtractionError(
            SurfaceStatus.ROOT_FAILURE, "bracketed edge-root polishing failed"
        )
    points = firsts + parameters[:, np.newaxis] * directions
    if np.max(np.abs(_evaluate_B(field, points) - b)) > config.B_tolerance:
        raise SurfaceExtractionError(
            SurfaceStatus.ROOT_FAILURE,
            "edge-root residual exceeds B_tolerance after polishing",
        )
    return _canonicalize_points(points, period, config.merge_tolerance)


def _polish_pyvista_background_edge_roots(
    contour_points: np.ndarray,
    background: BackgroundMesh,
    field: BoozerFieldLike,
    b: float,
    period: float,
    config: SurfaceExtractionConfig,
) -> np.ndarray:
    """Polish each VTK contour point on its originating background edge.

    VTK linearly interpolates one contour point on every tetrahedron edge
    whose sampled endpoint values straddle ``b``.  A free multidimensional
    Newton projection can leave that edge and converge to another nearby
    sheet of a nonlinear ``B=b`` surface.  Reconstruct the originating edge
    from VTK's linear point, then use the same bracketed chord solve as the
    marching-tetrahedra extractor.  Parent IDs remain intentionally absent
    from the returned PyVista prototype; the reconstructed edges only bound
    root polishing (DESIGN.md §§8.3 and 21.2).
    """
    points = np.asarray(contour_points, dtype=np.float64).reshape(-1, 3)
    values = np.asarray(background.B, dtype=np.float64) - b
    if np.any(np.abs(values) <= config.B_tolerance):
        raise SurfaceExtractionError(
            SurfaceStatus.DEGENERATE,
            "B=b passes through a background vertex; refine or perturb the slice",
        )
    local_edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    active_edges = sorted(
        {
            tuple(sorted((int(tetrahedron[first]), int(tetrahedron[second]))))
            for tetrahedron in background.tetrahedra
            for first, second in local_edges
            if values[tetrahedron[first]] * values[tetrahedron[second]] < 0.0
        }
    )
    if len(points) != len(active_edges):
        raise SurfaceExtractionError(
            SurfaceStatus.DEGENERATE,
            "PyVista contour points do not correspond one-to-one with active "
            "background edges",
        )
    if not len(points):
        return points
    edge_array = np.asarray(active_edges, dtype=np.int64)
    first = edge_array[:, 0]
    second = edge_array[:, 1]
    fraction = -values[first] / (values[second] - values[first])
    linear_points = background.points[first] + fraction[:, np.newaxis] * (
        background.points[second] - background.points[first]
    )
    distances, edge_ids = cKDTree(linear_points).query(points)
    coordinate_scale = max(
        1.0,
        float(np.max(np.abs(points))),
        float(np.max(np.abs(linear_points))),
    )
    vtk_tolerance = max(
        config.merge_tolerance,
        8.0 * np.sqrt(np.finfo(float).eps) * coordinate_scale,
    )
    if np.any(distances > vtk_tolerance) or len(np.unique(edge_ids)) != len(points):
        raise SurfaceExtractionError(
            SurfaceStatus.DEGENERATE,
            "PyVista contour geometry cannot be matched uniquely to background edges",
        )
    polished = _polish_background_edges(
        background, active_edges, field, b, period, config
    )
    return polished[edge_ids]


def _chandrupatla_roots(
    residuals,
    lower_values: np.ndarray,
    upper_values: np.ndarray,
    config: SurfaceExtractionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized bracketed root find on the unit parameter interval.

    Chandrupatla's algorithm (inverse quadratic interpolation inside a
    guaranteed bracket, bisection otherwise) with elementwise state, so one
    call of ``residuals(parameters, active_mask)`` per iteration serves every
    still-active bracket. Every ``(lower, upper)`` pair must already straddle
    a sign change. Returns ``(roots, unconverged_mask)``; the caller decides
    what an unconverged bracket means.
    """
    n = len(lower_values)
    x1 = np.zeros(n)
    f1 = np.asarray(lower_values, dtype=np.float64).copy()
    x2 = np.ones(n)
    f2 = np.asarray(upper_values, dtype=np.float64).copy()
    x3 = np.zeros(n)
    f3 = np.zeros(n)
    roots = np.where(np.abs(f1) <= np.abs(f2), x1, x2)
    t = np.full(n, 0.5)
    active = np.ones(n, dtype=bool)
    rtol = 4.0 * np.finfo(float).eps
    for _ in range(100):
        x = x1 + t * (x2 - x1)
        f = np.zeros(n)
        f[active] = residuals(x[active], active)
        same = np.sign(f) == np.sign(f1)
        move = active & same
        x3[move] = x1[move]
        f3[move] = f1[move]
        swap = active & ~same
        x3[swap] = x2[swap]
        f3[swap] = f2[swap]
        x2[swap] = x1[swap]
        f2[swap] = f1[swap]
        x1[active] = x[active]
        f1[active] = f[active]
        best = np.where(np.abs(f1) <= np.abs(f2), x1, x2)
        roots[active] = best[active]
        spread = np.abs(x2 - x1)
        tolerance = config.parameter_tolerance + rtol * np.abs(best)
        active &= ~((f == 0.0) | (spread <= 2.0 * tolerance))
        if not active.any():
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = (x1 - x2) / (x3 - x2)
            phi = (f1 - f2) / (f3 - f2)
            interpolate = (phi**2 < xi) & ((1.0 - phi) ** 2 < 1.0 - xi)
            candidate = f1 / (f1 - f2) * (f3 / (f3 - f2)) - (x3 - x1) / (x2 - x1) * (
                f1 / (f3 - f1)
            ) * (f2 / (f2 - f3))
        t = np.where(interpolate & np.isfinite(candidate), candidate, 0.5)
        limit = tolerance / np.maximum(spread, np.finfo(float).tiny)
        t = np.clip(t, limit, 1.0 - limit)
    return roots, active


def _split_by_physical_g(
    full: SurfaceMesh,
    field: BoozerFieldLike,
    config: SurfaceExtractionConfig,
) -> SurfaceExtraction:
    crossing_cache: dict[tuple[str, int, int], dict[str, object]] = {}
    # The surface patch a marching triangle approximates lives inside one
    # background tetrahedron, so the largest marching-triangle edge is the
    # locality scale for split-point polishing on edges much shorter than a
    # background cell.
    patch_scale = _max_triangle_edge_length(full)

    def vertex_record(vertex: int) -> dict[str, object]:
        key = ("v", int(vertex), int(vertex))
        if key not in crossing_cache:
            crossing_cache[key] = {
                "key": key,
                "point": full.points[vertex],
                "B": float(full.B[vertex]),
                "g": float(full.g[vertex]),
                "tag": int(full.boundary_tags[vertex]),
                "parent_edge": full.point_parent_edges[vertex],
            }
        return crossing_cache[key]

    def crossing_record(first: int, second: int) -> dict[str, object]:
        if abs(full.g[first]) <= config.g_tolerance:
            return vertex_record(first)
        if abs(full.g[second]) <= config.g_tolerance:
            return vertex_record(second)
        lower, upper = sorted((int(first), int(second)))
        key = ("e", lower, upper)
        if key not in crossing_cache:
            point, resolved = _polish_g_crossing(
                full.points[first],
                full.points[second],
                float(full.g[first]),
                float(full.g[second]),
                field,
                full.level,
                full.period,
                config,
                patch_scale=patch_scale,
            )
            B_value = float(_evaluate_B(field, point[np.newaxis, :])[0])
            g_value = float(_physical_g(field, point[np.newaxis, :])[0])
            tag = SurfaceMesh.G_ZERO if resolved else SurfaceMesh.G_JUMP
            for bit in (
                SurfaceMesh.EDGE,
                SurfaceMesh.AXIS,
                SurfaceMesh.PERIODIC_SEAM,
            ):
                if (full.boundary_tags[first] & bit) and (
                    full.boundary_tags[second] & bit
                ):
                    tag |= bit
            # A split point polished onto the outer boundary (where the g=0
            # curve exits the domain) is an EDGE point regardless of its
            # parent vertices' provenance.
            if abs(float(np.linalg.norm(point[:2])) - 1.0) <= config.merge_tolerance:
                tag |= SurfaceMesh.EDGE
            crossing_cache[key] = {
                "key": key,
                "point": point,
                "B": B_value,
                "g": g_value,
                "tag": tag,
                "parent_edge": np.array([-1, -1], dtype=np.int64),
            }
        return crossing_cache[key]

    def clipped_polygon(triangle: np.ndarray, incoming: bool):
        polygon = [vertex_record(int(vertex)) for vertex in triangle]

        def inside(record: dict[str, object]) -> bool:
            value = float(record["g"])
            return (
                value <= config.g_tolerance
                if incoming
                else value >= -config.g_tolerance
            )

        output: list[dict[str, object]] = []
        for current_index, current in enumerate(polygon):
            following = polygon[(current_index + 1) % len(polygon)]
            current_inside = inside(current)
            following_inside = inside(following)
            if current_inside:
                output.append(current)
            if current_inside != following_inside:
                first = int(current["key"][1])
                second = int(following["key"][1])
                output.append(crossing_record(first, second))
        deduplicated: list[dict[str, object]] = []
        for record in output:
            if not deduplicated or record["key"] != deduplicated[-1]["key"]:
                deduplicated.append(record)
        if len(deduplicated) > 1 and deduplicated[0]["key"] == deduplicated[-1]["key"]:
            deduplicated.pop()
        return deduplicated

    incoming = _build_clipped_surface(full, True, clipped_polygon, config)
    outgoing = _build_clipped_surface(full, False, clipped_polygon, config)
    g_zero = _build_g_zero_curves(full, crossing_record, vertex_record, config)
    n_unresolved = sum(
        1
        for record in crossing_cache.values()
        if int(record["tag"]) & SurfaceMesh.G_JUMP
    )
    return SurfaceExtraction(
        b=full.level,
        full=full,
        incoming=incoming,
        outgoing=outgoing,
        g_zero=g_zero,
        status=(SurfaceStatus.UNRESOLVED if n_unresolved else SurfaceStatus.REGULAR),
        n_unresolved_splits=n_unresolved,
    )


def _build_clipped_surface(full, incoming, clipped_polygon, config) -> SurfaceMesh:
    points: list[np.ndarray] = []
    B_values: list[float] = []
    g_values: list[float] = []
    tags: list[int] = []
    parent_edges: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    parent_tetrahedra: list[int] = []
    point_ids: dict[tuple[str, int, int], int] = {}

    def point_id(record: dict[str, object]) -> int:
        key = record["key"]
        if key not in point_ids:
            point_ids[key] = len(points)
            points.append(np.asarray(record["point"], dtype=np.float64))
            B_values.append(float(record["B"]))
            g_values.append(float(record["g"]))
            tag = int(record["tag"])
            if abs(float(record["g"])) <= config.g_tolerance:
                tag |= SurfaceMesh.G_ZERO
            tags.append(tag)
            parent_edges.append(np.asarray(record["parent_edge"], dtype=np.int64))
        return point_ids[key]

    for triangle_id, triangle in enumerate(full.triangles):
        polygon = clipped_polygon(triangle, incoming)
        if len(polygon) < 3:
            continue
        ids = [point_id(record) for record in polygon]
        for index in range(1, len(ids) - 1):
            candidate = (ids[0], ids[index], ids[index + 1])
            if len(set(candidate)) != 3:
                continue
            triangles.append(candidate)
            parent_tetrahedra.append(int(full.triangle_parent_tetrahedra[triangle_id]))

    point_array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    triangle_array = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    return SurfaceMesh(
        level=full.level,
        period=full.period,
        points=point_array,
        triangles=triangle_array,
        B=np.asarray(B_values, dtype=np.float64),
        g=np.asarray(g_values, dtype=np.float64),
        boundary_tags=np.asarray(tags, dtype=np.int64),
        point_parent_edges=np.asarray(parent_edges, dtype=np.int64).reshape(-1, 2),
        triangle_parent_tetrahedra=np.asarray(parent_tetrahedra, dtype=np.int64),
        component_ids=_component_ids(triangle_array),
    )


def _build_g_zero_curves(
    full: SurfaceMesh, crossing_record, vertex_record, config: SurfaceExtractionConfig
) -> SurfaceCurveMesh:
    point_ids: dict[tuple[str, int, int], int] = {}
    points: list[np.ndarray] = []
    B_values: list[float] = []
    g_values: list[float] = []
    tags: list[int] = []
    segments: list[tuple[int, int]] = []
    segment_keys: set[tuple[tuple[str, int, int], tuple[str, int, int]]] = set()

    def curve_point_id(record: dict[str, object]) -> int:
        key = record["key"]
        if key not in point_ids:
            point_ids[key] = len(points)
            points.append(np.asarray(record["point"], dtype=np.float64))
            B_values.append(float(record["B"]))
            g_values.append(float(record["g"]))
            tags.append(int(record["tag"]) | SurfaceMesh.G_ZERO)
        return point_ids[key]

    for triangle in full.triangles:
        records: list[dict[str, object]] = []
        for vertex in triangle:
            if abs(full.g[vertex]) <= config.g_tolerance:
                records.append(vertex_record(int(vertex)))
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            if full.g[first] * full.g[second] < 0.0:
                records.append(crossing_record(int(first), int(second)))
        unique = {record["key"]: record for record in records}
        records = list(unique.values())
        if len(records) > 2:
            raise SurfaceExtractionError(
                SurfaceStatus.DEGENERATE,
                "a surface triangle contains an unresolved multiway g=0 event",
            )
        # A G_JUMP split vertex is not a g=0 point; the g=0 curve honestly
        # omits it rather than drawing a curve segment that is not there.
        records = [
            record for record in records if int(record["tag"]) & SurfaceMesh.G_ZERO
        ]
        if len(records) != 2:
            continue
        key = tuple(sorted((records[0]["key"], records[1]["key"])))
        if key in segment_keys:
            continue
        segment_keys.add(key)
        segments.append((curve_point_id(records[0]), curve_point_id(records[1])))

    return SurfaceCurveMesh(
        period=full.period,
        points=np.asarray(points, dtype=np.float64).reshape(-1, 3),
        segments=np.asarray(segments, dtype=np.int64).reshape(-1, 2),
        B=np.asarray(B_values, dtype=np.float64),
        g=np.asarray(g_values, dtype=np.float64),
        boundary_tags=np.asarray(tags, dtype=np.int64),
    )


def _polish_g_crossing(
    first: np.ndarray,
    second: np.ndarray,
    first_g: float,
    second_g: float,
    field: BoozerFieldLike,
    b: float,
    period: float,
    config: SurfaceExtractionConfig,
    patch_scale: float = 0.0,
) -> tuple[np.ndarray, bool]:
    """Locate the split point on one bracketed surface edge (§8.3).

    Returns ``(point, resolved)``. When ``resolved`` is true the point
    satisfies ``B=b`` and ``g=0`` to tolerance — found by the planar Newton
    fast path, by the bracketed projected-chord solve, by a pencil of
    plane-curve continuations returning the crossing nearest the edge, or,
    when the ``g=0`` curve exits the domain near the outer boundary, on the
    ``x^2+y^2=1`` cylinder (ADR 0001).

    When every one of those bracketed searches proves that no local ``g=0``
    point exists — a marching triangle bridging two sheets of an
    under-resolved surface, whose genuine crossings are many cells away —
    ``resolved`` is false and the point is the ``g`` sign discontinuity of
    the projected chord path: on the surface (``B=b`` to tolerance), local
    by construction, but with ``g != 0``. Callers must record such points
    as ``G_JUMP``, not ``G_ZERO``.

    ``patch_scale`` is the size of the surface patch one background cell
    holds (the largest marching-triangle edge); it keeps the fallbacks'
    search range and acceptance radius meaningful on marching edges much
    shorter than a background cell.
    """
    if first_g * second_g >= 0.0:
        raise SurfaceExtractionError(
            SurfaceStatus.ROOT_FAILURE, "g=0 surface edge is not bracketed"
        )
    second_local = _unwrap_point_relative(second, first, period)
    tangent = second_local - first
    displacement_limit = config.merge_tolerance + float(np.linalg.norm(tangent))
    point = _polish_g_crossing_planar(
        first, tangent, first_g, second_g, field, b, config, displacement_limit
    )
    if point is None:
        point = _polish_g_crossing_bracketed(
            first,
            tangent,
            first_g,
            second_g,
            field,
            b,
            config,
            displacement_limit,
            patch_scale,
        )
    if point is not None:
        return _canonicalize_point(point, period, config.merge_tolerance), True
    point = _locate_g_jump_on_chord(
        first, tangent, first_g, second_g, field, b, config, displacement_limit
    )
    if point is None:
        raise SurfaceExtractionError(
            SurfaceStatus.ROOT_FAILURE,
            "simultaneous B=b and g=0 split-point polishing failed: the "
            "planar Newton solve, the bracketed fallbacks, and the g-jump "
            "locator were all rejected by the locality and residual checks "
            f"(edge length={np.linalg.norm(tangent):.3g}, "
            f"g bracket=({first_g:.3g}, {second_g:.3g}))",
        )
    return _canonicalize_point(point, period, config.merge_tolerance), False


def _locate_g_jump_on_chord(
    first: np.ndarray,
    tangent: np.ndarray,
    first_g: float,
    second_g: float,
    field: BoozerFieldLike,
    b: float,
    config: SurfaceExtractionConfig,
    displacement_limit: float,
) -> np.ndarray | None:
    """Bisect the chord to the ``g`` sign discontinuity of its projection.

    Used only after the bracketed searches have established that no local
    ``g=0`` point exists: the marching edge bridges two surface sheets of
    opposite ``g`` sign, and the projection of the chord onto ``B=b`` jumps
    between them at some chord parameter. Sign bisection converges to that
    parameter regardless of the discontinuity, and the returned projected
    point lies on one of the two sheets right at the gap — the honest
    seam for partitioning a triangle whose true ``g=0`` curve is elsewhere.
    """

    def project(parameter: float) -> np.ndarray | None:
        return _project_to_level_near(
            first + parameter * tangent, field, b, config, displacement_limit
        )

    # Sample the projectable part of the chord; the projection can stall in
    # the vanishing-gradient interior of the tube, so failed samples are
    # simply skipped. The endpoints are already on the surface with known
    # opposite signs, so an adjacent sign change always exists.
    samples: list[tuple[float, np.ndarray, float]] = [
        (0.0, np.asarray(first, dtype=np.float64), float(first_g))
    ]
    for parameter in np.linspace(0.0, 1.0, 65)[1:-1]:
        candidate = project(float(parameter))
        if candidate is not None:
            samples.append(
                (
                    float(parameter),
                    candidate,
                    float(_physical_g(field, candidate[np.newaxis, :])[0]),
                )
            )
    samples.append((1.0, first + tangent, float(second_g)))
    flip = None
    for left, right in zip(samples, samples[1:]):
        if np.sign(left[2]) != np.sign(right[2]):
            flip = (left, right)
            break
    if flip is None:
        return None
    (t_low, p_low, g_low), (t_high, p_high, g_high) = flip
    for _ in range(40):
        if t_high - t_low <= config.parameter_tolerance:
            break
        t_mid = 0.5 * (t_low + t_high)
        candidate = project(t_mid)
        if candidate is None:
            break
        g_mid = float(_physical_g(field, candidate[np.newaxis, :])[0])
        if np.sign(g_mid) == np.sign(g_low):
            t_low, p_low, g_low = t_mid, candidate, g_mid
        else:
            t_high, p_high, g_high = t_mid, candidate, g_mid
    # Both bracket points sit at the gap on their own sheets. Prefer a
    # genuine interior point over a pre-existing endpoint, but a sheet can
    # bulge outside the unit disk near the outer boundary, so fall back
    # through the remaining samples by proximity to the flip; the in-disk
    # edge endpoints guarantee an acceptable candidate exists.
    t_flip = 0.5 * (t_low + t_high)
    preferred: list[np.ndarray] = []
    if t_low <= 0.0:
        preferred = [p_high, p_low]
    elif t_high >= 1.0:
        preferred = [p_low, p_high]
    else:
        preferred = [p_low, p_high] if abs(g_low) <= abs(g_high) else [p_high, p_low]
    fallbacks = [
        sample[1]
        for sample in sorted(samples, key=lambda sample: abs(sample[0] - t_flip))
    ]
    for point in preferred + fallbacks:
        if (
            float(np.sum(point[:2] ** 2)) <= 1.0 + config.merge_tolerance
            and abs(float(_evaluate_B(field, point[np.newaxis, :])[0] - b))
            <= config.B_tolerance
        ):
            return point
    return None


def _polish_g_crossing_planar(
    first: np.ndarray,
    tangent: np.ndarray,
    first_g: float,
    second_g: float,
    field: BoozerFieldLike,
    b: float,
    config: SurfaceExtractionConfig,
    displacement_limit: float,
) -> np.ndarray | None:
    """Fast planar Newton solve; ``None`` when its checks reject the root."""
    initial_parameter = first_g / (first_g - second_g)
    initial = first + initial_parameter * tangent
    gradient = _logical_B_gradient(field, initial)
    gradient_norm = np.linalg.norm(gradient)
    if gradient_norm <= np.finfo(float).eps:
        return None
    normal = gradient / gradient_norm

    def equations(parameters: np.ndarray) -> np.ndarray:
        point = initial + parameters[0] * tangent + parameters[1] * normal
        return np.array(
            [
                _evaluate_B(field, point[np.newaxis, :])[0] - b,
                _physical_g(field, point[np.newaxis, :])[0],
            ]
        )

    solution = root(
        equations,
        np.zeros(2),
        method="hybr",
        options={"xtol": config.parameter_tolerance},
    )
    parameter = initial_parameter + solution.x[0]
    point = initial + solution.x[0] * tangent + solution.x[1] * normal
    residual = equations(solution.x)
    if (
        not np.all(np.isfinite(solution.x))
        or not -config.merge_tolerance <= parameter <= 1.0 + config.merge_tolerance
        or abs(solution.x[1]) > displacement_limit
        or abs(residual[0]) > config.B_tolerance
        or abs(residual[1]) > config.g_tolerance
        or float(np.sum(point[:2] ** 2)) > 1.0 + config.merge_tolerance
    ):
        return None
    return point


def _polish_g_crossing_bracketed(
    first: np.ndarray,
    tangent: np.ndarray,
    first_g: float,
    second_g: float,
    field: BoozerFieldLike,
    b: float,
    config: SurfaceExtractionConfig,
    displacement_limit: float,
    patch_scale: float,
) -> np.ndarray | None:
    """Bracketed fallback that cannot leave the edge neighborhood.

    Both endpoints lie on ``B=b`` with ``g`` of opposite signs, so ``g``
    after projecting a chord point back onto ``B=b`` is a bracketed scalar
    function of the chord parameter — unless the surface curves so strongly
    between the endpoints (a thin tube near ``min B``) that the projection
    jumps sheets and the projected ``g`` is discontinuous. When the direct
    chord solve is rejected for that reason, crossings are collected by
    tracing the intersection curves of ``B=b`` with a pencil of planes
    through the edge, and the crossing nearest the edge is returned.

    The surface patch that the parent triangle approximates lies inside its
    background tetrahedron and hence inside the unit disk; every accepted
    candidate must stay inside the disk and within the locality scale — a
    few edge lengths, or twice the background patch scale for marching
    edges much shorter than their background cell. Near the outer boundary
    the surface's ``g=0`` curve may leave the domain entirely, in which
    case no interior crossing exists and the split point is where that
    curve exits: the ``g`` flip along the surface's boundary curve on the
    ``x^2+y^2=1`` cylinder. A distant or out-of-domain root is never
    returned (§21.2).
    """
    second = first + tangent
    edge_length = float(np.linalg.norm(tangent))
    acceptance_radius = config.merge_tolerance + max(
        4.0 * edge_length, 2.0 * patch_scale
    )
    point = _brentq_projected_chord(first, second, first_g, second_g, field, b, config)
    if point is not None and (
        _distance_to_segment(point, first, second) <= displacement_limit
    ):
        return point
    point = _trace_plane_curve_to_g_zero(
        first, second, first_g, second_g, field, b, config, patch_scale
    )
    if point is not None and (
        _distance_to_segment(point, first, second) <= acceptance_radius
    ):
        return point
    if (
        max(np.linalg.norm(first[:2]), np.linalg.norm(second[:2])) + acceptance_radius
        >= 1.0
    ):
        point = _polish_g_zero_on_outer_boundary(
            first, second, first_g, second_g, field, b, config, patch_scale
        )
        if point is not None and (
            _distance_to_segment(point, first, second) <= acceptance_radius
        ):
            return point
    return None


def _polish_g_zero_on_outer_boundary(
    first: np.ndarray,
    second: np.ndarray,
    first_g: float,
    second_g: float,
    field: BoozerFieldLike,
    b: float,
    config: SurfaceExtractionConfig,
    patch_scale: float,
) -> np.ndarray | None:
    """Find where the surface's ``g=0`` curve exits through ``x^2+y^2=1``.

    On the outer cylinder, parametrized by ``(theta, zeta)`` with
    ``(x, y) = (cos theta, sin theta)``, the level set ``B(1, theta,
    zeta) = b`` is a one-dimensional implicit curve — the boundary curve of
    the domain-clipped surface. When the interior ``g=0`` curve leaves the
    domain near the edge, ``g`` restricted to this boundary curve changes
    sign at the exit point, so the same predictor-corrector continuation
    used for the in-plane traces applies in the chart, seeded at an edge
    endpoint that lies on the boundary (or at the projected mid-chord).
    Every candidate lies on the boundary exactly, so it is in-domain by
    construction.
    """
    length = float(np.linalg.norm(second - first))
    scale = max(length, 0.5 * patch_scale)

    def cylinder_point(theta: float, zeta: float) -> np.ndarray:
        return np.array([np.cos(theta), np.sin(theta), zeta])

    def cylinder_residual(theta: float, zeta: float) -> float:
        return float(field.B(1.0, theta, zeta)) - b

    def cylinder_gradient(theta: float, zeta: float) -> np.ndarray:
        dB_dtheta, dB_dzeta = _field_quantities(
            field, 1.0, theta, zeta, ("dB_dtheta", "dB_dzeta")
        )
        return np.array([float(dB_dtheta), float(dB_dzeta)])

    def correct(theta: float, zeta: float, cap: float) -> tuple[float, float] | None:
        position = np.array([theta, zeta])
        start_position = position.copy()
        residual = cylinder_residual(*position)
        for _ in range(30):
            if abs(residual) <= config.B_tolerance:
                return float(position[0]), float(position[1])
            gradient2 = cylinder_gradient(*position)
            norm_squared = float(np.dot(gradient2, gradient2))
            if norm_squared <= np.finfo(float).eps:
                return None
            step = (-residual / norm_squared) * gradient2
            improved = None
            for _halving in range(20):
                trial = position + step
                if np.linalg.norm(trial - start_position) > cap:
                    step = 0.5 * step
                    continue
                trial_residual = cylinder_residual(*trial)
                if abs(trial_residual) < abs(residual):
                    improved = (trial, trial_residual)
                    break
                step = 0.5 * step
            if improved is None:
                return None
            position, residual = improved
        return None

    def chart_g(theta: float, zeta: float) -> float:
        return float(_physical_g(field, cylinder_point(theta, zeta)[np.newaxis, :])[0])

    # Seed on the boundary curve: an edge endpoint already on the cylinder
    # is exact; otherwise correct the projected mid-chord onto the curve.
    seed: tuple[float, float] | None = None
    seed_g = 0.0
    for endpoint, endpoint_g in ((first, first_g), (second, second_g)):
        if abs(float(np.linalg.norm(endpoint[:2])) - 1.0) <= config.merge_tolerance:
            seed = (
                float(np.arctan2(endpoint[1], endpoint[0])),
                float(endpoint[2]),
            )
            seed_g = float(endpoint_g)
            break
    if seed is None:
        middle = 0.5 * (first + second)
        corrected = correct(
            float(np.arctan2(middle[1], middle[0])),
            float(middle[2]),
            cap=2.0 * scale,
        )
        if corrected is None:
            return None
        seed = corrected
        seed_g = chart_g(*seed)
        if abs(seed_g) <= config.g_tolerance:
            return cylinder_point(*seed)

    def polish_between(
        start: np.ndarray, start_g: float, end: np.ndarray, end_g: float
    ) -> np.ndarray | None:
        span = end - start
        cap = float(np.linalg.norm(span)) + config.merge_tolerance

        def bracketed_g(parameter: float) -> float:
            if parameter <= 0.0:
                return start_g
            if parameter >= 1.0:
                return end_g
            sample = start + parameter * span
            corrected = correct(sample[0], sample[1], cap)
            if corrected is None:
                raise _LocalProjectionFailure()
            return chart_g(*corrected)

        try:
            parameter = brentq(
                bracketed_g,
                0.0,
                1.0,
                xtol=config.parameter_tolerance,
                rtol=4.0 * np.finfo(float).eps,
                maxiter=200,
            )
        except (_LocalProjectionFailure, ValueError):
            return None
        sample = start + float(np.clip(parameter, 0.0, 1.0)) * span
        corrected = correct(sample[0], sample[1], cap)
        if corrected is None:
            return None
        point = cylinder_point(*corrected)
        if (
            abs(float(_evaluate_B(field, point[np.newaxis, :])[0] - b))
            > config.B_tolerance
            or abs(chart_g(*corrected)) > config.g_tolerance
        ):
            return None
        return point

    nominal_step = scale / 16.0
    arc_budget = 8.0 * scale
    candidates: list[np.ndarray] = []
    for orientation in (1.0, -1.0):
        position = np.array(seed, dtype=np.float64)
        position_g = seed_g
        previous_direction: np.ndarray | None = None
        traveled = 0.0
        step_size = nominal_step
        for _ in range(1024):
            if traveled > arc_budget or len(candidates) >= 8:
                break
            gradient2 = cylinder_gradient(*position)
            norm2 = float(np.linalg.norm(gradient2))
            if norm2 <= np.finfo(float).eps:
                break
            tangent2 = np.array([-gradient2[1], gradient2[0]]) / norm2
            if previous_direction is None:
                tangent2 = orientation * tangent2
            elif float(np.dot(tangent2, previous_direction)) < 0.0:
                tangent2 = -tangent2
            corrected = None
            while step_size >= nominal_step * 2.0**-20:
                trial = position + step_size * tangent2
                corrected = correct(trial[0], trial[1], cap=step_size)
                if corrected is not None:
                    break
                step_size *= 0.5
            if corrected is None:
                break
            following = np.array(corrected, dtype=np.float64)
            advance = float(np.linalg.norm(following - position))
            if advance <= config.merge_tolerance:
                break
            following_g = chart_g(*following)
            if abs(following_g) <= config.g_tolerance:
                candidates.append(cylinder_point(*following))
            elif position_g * following_g < 0.0:
                candidate = polish_between(position, position_g, following, following_g)
                if candidate is not None:
                    candidates.append(candidate)
            previous_direction = (following - position) / advance
            traveled += advance
            position, position_g = following, following_g
            step_size = min(2.0 * step_size, nominal_step)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: _distance_to_segment(candidate, first, second),
    )


def _trace_plane_curve_to_g_zero(
    first: np.ndarray,
    second: np.ndarray,
    first_g: float,
    second_g: float,
    field: BoozerFieldLike,
    b: float,
    config: SurfaceExtractionConfig,
    patch_scale: float,
) -> np.ndarray | None:
    """Trace ``B=b`` within planes through the edge, return the nearest
    ``g=0`` crossing.

    A plane containing the chord cuts the level set in a one-dimensional
    implicit curve; every point of the curve lies on the surface. When the
    cut is transversal (the tube cross-section for ``b`` near ``min B``)
    that curve passes through both endpoints, ``g`` restricted to it is
    continuous with opposite signs at the endpoints, and the direct arc
    carries a crossing near the edge. A near-axial plane instead cuts the
    tube in two disconnected curves, one per endpoint, with no bracket on
    either — so a pencil of planes rotated about the chord is tried, seeded
    by the ``grad B`` direction. Predictor steps follow the in-plane curve
    tangent with orientation continuity — no step-toward-the-target
    heuristic, which stalls when the far endpoint sits on the opposite
    sheet — and corrector steps are damped in-plane Newton back onto
    ``B=b``. Every sign flip met in either direction of every plane is
    polished on its short bracketing sub-chord, and the candidate closest
    to the edge wins. A direction ends at the far endpoint, at the unit-disk
    boundary (beyond it the field is unphysical extrapolation and no valid
    crossing can live), or at the arc-length budget.
    """
    chord = second - first
    length = float(np.linalg.norm(chord))
    if length <= 0.0:
        return None
    u = chord / length
    w0 = None
    for probe in (0.5 * (first + second), first, second):
        gradient = _logical_B_gradient(field, probe)
        candidate = gradient - float(np.dot(gradient, u)) * u
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm > 1.0e-8 * max(1.0, float(np.linalg.norm(gradient))):
            w0 = candidate / candidate_norm
            break
    if w0 is None:
        return None
    w_perpendicular = np.cross(u, w0)
    # The search scale: the chord itself when the marching edge is a fair
    # sample of the background cell, the patch scale when the edge is much
    # shorter than the cell that holds the surface patch.
    scale = max(length, 0.5 * patch_scale)

    candidates: list[np.ndarray] = []
    for angle in (0.0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi):
        w = np.cos(angle) * w0 + np.sin(angle) * w_perpendicular
        candidates.extend(
            _trace_one_plane_for_g_zero(
                first, second, first_g, second_g, u, w, length, scale, field, b, config
            )
        )
        near = [
            candidate
            for candidate in candidates
            if _distance_to_segment(candidate, first, second) <= 1.5 * scale
        ]
        if near:
            break
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: _distance_to_segment(candidate, first, second),
    )


def _trace_one_plane_for_g_zero(
    first: np.ndarray,
    second: np.ndarray,
    first_g: float,
    second_g: float,
    u: np.ndarray,
    w: np.ndarray,
    length: float,
    scale: float,
    field: BoozerFieldLike,
    b: float,
    config: SurfaceExtractionConfig,
) -> list[np.ndarray]:
    """Collect polished ``g=0`` crossings on ``B=b`` in one cutting plane."""

    def plane_point(a: float, c: float) -> np.ndarray:
        return first + a * u + c * w

    def plane_residual(a: float, c: float) -> float:
        return float(_evaluate_B(field, plane_point(a, c)[np.newaxis, :])[0] - b)

    def plane_gradient(a: float, c: float) -> np.ndarray:
        gradient = _logical_B_gradient(field, plane_point(a, c))
        return np.array([float(np.dot(gradient, u)), float(np.dot(gradient, w))])

    def correct(a: float, c: float, cap: float) -> tuple[float, float] | None:
        position = np.array([a, c])
        start_position = position.copy()
        residual = plane_residual(*position)
        for _ in range(30):
            if abs(residual) <= config.B_tolerance:
                return float(position[0]), float(position[1])
            gradient2 = plane_gradient(*position)
            norm_squared = float(np.dot(gradient2, gradient2))
            if norm_squared <= np.finfo(float).eps:
                return None
            step = (-residual / norm_squared) * gradient2
            improved = None
            for _halving in range(20):
                trial = position + step
                if np.linalg.norm(trial - start_position) > cap:
                    step = 0.5 * step
                    continue
                trial_residual = plane_residual(*trial)
                if abs(trial_residual) < abs(residual):
                    improved = (trial, trial_residual)
                    break
                step = 0.5 * step
            if improved is None:
                return None
            position, residual = improved
        return None

    def collect(candidates: list[np.ndarray], point: np.ndarray | None) -> None:
        if point is not None and (
            float(np.sum(point[:2] ** 2)) <= 1.0 + config.merge_tolerance
        ):
            candidates.append(point)

    nominal_step = scale / 16.0
    arc_budget = 8.0 * scale
    candidates: list[np.ndarray] = []
    for orientation in (1.0, -1.0):
        a, c = 0.0, 0.0
        current = np.asarray(first, dtype=np.float64)
        current_g = float(first_g)
        previous_direction: np.ndarray | None = None
        traveled = 0.0
        step_size = nominal_step
        for _ in range(1024):
            if traveled > arc_budget or len(candidates) >= 8:
                break
            gradient2 = plane_gradient(a, c)
            norm2 = float(np.linalg.norm(gradient2))
            if norm2 <= np.finfo(float).eps:
                break
            tangent2 = np.array([-gradient2[1], gradient2[0]]) / norm2
            if previous_direction is None:
                if orientation * tangent2[0] < 0.0:
                    tangent2 = -tangent2
            elif float(np.dot(tangent2, previous_direction)) < 0.0:
                tangent2 = -tangent2
            corrected = None
            while step_size >= nominal_step * 2.0**-20:
                trial = (a + step_size * tangent2[0], c + step_size * tangent2[1])
                corrected = correct(*trial, cap=step_size)
                if corrected is not None:
                    break
                step_size *= 0.5
            if corrected is None:
                break
            a_next, c_next = corrected
            advance = float(np.hypot(a_next - a, c_next - c))
            if advance <= config.merge_tolerance:
                break
            point = plane_point(a_next, c_next)
            if float(np.sum(point[:2] ** 2)) > 1.0:
                # The trace left the plasma domain; the field beyond the unit
                # disk is extrapolation, so no valid crossing lies this way.
                break
            point_g = float(_physical_g(field, point[np.newaxis, :])[0])
            if abs(point_g) <= config.g_tolerance:
                collect(candidates, point)
            elif current_g * point_g < 0.0:
                collect(
                    candidates,
                    _brentq_projected_chord(
                        current, point, current_g, point_g, field, b, config
                    ),
                )
            if np.hypot(a_next - length, c_next) <= step_size:
                if point_g * second_g < 0.0:
                    collect(
                        candidates,
                        _brentq_projected_chord(
                            point, second, point_g, second_g, field, b, config
                        ),
                    )
                break
            previous_direction = (
                np.array([a_next - a, c_next - c], dtype=np.float64) / advance
            )
            traveled += advance
            a, c = a_next, c_next
            current, current_g = point, point_g
            step_size = min(2.0 * step_size, nominal_step)
    return candidates


def _brentq_projected_chord(
    left: np.ndarray,
    right: np.ndarray,
    left_g: float,
    right_g: float,
    field: BoozerFieldLike,
    b: float,
    config: SurfaceExtractionConfig,
) -> np.ndarray | None:
    """Solve ``g=0`` along one surface chord via projection; ``None`` on any
    projection failure or residual-check rejection."""
    tangent = right - left
    cap = float(np.linalg.norm(tangent)) + config.merge_tolerance

    def projected(parameter: float) -> np.ndarray | None:
        return _project_to_level_near(left + parameter * tangent, field, b, config, cap)

    def projected_g(parameter: float) -> float:
        if parameter <= 0.0:
            return left_g
        if parameter >= 1.0:
            return right_g
        point = projected(parameter)
        if point is None:
            raise _LocalProjectionFailure()
        return float(_physical_g(field, point[np.newaxis, :])[0])

    try:
        parameter = brentq(
            projected_g,
            0.0,
            1.0,
            xtol=config.parameter_tolerance,
            rtol=4.0 * np.finfo(float).eps,
            maxiter=200,
        )
    except (_LocalProjectionFailure, ValueError):
        return None
    point = projected(float(np.clip(parameter, 0.0, 1.0)))
    if point is None:
        return None
    residual_B = float(_evaluate_B(field, point[np.newaxis, :])[0] - b)
    residual_g = float(_physical_g(field, point[np.newaxis, :])[0])
    if (
        abs(residual_B) > config.B_tolerance
        or abs(residual_g) > config.g_tolerance
        or float(np.sum(point[:2] ** 2)) > 1.0 + config.merge_tolerance
    ):
        return None
    return point


def _max_triangle_edge_length(surface: SurfaceMesh) -> float:
    """Largest marching-triangle edge length, with seam triangles unwrapped.

    Marching triangles live inside single background tetrahedra, so this is
    a per-extraction proxy for the background cell size in logical units.
    """
    if not len(surface.triangles):
        return 0.0
    vertices = surface.points[surface.triangles].copy()
    for index in (1, 2):
        difference = vertices[:, index, 2] - vertices[:, 0, 2]
        vertices[:, index, 2] -= surface.period * np.round(difference / surface.period)
    edges = np.stack(
        (
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 1],
            vertices[:, 0] - vertices[:, 2],
        )
    )
    return float(np.max(np.linalg.norm(edges, axis=-1)))


def _distance_to_segment(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    """Euclidean distance from ``point`` to the segment ``start``–``end``,
    all in the same locally unwrapped coordinates."""
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared == 0.0:
        return float(np.linalg.norm(point - start))
    parameter = np.clip(
        float(np.dot(point - start, direction)) / length_squared, 0.0, 1.0
    )
    return float(np.linalg.norm(point - (start + parameter * direction)))


class _LocalProjectionFailure(Exception):
    """The displacement-capped projection onto ``B=b`` did not converge."""


def _project_to_level_near(
    point: np.ndarray,
    field: BoozerFieldLike,
    b: float,
    config: SurfaceExtractionConfig,
    max_displacement: float,
) -> np.ndarray | None:
    """Project ``point`` onto ``B=b`` along the local gradient, staying local.

    Damped Newton with a monotone-residual line search; the iterate never
    moves farther than ``max_displacement`` from the seed, which pins the
    result to the ``B=b`` sheet nearest the seed. Returns ``None`` when no
    such local projection is found — never a distant substitute (§21.2).
    """
    start = np.asarray(point, dtype=np.float64)
    current = start.copy()
    residual = float(_evaluate_B(field, current[np.newaxis, :])[0] - b)
    for _ in range(60):
        if abs(residual) <= config.B_tolerance:
            return current
        gradient = _logical_B_gradient(field, current)
        norm_squared = float(np.dot(gradient, gradient))
        if norm_squared <= np.finfo(float).eps:
            return None
        step = (-residual / norm_squared) * gradient
        improved = None
        for _halving in range(30):
            trial = current + step
            if np.linalg.norm(trial - start) > max_displacement:
                step = 0.5 * step
                continue
            trial_residual = float(_evaluate_B(field, trial[np.newaxis, :])[0] - b)
            if abs(trial_residual) < abs(residual):
                improved = (trial, trial_residual)
                break
            step = 0.5 * step
        if improved is None:
            return None
        current, residual = improved
    return None


def _field_quantities(field: BoozerFieldLike, s, theta, zeta, quantities):
    """Evaluate several field quantities, fused into one Fourier pass when the
    backend offers ``fourier_quantities`` (one shared phase/trig table), and
    through the individual §7.2 protocol methods otherwise. ``quantities``
    are protocol method names; results come back as float64 arrays in
    matching order."""
    fused = getattr(field, "fourier_quantities", None)
    if fused is not None:
        return tuple(
            np.asarray(value, dtype=np.float64)
            for value in fused(s, theta, zeta, quantities)
        )
    return tuple(
        np.asarray(getattr(field, name)(s, theta, zeta), dtype=np.float64)
        for name in quantities
    )


def _evaluate_B(field: BoozerFieldLike, points: np.ndarray) -> np.ndarray:
    if not len(points):
        return np.empty(0, dtype=np.float64)
    s, theta, zeta = _logical_coordinates(points)
    return np.asarray(field.B(s, theta, zeta), dtype=np.float64)


def _physical_g(field: BoozerFieldLike, points: np.ndarray) -> np.ndarray:
    """Return physical ``b dot grad B = B D_B/(G+iota I)`` (§5.1)."""
    if not len(points):
        return np.empty(0, dtype=np.float64)
    s, theta, zeta = _logical_coordinates(points)
    B, derivative = _field_quantities(field, s, theta, zeta, ("B", "D_B"))
    current = np.asarray(field.G(s), dtype=np.float64) + np.asarray(
        field.iota(s), dtype=np.float64
    ) * np.asarray(field.I(s), dtype=np.float64)
    current_scale = max(1.0, float(np.max(np.abs(current))))
    if np.any(np.abs(current) <= 32.0 * np.finfo(float).eps * current_scale):
        raise SurfaceExtractionError(
            SurfaceStatus.DEGENERATE,
            "G + iota I is too small to determine the physical field direction",
        )
    return B * derivative / current


def _logical_coordinates(points: np.ndarray):
    s = np.sum(points[:, :2] ** 2, axis=1)
    theta = np.arctan2(points[:, 1], points[:, 0])
    theta[s == 0.0] = 0.0
    return s, theta, points[:, 2]


def _logical_B_gradient(field: BoozerFieldLike, point: np.ndarray) -> np.ndarray:
    return _logical_B_gradients(field, np.asarray(point)[np.newaxis, :])[0]


def _logical_B_gradients(field: BoozerFieldLike, points: np.ndarray) -> np.ndarray:
    """Return the logical ``(x, y, zeta)`` gradient of ``B`` at each point.

    One fused Fourier pass serves all regular points; the (rare) exact-axis
    points use the centered Cartesian differences of the §7.3 regular limit.
    """
    points = np.asarray(points, dtype=np.float64)
    s, theta, zeta = _logical_coordinates(points)
    result = np.empty((len(points), 3), dtype=np.float64)
    regular = s != 0.0
    if np.any(regular):
        dB_ds, dB_dtheta, dB_dzeta = _field_quantities(
            field,
            s[regular],
            theta[regular],
            zeta[regular],
            ("dB_ds", "dB_dtheta", "dB_dzeta"),
        )
        x = points[regular, 0]
        y = points[regular, 1]
        s_regular = s[regular]
        result[regular, 0] = 2.0 * x * dB_ds - y * dB_dtheta / s_regular
        result[regular, 1] = 2.0 * y * dB_ds + x * dB_dtheta / s_regular
        result[regular, 2] = dB_dzeta
    for index in np.flatnonzero(~regular):
        epsilon = 1.0e-6
        epsilon_s = epsilon**2
        zeta_axis = zeta[index]
        result[index, 0] = (
            float(field.B(epsilon_s, 0.0, zeta_axis))
            - float(field.B(epsilon_s, np.pi, zeta_axis))
        ) / (2.0 * epsilon)
        result[index, 1] = (
            float(field.B(epsilon_s, 0.5 * np.pi, zeta_axis))
            - float(field.B(epsilon_s, -0.5 * np.pi, zeta_axis))
        ) / (2.0 * epsilon)
        result[index, 2] = float(field.dB_dzeta(0.0, 0.0, zeta_axis))
    return result


def _edge_boundary_tag(background: BackgroundMesh, first: int, second: int) -> int:
    first_tag = int(background.boundary_tags[first])
    second_tag = int(background.boundary_tags[second])
    result = 0
    if (first_tag & BackgroundMesh.OUTER) and (second_tag & BackgroundMesh.OUTER):
        result |= SurfaceMesh.EDGE
    if (first_tag & BackgroundMesh.AXIS) and (second_tag & BackgroundMesh.AXIS):
        result |= SurfaceMesh.AXIS
    if (
        (first_tag & BackgroundMesh.ZETA_MIN) and (second_tag & BackgroundMesh.ZETA_MIN)
    ) or (
        (first_tag & BackgroundMesh.ZETA_MAX) and (second_tag & BackgroundMesh.ZETA_MAX)
    ):
        result |= SurfaceMesh.PERIODIC_SEAM
    return result


def _component_ids(triangles: np.ndarray) -> np.ndarray:
    n_triangles = len(triangles)
    if not n_triangles:
        return np.empty(0, dtype=np.int64)
    parent = np.arange(n_triangles, dtype=np.int64)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    owners: dict[tuple[int, int], int] = {}
    for triangle_id, triangle in enumerate(triangles):
        for edge in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            key = tuple(sorted(map(int, edge)))
            if key in owners:
                union(triangle_id, owners[key])
            else:
                owners[key] = triangle_id
    roots = np.array([find(index) for index in range(n_triangles)], dtype=np.int64)
    unique = {root: index for index, root in enumerate(np.unique(roots))}
    return np.array([unique[root] for root in roots], dtype=np.int64)


def _orient_triangles(
    points: np.ndarray,
    triangles: np.ndarray,
    field: BoozerFieldLike,
    period: float,
) -> np.ndarray:
    result = np.asarray(triangles, dtype=np.int64).copy()
    if not len(result):
        return result
    vertices = points[result].copy()
    for index in (1, 2):
        difference = vertices[:, index, 2] - vertices[:, 0, 2]
        vertices[:, index, 2] -= period * np.round(difference / period)
    normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    if np.any(np.linalg.norm(normals, axis=1) <= 32.0 * np.finfo(float).eps):
        raise SurfaceExtractionError(
            SurfaceStatus.DEGENERATE, "marching produced a zero-area triangle"
        )
    gradients = _logical_B_gradients(field, np.mean(vertices, axis=1))
    flip = np.einsum("ij,ij->i", normals, gradients) < 0.0
    result[flip] = result[flip][:, [0, 2, 1]]
    return result


def _unwrap_triangle(vertices: np.ndarray, period: float) -> np.ndarray:
    result = np.asarray(vertices, dtype=np.float64).copy()
    for index in (1, 2):
        difference = result[index, 2] - result[0, 2]
        result[index, 2] -= period * np.round(difference / period)
    return result


def _unwrap_point_relative(
    point: np.ndarray, reference: np.ndarray, period: float
) -> np.ndarray:
    result = np.asarray(point, dtype=np.float64).copy()
    difference = result[2] - reference[2]
    result[2] -= period * np.round(difference / period)
    return result


def _canonicalize_point(
    point: np.ndarray, period: float, tolerance: float
) -> np.ndarray:
    result = np.asarray(point, dtype=np.float64).copy()
    result[2] %= period
    if result[2] <= tolerance or period - result[2] <= tolerance:
        result[2] = 0.0
    return result


def _canonicalize_points(
    points: np.ndarray, period: float, tolerance: float
) -> np.ndarray:
    """Vectorized :func:`_canonicalize_point` over an ``(n, 3)`` array."""
    result = np.asarray(points, dtype=np.float64).copy()
    result[:, 2] %= period
    snap = (result[:, 2] <= tolerance) | (period - result[:, 2] <= tolerance)
    result[snap, 2] = 0.0
    return result


def _periodic_distance(first: np.ndarray, second: np.ndarray, period: float) -> float:
    difference = np.asarray(first) - np.asarray(second)
    difference[2] -= period * np.round(difference[2] / period)
    return float(np.linalg.norm(difference))


def _coordinate_boundary_tags(
    points: np.ndarray, period: float, tolerance: float
) -> np.ndarray:
    tags = np.zeros(len(points), dtype=np.int64)
    radius = np.linalg.norm(points[:, :2], axis=1)
    tags[np.abs(radius - 1.0) <= tolerance] |= SurfaceMesh.EDGE
    tags[radius <= tolerance] |= SurfaceMesh.AXIS
    tags[
        (points[:, 2] <= tolerance) | (period - points[:, 2] <= tolerance)
    ] |= SurfaceMesh.PERIODIC_SEAM
    return tags


def _pyvista_boundary_tags(
    points, triangles, outer_indicator, seam_sides, period, tolerance
):
    """Recover VTK contour boundary provenance before projection moves points."""
    if outer_indicator.shape != (len(points),):
        raise SurfaceExtractionError(
            SurfaceStatus.DEGENERATE,
            "PyVista contour returned an invalid OUTER boundary indicator",
        )
    tags = _coordinate_boundary_tags(points, period, tolerance)
    tags &= ~SurfaceMesh.EDGE
    if not len(triangles):
        return tags
    edges = np.sort(
        np.vstack(
            (
                triangles[:, [0, 1]],
                triangles[:, [1, 2]],
                triangles[:, [2, 0]],
            )
        ),
        axis=1,
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    same_seam = (seam_sides[boundary_edges[:, 0]] != 0) & (
        seam_sides[boundary_edges[:, 0]] == seam_sides[boundary_edges[:, 1]]
    )
    physical_edge = np.all(
        np.isclose(
            outer_indicator[boundary_edges],
            1.0,
            rtol=0.0,
            atol=32.0 * np.finfo(float).eps,
        ),
        axis=1,
    )
    if np.any(~(same_seam | physical_edge)):
        raise SurfaceExtractionError(
            SurfaceStatus.DEGENERATE,
            "PyVista contour has a boundary without seam or EDGE provenance",
        )
    if np.any(physical_edge):
        tags[np.unique(boundary_edges[physical_edge])] |= SurfaceMesh.EDGE
    return tags


def _match_pyvista_periodic_seam(points, seam_sides, tolerance):
    """Identify VTK's lower/upper contour copies before coordinate merging.

    VTK may interpolate matching periodic boundary triangles a few ulps apart.
    The boundary side retained from the unprojected contour supplies the missing
    provenance: copies must form a one-to-one lower/upper assignment, and the
    assignment must agree to the square-root-epsilon scale of VTK geometry.
    """
    lower = np.flatnonzero(seam_sides == -1)
    upper = np.flatnonzero(seam_sides == 1)
    if not len(lower) and not len(upper):
        return points
    if len(lower) != len(upper):
        raise SurfaceExtractionError(
            SurfaceStatus.PERIODIC_MISMATCH,
            "PyVista contour has unequal lower/upper periodic seam point counts",
        )
    distances = np.linalg.norm(
        points[lower, np.newaxis, :2] - points[np.newaxis, upper, :2], axis=2
    )
    lower_assignment, upper_assignment = linear_sum_assignment(distances)
    matched_distances = distances[lower_assignment, upper_assignment]
    coordinate_scale = max(1.0, float(np.max(np.abs(points[:, :2]))))
    vtk_tolerance = max(
        tolerance, 8.0 * np.sqrt(np.finfo(float).eps) * coordinate_scale
    )
    if np.any(matched_distances > vtk_tolerance):
        raise SurfaceExtractionError(
            SurfaceStatus.PERIODIC_MISMATCH,
            "PyVista contour seam copies do not form coincident periodic pairs",
        )
    matched = np.asarray(points, dtype=np.float64).copy()
    matched[upper[upper_assignment]] = matched[lower[lower_assignment]]
    return matched


def _merge_coordinate_copies(points, triangles, period, tolerance):
    canonical_points: list[np.ndarray] = []
    point_buckets: dict[tuple[int, int, int], list[int]] = {}
    remap = np.empty(len(points), dtype=np.int64)
    for index, point in enumerate(points):
        canonical = _canonicalize_point(point, period, tolerance)
        key = tuple(np.floor(canonical / tolerance).astype(np.int64))
        point_id = None
        for offset in product((-1, 0, 1), repeat=3):
            neighbor = tuple(key[axis] + offset[axis] for axis in range(3))
            for candidate in point_buckets.get(neighbor, ()):
                if np.linalg.norm(canonical - canonical_points[candidate]) <= tolerance:
                    point_id = candidate
                    break
            if point_id is not None:
                break
        if point_id is None:
            point_id = len(canonical_points)
            canonical_points.append(canonical)
            point_buckets.setdefault(key, []).append(point_id)
        remap[index] = point_id
    remapped = remap[triangles]
    for triangle in remapped:
        if len(np.unique(triangle)) < 3:
            raise SurfaceExtractionError(
                SurfaceStatus.DEGENERATE,
                "coordinate merging collapsed a PyVista contour triangle",
            )
    return (
        np.asarray(canonical_points, dtype=np.float64).reshape(-1, 3),
        remapped,
        remap,
    )
