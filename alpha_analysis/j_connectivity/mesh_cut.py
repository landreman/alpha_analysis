"""Constrained transition cuts on incoming pitch surfaces.

This NumPy-only module implements DESIGN.md §§10.3--10.5. Logical points are
dimensionless ``(x=sqrt(s) cos(theta), y=sqrt(s) sin(theta), zeta)`` with
``zeta`` in radians. Actions have the length units of ``G`` and ``I``.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace
from heapq import heappop, heappush
from pathlib import Path

import numpy as np

from .field import BoozerFieldLike
from .surface_extract import (
    SurfaceExtractionConfig,
    SurfaceMesh,
    _evaluate_B,
    _physical_g,
    _project_to_level_near,
    surface_flux,
)
from .transitions import TransitionCurve
from .types import FloatArray, IntArray, TransitionStatus
from .well_trace import WellTraceConfig, trace_regular_well


class ConstrainedCutError(RuntimeError):
    """A curve could not be inserted without guessing surface topology."""


class _TransitionCutConflict(ConstrainedCutError):
    """One transition's cut cannot be completed without corrupting the mesh.

    Raised where completing the current transition would destroy an earlier
    constrained chain, exceed the boundary-snap allowance at the point of
    use, or assign parent/child sides without a decisive data margin.  The
    caller demotes that single transition to an explicit unresolved hyperedge
    instead of aborting every resolved transition on the slice.
    """


@dataclass(frozen=True)
class ConstrainedCutConfig:
    """Geometric controls for DESIGN.md §10.3 polyline insertion.

    Logical-distance controls are dimensionless and ``B_tolerance`` has field
    units. A failed local projection raises instead of selecting another
    nearby level-set sheet (DESIGN.md §21.2).

    ``side_assignment_margin_ratio`` makes the parent/child side assignment
    refuse to guess: the two candidate assignment costs must differ by at
    least this fraction of the mean parent/child-1 action jump, or the
    transition is reported unresolved rather than assigned by a coin-flip
    comparison a single bad trace could decide.
    """

    snap_tolerance: float = 1.0e-10
    max_surface_distance_ratio: float = 0.5
    B_tolerance: float = 1.0e-9
    path_anchor_count: int = 8
    min_transition_strip_edge_ratio: float = 0.1
    side_assignment_margin_ratio: float = 0.1
    max_corridor_faces: int = 64

    def __post_init__(self) -> None:
        for name in (
            "snap_tolerance",
            "max_surface_distance_ratio",
            "B_tolerance",
            "min_transition_strip_edge_ratio",
            "side_assignment_margin_ratio",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.path_anchor_count < 1:
            raise ValueError("path_anchor_count must be positive")
        if self.max_corridor_faces < 1:
            raise ValueError("max_corridor_faces must be positive")


@dataclass(frozen=True)
class CutTransitionPort:
    """One transition port after its constrained curve entered the mesh.

    Vertex IDs use the common transition-``u`` ordering. ``sheet_id=-1`` and
    vertex IDs ``-1`` occur only for an explicitly unresolved transition.
    """

    transition_id: int
    role: str
    sheet_id: int
    polyline_vertex_ids: IntArray
    action_values: FloatArray

    def __post_init__(self) -> None:
        vertex_ids = np.asarray(self.polyline_vertex_ids, dtype=np.int64)
        action = np.asarray(self.action_values, dtype=np.float64)
        if vertex_ids.ndim != 1 or action.shape != vertex_ids.shape:
            raise ValueError("cut-port IDs and actions must share one sample axis")
        object.__setattr__(self, "polyline_vertex_ids", vertex_ids)
        object.__setattr__(self, "action_values", action)


@dataclass(frozen=True)
class CutEvent:
    """An unresolved nongeneric event with arbitrary incident port endpoints.

    Indices identify the cut surface's ports and their sample endpoints. A
    failed incident arc keeps its port with sheet/vertex ID -1; it is never
    silently interpreted as absent connectivity.
    """

    event_id: int
    marginal_points: FloatArray
    port_indices: IntArray
    sample_indices: IntArray
    unresolved: bool = True

    def __post_init__(self):
        points = np.asarray(self.marginal_points, dtype=np.float64).reshape(-1, 3)
        ports = np.asarray(self.port_indices, dtype=np.int64)
        samples = np.asarray(self.sample_indices, dtype=np.int64)
        if ports.ndim != 1 or samples.shape != ports.shape:
            raise ValueError("event port and sample indices must share one axis")
        if np.any(ports < 0) or np.any(samples < 0):
            raise ValueError("event incidence indices must be nonnegative")
        object.__setattr__(self, "marginal_points", points)
        object.__setattr__(self, "port_indices", ports)
        object.__setattr__(self, "sample_indices", samples)


@dataclass(frozen=True)
class CutSurface:
    """A transition-cut mesh with cell-located sheet IDs and port incidence."""

    surface: SurfaceMesh
    action_values: FloatArray
    sheet_ids: IntArray
    cut_edges: IntArray
    ports: tuple[CutTransitionPort, ...]
    unresolved_transition_ids: IntArray
    unresolved_transition_reasons: tuple[str, ...] = ()
    events: tuple[CutEvent, ...] = ()
    corridor_count: int = 0
    max_corridor_faces_used: int = 0
    unresolved_event_action_vertex_ids: IntArray = dataclass_field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )

    def __post_init__(self) -> None:
        action = np.asarray(self.action_values, dtype=np.float64)
        sheets = np.asarray(self.sheet_ids, dtype=np.int64)
        edges = np.asarray(self.cut_edges, dtype=np.int64).reshape(-1, 2)
        unresolved = np.asarray(self.unresolved_transition_ids, dtype=np.int64)
        reasons = tuple(self.unresolved_transition_reasons)
        event_unknown = np.asarray(
            self.unresolved_event_action_vertex_ids, dtype=np.int64
        )
        if event_unknown.ndim != 1 or np.any(
            (event_unknown < 0) | (event_unknown >= len(action))
        ):
            raise ValueError("unresolved event-action vertex lies outside the surface")
        if np.any(np.isfinite(action[event_unknown])):
            raise ValueError("an unresolved event action must be NaN")
        if action.shape != (len(self.surface.points),):
            raise ValueError("cut action must have one value per surface point")
        if sheets.shape != (len(self.surface.triangles),):
            raise ValueError("sheet_ids must have one value per triangle")
        if edges.size and (edges.min() < 0 or edges.max() >= len(self.surface.points)):
            raise ValueError("cut edge index lies outside the surface")
        if not reasons and len(unresolved):
            reasons = ("unresolved",) * len(unresolved)
        if len(reasons) != len(unresolved) or any(not reason for reason in reasons):
            raise ValueError("each unresolved transition needs one nonempty reason")
        for port in self.ports:
            ids = port.polyline_vertex_ids
            if np.any((ids < 0) != (port.sheet_id < 0)):
                raise ValueError("only unresolved ports may use negative vertex IDs")
            if ids.size and ids.min() >= 0 and ids.max() >= len(self.surface.points):
                raise ValueError("cut port index lies outside the surface")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("event IDs must be unique")
        for event in self.events:
            for port_index, sample_index in zip(
                event.port_indices, event.sample_indices
            ):
                if port_index >= len(self.ports) or sample_index >= len(
                    self.ports[port_index].action_values
                ):
                    raise ValueError("event incidence lies outside its cut port")
        object.__setattr__(self, "action_values", action)
        object.__setattr__(self, "sheet_ids", sheets)
        object.__setattr__(self, "cut_edges", edges)
        object.__setattr__(self, "unresolved_transition_ids", unresolved)
        object.__setattr__(self, "unresolved_transition_reasons", reasons)
        object.__setattr__(self, "unresolved_event_action_vertex_ids", event_unknown)

    def to_pyvista(self):
        """Return a PyVista surface with named point and cell arrays (§17.1)."""
        view = self.surface.to_pyvista()
        view.point_data["A action length [length]"] = self.action_values
        view.cell_data["sheet ID [integer]"] = self.sheet_ids
        return view

    def unresolved_action_flux(self, field: BoozerFieldLike) -> float:
        """Measure all cells containing unknown action by |ds wedge d alpha|.

        This is the dimensionless surface measure of §4.4, not a bound on
        bounce-time-weighted volume or on Theta. No NaN cell is discarded;
        later quadrature must retain this unresolved contribution (§21.2).
        Event/transition connectivity uncertainty remains separately explicit.
        """
        unknown = np.any(
            ~np.isfinite(self.action_values[self.surface.triangles]), axis=1
        )
        surface = replace(
            self.surface,
            triangles=self.surface.triangles[unknown],
            triangle_parent_tetrahedra=self.surface.triangle_parent_tetrahedra[unknown],
            component_ids=self.surface.component_ids[unknown],
        )
        return surface_flux(surface, field)

    def to_networkx(self):
        """Return the coarse sheet/transition diagnostic graph from §10.5."""
        from . import optional_import

        networkx = optional_import("networkx", extra="connectivity")
        graph = networkx.MultiGraph()
        for sheet_id in np.unique(self.sheet_ids):
            cells = np.flatnonzero(self.sheet_ids == sheet_id)
            vertices = np.unique(self.surface.triangles[cells])
            finite = self.action_values[vertices]
            finite = finite[np.isfinite(finite)]
            logical_area = 0.0
            for triangle in self.surface.triangles[cells]:
                origin = self.surface.points[triangle[0]]
                first = _periodic_delta(
                    origin, self.surface.points[triangle[1]], self.surface.period
                )
                second = _periodic_delta(
                    origin, self.surface.points[triangle[2]], self.surface.period
                )
                logical_area += 0.5 * float(np.linalg.norm(np.cross(first, second)))
            graph.add_node(
                ("sheet", int(sheet_id)),
                kind="sheet",
                touches_edge=bool(
                    np.any(
                        (self.surface.boundary_tags[vertices] & SurfaceMesh.EDGE) != 0
                    )
                ),
                action_min=float(np.min(finite)) if len(finite) else np.nan,
                action_max=float(np.max(finite)) if len(finite) else np.nan,
                logical_area=logical_area,
                triangle_count=int(len(cells)),
                unresolved=bool(np.any(~np.isfinite(self.action_values[vertices]))),
            )
        unresolved = set(map(int, self.unresolved_transition_ids))
        for transition_id in sorted({port.transition_id for port in self.ports}):
            node = ("transition", int(transition_id))
            graph.add_node(
                node, kind="transition", unresolved=transition_id in unresolved
            )
            for port_index, port in enumerate(self.ports):
                if port.transition_id == transition_id and port.sheet_id >= 0:
                    graph.add_edge(
                        node,
                        ("sheet", int(port.sheet_id)),
                        port_index=int(port_index),
                        role=port.role,
                    )
        for event in self.events:
            node = ("event", int(event.event_id))
            graph.add_node(node, kind="event", unresolved=event.unresolved)
            for port_index, sample_index in zip(
                event.port_indices, event.sample_indices
            ):
                port = self.ports[int(port_index)]
                if port.sheet_id >= 0:
                    graph.add_edge(
                        node,
                        ("sheet", int(port.sheet_id)),
                        port_index=int(port_index),
                        sample_index=int(sample_index),
                        role=port.role,
                    )
        return graph


class _UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int64)

    def find(self, item: int) -> int:
        root = int(item)
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            parent = int(self.parent[item])
            self.parent[item] = root
            item = parent
        return root

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[max(first_root, second_root)] = min(first_root, second_root)

    def labels(self) -> np.ndarray:
        roots = np.array([self.find(index) for index in range(len(self.parent))])
        _, labels = np.unique(roots, return_inverse=True)
        return labels.astype(np.int64)


def _periodic_delta(first: FloatArray, second: FloatArray, period: float) -> FloatArray:
    delta = np.asarray(second, dtype=float) - np.asarray(first, dtype=float)
    delta[2] -= period * np.round(delta[2] / period)
    return delta


def _closest_point_triangle(point, first, second, third):
    """Return the closest point and barycentric weights on one 3-D triangle."""
    ab = second - first
    ac = third - first
    ap = point - first
    d1, d2 = float(np.dot(ab, ap)), float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return first, np.array([1.0, 0.0, 0.0])
    bp = point - second
    d3, d4 = float(np.dot(ab, bp)), float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return second, np.array([0.0, 1.0, 0.0])
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return first + v * ab, np.array([1.0 - v, v, 0.0])
    cp = point - third
    d5, d6 = float(np.dot(ab, cp)), float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return third, np.array([0.0, 0.0, 1.0])
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return first + w * ac, np.array([1.0 - w, 0.0, w])
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return second + w * (third - second), np.array([0.0, 1.0 - w, w])
    denominator = 1.0 / (va + vb + vc)
    v, w = vb * denominator, vc * denominator
    return first + v * ab + w * ac, np.array([1.0 - v - w, v, w])


class _MutableMesh:
    def __init__(
        self, surface, action_values, field, config, trace_config=None
    ) -> None:
        self.level = float(surface.level)
        self.period = float(surface.period)
        self.points = [row.copy() for row in surface.points]
        self.triangles = [list(map(int, row)) for row in surface.triangles]
        self.B = list(map(float, surface.B))
        self.g = list(map(float, surface.g))
        self.tags = list(map(int, surface.boundary_tags))
        self.parent_edges = [row.copy() for row in surface.point_parent_edges]
        self.parent_tetrahedra = list(map(int, surface.triangle_parent_tetrahedra))
        self.component_ids = list(map(int, surface.component_ids))
        self.action = list(map(float, action_values))
        # Insertion happens before branch-specific limiting actions are
        # assigned. Keep the interpolation DAG so off-cut descendants can
        # be refreshed from the correct sheet copies afterwards (§10.3).
        self.action_stencils: dict[int, tuple[tuple[int, ...], np.ndarray]] = {}
        self.field = field
        self.config = config
        self.trace_config = trace_config
        # Edges already claimed by an inserted constrained chain. A later
        # constraint must not flip one away: the recorded cut path would then
        # reference a destroyed edge and the cut would silently dangle.
        self.constrained_edges: set[tuple[int, int]] = set()
        self.corridor_count = 0
        self.max_corridor_faces_used = 0
        # Component-provenance enforcement (DESIGN.md §23 milestone 10.3):
        # while set, every location, vertex snap, and insertion is restricted
        # to triangles of this surface component, so an off-surface projection
        # can never jump to a disconnected sheet that happens to be nearer in
        # Euclidean coordinates (§21.2).
        self.active_component: int | None = None

    def _unwrapped_triangle(self, triangle_id, point):
        ids = self.triangles[triangle_id]
        vertices = np.asarray([self.points[index] for index in ids], dtype=float)
        vertices[:, 2] += self.period * np.round(
            (float(point[2]) - vertices[:, 2]) / self.period
        )
        return ids, vertices

    def _nearest_location(self, point, component=None):
        if component is None:
            component = self.active_component
        if not self.triangles:
            raise ConstrainedCutError("cannot insert a curve into an empty surface")
        triangle_ids = np.arange(len(self.triangles), dtype=np.int64)
        if component is not None:
            triangle_ids = triangle_ids[
                np.asarray(self.component_ids, dtype=np.int64) == int(component)
            ]
            if not len(triangle_ids):
                raise ConstrainedCutError(
                    f"surface has no triangles on component {component}"
                )
        vertices = np.asarray(self.points, dtype=float)[
            np.asarray(self.triangles, dtype=np.int64)[triangle_ids]
        ]
        vertices[:, :, 2] += self.period * np.round(
            (float(point[2]) - vertices[:, :, 2]) / self.period
        )
        # A vertex gives an upper bound on nearest-triangle distance; a
        # triangle's axis-aligned box gives a lower bound. Only cull boxes
        # beyond that upper bound, with padding to retain roundoff ties.
        # The exact closest-point test and original triangle ordering below
        # are unchanged, including deterministic selection at shared edges.
        upper_bound = float(np.min(np.linalg.norm(vertices - point, axis=2)))
        box_delta = np.maximum(
            np.maximum(vertices.min(axis=1) - point, point - vertices.max(axis=1)),
            0.0,
        )
        candidates = triangle_ids[
            np.flatnonzero(
                np.linalg.norm(box_delta, axis=1)
                <= upper_bound + self.config.snap_tolerance
            )
        ]
        best = None
        for triangle_id in candidates:
            ids, vertices = self._unwrapped_triangle(triangle_id, point)
            closest, barycentric = _closest_point_triangle(point, *vertices)
            distance = float(np.linalg.norm(closest - point))
            if best is None or distance < best[0]:
                edge_scale = max(
                    np.linalg.norm(vertices[1] - vertices[0]),
                    np.linalg.norm(vertices[2] - vertices[1]),
                    np.linalg.norm(vertices[0] - vertices[2]),
                )
                best = (distance, triangle_id, ids, closest, barycentric, edge_scale)
        if best is None:
            raise ConstrainedCutError("cannot insert a curve into an empty surface")
        return best

    def _field_line_chart(self, points, origin_zeta):
        points = np.asarray(points, dtype=float)
        zeta = points[..., 2] + self.period * np.round(
            (origin_zeta - points[..., 2]) / self.period
        )
        s = np.sum(points[..., :2] ** 2, axis=-1)
        alpha = (
            np.arctan2(points[..., 1], points[..., 0])
            - np.asarray(self.field.iota(s)) * zeta
        )
        return np.stack(
            (np.sqrt(s) * np.cos(alpha), np.sqrt(s) * np.sin(alpha)), axis=-1
        )

    def _new_point(self, point, barycentric, ids, tag, edge_scale):
        point = np.asarray(point, dtype=float).copy()
        point[2] %= self.period
        chart_insertion = (
            getattr(self, "normal_insertion", False)
            and self.field is not None
            and not (tag & SurfaceMesh.G_ZERO)
        )
        if chart_insertion:
            chart = np.dot(
                barycentric,
                self._field_line_chart(np.asarray(self.points)[list(ids)], point[2]),
            )
            s = float(np.dot(chart, chart))
            alpha = float(np.arctan2(chart[1], chart[0]))
            iota = float(self.field.iota(s))
            theta = alpha + iota * point[2]
            point[:2] = np.sqrt(s) * np.array([np.cos(theta), np.sin(theta)])
        if self.field is not None:
            residual = abs(
                float(_evaluate_B(self.field, point[np.newaxis, :])[0]) - self.level
            )
            if residual > self.config.B_tolerance:
                projected = None
                if chart_insertion:
                    zeta = float(point[2])
                    for _ in range(32):
                        theta = alpha + iota * zeta
                        value = float(self.field.B(s, theta, zeta)) - self.level
                        candidate = np.array(
                            [
                                np.sqrt(s) * np.cos(theta),
                                np.sqrt(s) * np.sin(theta),
                                zeta % self.period,
                            ]
                        )
                        if (
                            np.linalg.norm(
                                _periodic_delta(point, candidate, self.period)
                            )
                            > self.config.max_surface_distance_ratio * edge_scale
                        ):
                            break
                        if abs(value) <= self.config.B_tolerance:
                            projected = candidate
                            break
                        derivative = float(self.field.D_B(s, theta, zeta))
                        if not np.isfinite(derivative) or derivative == 0:
                            break
                        zeta -= value / derivative
                else:
                    projected = _project_to_level_near(
                        point,
                        self.field,
                        self.level,
                        SurfaceExtractionConfig(B_tolerance=self.config.B_tolerance),
                        self.config.max_surface_distance_ratio * edge_scale,
                    )
                if projected is None:
                    raise ConstrainedCutError(
                        "a constrained vertex could not be projected locally to B=b"
                    )
                point = projected
            B = float(_evaluate_B(self.field, point[np.newaxis, :])[0])
            g = float(_physical_g(self.field, point[np.newaxis, :])[0])
        else:
            B = self.level
            g = float(np.dot(barycentric, np.asarray([self.g[index] for index in ids])))
        self.points.append(point)
        self.B.append(B)
        self.g.append(g)
        self.tags.append(int(tag))
        self.parent_edges.append(np.array([-1, -1], dtype=np.int64))
        self.action.append(
            float(
                np.dot(barycentric, np.asarray([self.action[index] for index in ids]))
            )
        )
        new_id = len(self.points) - 1
        self.action_stencils[new_id] = (
            tuple(map(int, ids)),
            np.asarray(barycentric, dtype=float).copy(),
        )
        return new_id

    def _split_triangle(self, triangle_id, point, barycentric, tag, edge_scale):
        ids = self.triangles[triangle_id]
        new_id = self._new_point(point, barycentric, ids, tag, edge_scale)
        first, second, third = ids
        parent = self.parent_tetrahedra[triangle_id]
        component = self.component_ids[triangle_id]
        self.triangles[triangle_id] = [first, second, new_id]
        self.triangles.extend(([second, third, new_id], [third, first, new_id]))
        self.parent_tetrahedra.extend((parent, parent))
        self.component_ids.extend((component, component))
        return new_id

    def _split_edge(self, edge, point, fraction, tag, edge_scale):
        first, second = map(int, edge)
        if tuple(sorted((first, second))) in self.constrained_edges:
            # Splitting a claimed edge would leave an earlier transition's
            # recorded cut referencing a destroyed edge — a silent dangling
            # cut (§21.2). The caller demotes this transition instead.
            raise _TransitionCutConflict(
                f"inserting this curve would split constrained cut edge "
                f"({first}, {second}) claimed by an earlier transition"
            )
        barycentric = np.array([1.0 - fraction, fraction])
        new_id = self._new_point(point, barycentric, [first, second], tag, edge_scale)
        owners = [
            index
            for index, triangle in enumerate(self.triangles)
            if first in triangle and second in triangle
        ]
        if not owners:
            raise ConstrainedCutError("attempted to split an edge absent from the mesh")
        additions = []
        parent_additions = []
        component_additions = []
        for triangle_id in owners:
            triangle = self.triangles[triangle_id]
            for index in range(3):
                a = triangle[index]
                b = triangle[(index + 1) % 3]
                c = triangle[(index + 2) % 3]
                if {a, b} == {first, second}:
                    self.triangles[triangle_id] = [a, new_id, c]
                    additions.append([new_id, b, c])
                    parent_additions.append(self.parent_tetrahedra[triangle_id])
                    component_additions.append(self.component_ids[triangle_id])
                    break
        self.triangles.extend(additions)
        self.parent_tetrahedra.extend(parent_additions)
        self.component_ids.extend(component_additions)
        return new_id

    def _component_vertices(self, component):
        return {
            int(vertex)
            for triangle, owner in zip(self.triangles, self.component_ids)
            if owner == component
            for vertex in triangle
        }

    def insert_point(self, point, *, tag=0, preserve=True):
        point = np.asarray(point, dtype=float)
        snap_candidates = (
            range(len(self.points))
            if self.active_component is None
            else sorted(self._component_vertices(self.active_component))
        )
        distances = np.array(
            [
                np.linalg.norm(_periodic_delta(point, self.points[index], self.period))
                for index in snap_candidates
            ]
        )
        nearest = int(snap_candidates[int(np.argmin(distances))])
        if distances.min() <= self.config.snap_tolerance:
            self.tags[nearest] |= int(tag)
            return nearest
        distance, triangle_id, ids, closest, barycentric, edge_scale = (
            self._nearest_location(point)
        )
        if getattr(self, "normal_insertion", False) and self.field is not None:
            # Incoming wells are locally graphs over (s,alpha). A true B=b
            # point need not lie on the current PL face: Euclidean nearest
            # insertion can fold a face near an event. Use the local lifted
            # field-line chart to select its face, keeping component provenance
            # and the original logical-distance acceptance allowance.
            component = self.component_ids[triangle_id]
            candidate_ids = np.flatnonzero(np.asarray(self.component_ids) == component)
            triangles = np.asarray(self.triangles, dtype=np.int64)[candidate_ids]
            coordinates = np.asarray(self.points)[triangles].copy()
            coordinates[:, :, 2] += self.period * np.round(
                (point[2] - coordinates[:, :, 2]) / self.period
            )
            flux = np.sum(coordinates[:, :, :2] ** 2, axis=2)
            theta = np.arctan2(coordinates[:, :, 1], coordinates[:, :, 0])
            alpha = theta - np.asarray(self.field.iota(flux)) * coordinates[:, :, 2]
            chart = np.stack(
                (np.sqrt(flux) * np.cos(alpha), np.sqrt(flux) * np.sin(alpha)), axis=2
            )
            point_flux = float(np.sum(point[:2] ** 2))
            point_alpha = float(
                np.arctan2(point[1], point[0])
                - float(self.field.iota(point_flux)) * point[2]
            )
            query = np.sqrt(point_flux) * np.array(
                [np.cos(point_alpha), np.sin(point_alpha)]
            )
            first, second = chart[:, 1] - chart[:, 0], chart[:, 2] - chart[:, 0]
            delta = query - chart[:, 0]
            determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
            valid = np.abs(determinant) > np.finfo(float).tiny
            weights = np.full((len(triangles), 3), np.nan)
            weights[valid, 1] = (
                delta[valid, 0] * second[valid, 1] - delta[valid, 1] * second[valid, 0]
            ) / determinant[valid]
            weights[valid, 2] = (
                first[valid, 0] * delta[valid, 1] - first[valid, 1] * delta[valid, 0]
            ) / determinant[valid]
            weights[:, 0] = 1 - weights[:, 1] - weights[:, 2]
            valid &= np.all(weights >= -self.config.snap_tolerance, axis=1)
            if not np.any(valid):
                raise ConstrainedCutError(
                    "no same-component face contains the event point in its field-line chart"
                )
            locations = np.sum(weights[:, :, None] * coordinates, axis=1)
            distances = np.linalg.norm(locations - point, axis=1)
            distances[~valid] = np.inf
            nearest = int(np.argmin(distances))
            distance = float(distances[nearest])
            triangle_id = int(candidate_ids[nearest])
            ids = triangles[nearest]
            closest = locations[nearest]
            barycentric = weights[nearest]
            local = coordinates[nearest]
            edge_scale = max(
                np.linalg.norm(local[1] - local[0]),
                np.linalg.norm(local[2] - local[0]),
                np.linalg.norm(local[2] - local[1]),
            )
        allowed = self.config.max_surface_distance_ratio * max(
            edge_scale, self.config.snap_tolerance
        )
        if distance > allowed:
            raise ConstrainedCutError(
                f"curve point is {distance:.3e} from the nearest surface triangle"
            )
        active = np.flatnonzero(barycentric > self.config.snap_tolerance)
        insert_at = point if preserve else closest
        if len(active) == 1:
            vertex_id = int(ids[int(active[0])])
            self.tags[vertex_id] |= int(tag)
            return vertex_id
        if len(active) == 2:
            local_first, local_second = map(int, active)
            fraction = float(
                barycentric[local_second]
                / (barycentric[local_first] + barycentric[local_second])
            )
            edge = (ids[local_first], ids[local_second])
            if not preserve and tuple(sorted(map(int, edge))) in self.constrained_edges:
                # A helper anchor is guidance, not authoritative geometry:
                # snapping it to the claimed edge's nearest endpoint keeps the
                # earlier chain whole where a split would destroy it.
                return int(edge[0] if fraction <= 0.5 else edge[1])
            inherited = self.tags[edge[0]] & self.tags[edge[1]]
            return self._split_edge(
                edge, insert_at, fraction, inherited | int(tag), edge_scale
            )
        return self._split_triangle(
            triangle_id, insert_at, barycentric, int(tag), edge_scale
        )

    def insert_tagged_point(self, point, tag, *, component=None):
        """Insert an authoritative curve point on the nearest tagged edge.

        Standalone critical-curve refinement follows the true ``B=b, g=0``
        curve, so its point can sit slightly off the chordal surface boundary.
        Restricting the location search to explicit provenance prevents that
        harmless offset from turning a boundary curve into an interior cut.
        """
        point = np.asarray(point, dtype=float)
        if component is None:
            component = self.active_component
        allowed_vertices = (
            None if component is None else self._component_vertices(component)
        )
        tagged_vertices = [
            index
            for index, value in enumerate(self.tags)
            if (value & tag) != 0
            and (allowed_vertices is None or index in allowed_vertices)
        ]
        if tagged_vertices:
            distances = np.array(
                [
                    np.linalg.norm(
                        _periodic_delta(point, self.points[index], self.period)
                    )
                    for index in tagged_vertices
                ]
            )
            nearest = int(tagged_vertices[int(np.argmin(distances))])
            if np.min(distances) <= self.config.snap_tolerance:
                return nearest
        best = None
        for edge in self.edges():
            if allowed_vertices is not None and not all(
                vertex in allowed_vertices for vertex in edge
            ):
                continue
            if not all((self.tags[vertex] & tag) != 0 for vertex in edge):
                continue
            first = np.asarray(self.points[edge[0]], dtype=float)
            delta = _periodic_delta(first, self.points[edge[1]], self.period)
            query = first + _periodic_delta(first, point, self.period)
            norm_squared = float(np.dot(delta, delta))
            fraction = (
                0.0
                if norm_squared == 0.0
                else float(np.clip(np.dot(query - first, delta) / norm_squared, 0, 1))
            )
            closest = first + fraction * delta
            distance = float(np.linalg.norm(query - closest))
            if best is None or distance < best[0]:
                best = (distance, edge, fraction, np.sqrt(norm_squared))
        if best is None:
            raise ConstrainedCutError(
                "surface has no edge with required curve provenance"
            )
        distance, edge, fraction, edge_scale = best
        if distance > self.config.max_surface_distance_ratio * max(
            edge_scale, self.config.snap_tolerance
        ):
            raise ConstrainedCutError(
                f"critical point is {distance:.3e} from the nearest tagged edge"
            )
        if fraction <= self.config.snap_tolerance:
            return int(edge[0])
        if 1.0 - fraction <= self.config.snap_tolerance:
            return int(edge[1])
        return self._split_edge(edge, point, fraction, tag, edge_scale)

    def edges(self):
        result = set()
        for triangle in self.triangles:
            for index in range(3):
                result.add(tuple(sorted((triangle[index], triangle[(index + 1) % 3]))))
        return result

    def connect(self, first, second, first_point, second_point, depth=0):
        """Return the permitted mesh-aligned prototype path from §10.3."""
        return self.constrained_path(first, second, first_point, second_point)

    def constrained_path(self, first, second, first_point, second_point):
        """Return a finite mesh-edge approximation between mapped samples.

        DESIGN.md §10.3 explicitly permits this early mesh-aligned prototype
        when the direct backward map validates the branch location. Endpoints
        are the authoritative mapped ``T`` samples inserted above; the edge
        cost strongly penalizes deviation from their locally unwrapped chord.
        """
        start = np.asarray(first_point, dtype=float)
        segment = _periodic_delta(start, second_point, self.period)
        segment_norm_squared = float(np.dot(segment, segment))
        segment_length = np.sqrt(segment_norm_squared)
        adjacency = {}
        for edge in self.edges():
            edge_first = start + _periodic_delta(
                start, self.points[edge[0]], self.period
            )
            edge_second = edge_first + _periodic_delta(
                self.points[edge[0]], self.points[edge[1]], self.period
            )
            midpoint = 0.5 * (edge_first + edge_second)
            fraction = (
                0.0
                if segment_norm_squared == 0.0
                else float(
                    np.clip(
                        np.dot(midpoint - start, segment) / segment_norm_squared,
                        0.0,
                        1.0,
                    )
                )
            )
            distance = float(np.linalg.norm(midpoint - (start + fraction * segment)))
            length = float(np.linalg.norm(edge_second - edge_first))
            normalized = distance / max(segment_length, self.config.snap_tolerance)
            weight = length * (1.0 + 100.0 * normalized * normalized)
            if tuple(sorted(map(int, edge))) in self.constrained_edges:
                # Overlapping an existing cut chain would branch the cut
                # graph; route around it unless there is no other way.
                weight *= 1.0e6
            adjacency.setdefault(edge[0], []).append((edge[1], weight))
            adjacency.setdefault(edge[1], []).append((edge[0], weight))
        queue = [(0.0, int(first))]
        distance = {int(first): 0.0}
        previous = {}
        while queue:
            cost, vertex = heappop(queue)
            if vertex == second:
                break
            if cost != distance[vertex]:
                continue
            for neighbor, weight in adjacency.get(vertex, []):
                candidate = cost + weight
                if candidate < distance.get(neighbor, np.inf):
                    distance[neighbor] = candidate
                    previous[neighbor] = vertex
                    heappush(queue, (candidate, neighbor))
        if second not in distance:
            raise ConstrainedCutError(
                "mapped transition samples lie on separate meshes"
            )
        path = [int(second)]
        while path[-1] != first:
            path.append(previous[path[-1]])
        return path[::-1]

    def _local_projection(self, first, second):
        origin = np.asarray(self.points[first], dtype=float)
        chord = _periodic_delta(origin, self.points[second], self.period)
        length = float(np.linalg.norm(chord))
        if length <= self.config.snap_tolerance:
            raise ConstrainedCutError("a constrained segment has zero logical length")
        if getattr(self, "normal_insertion", False) and self.field is not None:
            chart = self._field_line_chart(np.asarray(self.points), origin[2])
            chart_origin = chart[first].copy()
            chord = chart[second] - chart_origin
            length = float(np.linalg.norm(chord))
            if length <= self.config.snap_tolerance:
                raise _TransitionCutConflict(
                    "a constrained arc collapses in its local field-line chart"
                )
            tangent = chord / length
            transverse = np.array([-tangent[1], tangent[0]])
            projected = {}

            def project(vertex):
                if vertex not in projected:
                    value = (
                        chart[vertex]
                        if vertex < len(chart)
                        else self._field_line_chart(self.points[vertex], origin[2])
                    )
                    delta = value - chart_origin
                    projected[vertex] = np.array(
                        [np.dot(delta, tangent), np.dot(delta, transverse)]
                    )
                return projected[vertex]

            return project, length
        tangent = chord / length
        normals = []
        for triangle in self.triangles:
            if first not in triangle and second not in triangle:
                continue
            vertices = np.asarray(
                [
                    origin + _periodic_delta(origin, self.points[v], self.period)
                    for v in triangle
                ]
            )
            normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            if np.linalg.norm(normal) > self.config.snap_tolerance:
                normals.append(normal / np.linalg.norm(normal))
        if not normals:
            raise ConstrainedCutError(
                "a constrained endpoint has no regular incident face"
            )
        reference = normals[0]
        normal = np.sum(
            [value if np.dot(value, reference) >= 0.0 else -value for value in normals],
            axis=0,
        )
        normal -= np.dot(normal, tangent) * tangent
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= self.config.snap_tolerance:
            raise ConstrainedCutError(
                "a constrained segment has no stable tangent plane"
            )
        normal /= normal_norm
        transverse = np.cross(normal, tangent)
        projected = {}

        def project(vertex):
            if vertex not in projected:
                delta = _periodic_delta(origin, self.points[vertex], self.period)
                projected[vertex] = np.array(
                    [np.dot(delta, tangent), np.dot(delta, transverse)]
                )
            return projected[vertex]

        return project, length

    def _flip_crossing_edge(self, edge, project):
        first, second = edge
        if tuple(sorted((int(first), int(second)))) in self.constrained_edges:
            return False
        owners = [
            index
            for index, triangle in enumerate(self.triangles)
            if first in triangle and second in triangle
        ]
        if len(owners) != 2:
            return False
        opposites = [
            next(vertex for vertex in self.triangles[index] if vertex not in edge)
            for index in owners
        ]
        third, fourth = opposites
        if third == fourth or tuple(sorted((third, fourth))) in self.edges():
            return False
        projected = {
            vertex: project(vertex) for vertex in (first, second, third, fourth)
        }

        def orient(a, b, c):
            first_delta = projected[b] - projected[a]
            second_delta = projected[c] - projected[a]
            return float(
                first_delta[0] * second_delta[1] - first_delta[1] * second_delta[0]
            )

        tolerance = self.config.snap_tolerance
        # The old and new diagonals must lie inside a convex projected quad.
        if orient(third, fourth, first) * orient(third, fourth, second) >= -tolerance:
            return False
        old_normals = []
        for owner in owners:
            triangle = self.triangles[owner]
            origin = np.asarray(self.points[triangle[0]], dtype=float)
            vertices = np.asarray(
                [
                    origin + _periodic_delta(origin, self.points[v], self.period)
                    for v in triangle
                ],
                dtype=float,
            )
            old_normals.append(
                np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            )
        reference_normal = old_normals[0] + old_normals[1]
        replacements = [[third, fourth, first], [fourth, third, second]]
        for replacement in replacements:
            origin = np.asarray(self.points[replacement[0]], dtype=float)
            vertices = np.asarray(
                [
                    origin + _periodic_delta(origin, self.points[v], self.period)
                    for v in replacement
                ],
                dtype=float,
            )
            normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            if np.linalg.norm(normal) <= tolerance:
                return False
            if np.dot(normal, reference_normal) < 0.0:
                replacement[0], replacement[1] = replacement[1], replacement[0]
        component = self.component_ids[owners[0]]
        if self.component_ids[owners[1]] != component:
            return False
        for owner, replacement in zip(owners, replacements):
            self.triangles[owner] = replacement
            self.parent_tetrahedra[owner] = -1
            self.component_ids[owner] = component
        return True

    def constrain_edge(self, first, second):
        """Insert one local straight segment by edge splits, then flips."""
        target = tuple(sorted((int(first), int(second))))
        if target in self.edges():
            return [int(first), int(second)]
        project, length = self._local_projection(first, second)
        tolerance = max(self.config.snap_tolerance, 1.0e-12 * length)
        endpoint_components = {
            self.component_ids[index]
            for index, triangle in enumerate(self.triangles)
            if first in triangle or second in triangle
        }
        if len(endpoint_components) != 1:
            raise ConstrainedCutError(
                "constrained endpoints do not share one surface component"
            )
        target_component = endpoint_components.pop()
        component_vertices = {
            vertex
            for index, triangle in enumerate(self.triangles)
            if self.component_ids[index] == target_component
            for vertex in triangle
        }
        origin = np.asarray(self.points[first], dtype=float)
        physical_chord = _periodic_delta(origin, self.points[second], self.period)
        physical_length = np.linalg.norm(physical_chord)
        tangent = physical_chord / length

        def local_crossing(edge, fraction, crossing_x):
            start = np.asarray(self.points[edge[0]], dtype=float)
            delta = _periodic_delta(start, self.points[edge[1]], self.period)
            crossing = start + fraction * delta
            gap = np.linalg.norm(
                _periodic_delta(origin, crossing, self.period) - crossing_x * tangent
            )
            return gap <= self.config.max_surface_distance_ratio * min(
                physical_length, np.linalg.norm(delta)
            )

        # A segment through an existing vertex is two constraints, not an
        # ambiguous four-triangle edge flip.
        on_segment = []
        for vertex in component_vertices:
            if vertex in (first, second):
                continue
            coordinate = project(vertex)
            if (
                tolerance < coordinate[0] < length - tolerance
                and abs(coordinate[1]) <= tolerance
                and np.linalg.norm(
                    _periodic_delta(origin, self.points[vertex], self.period)
                    - coordinate[0] * tangent
                )
                <= (
                    self.config.max_surface_distance_ratio * physical_length
                    if getattr(self, "normal_insertion", False)
                    else tolerance
                )
            ):
                on_segment.append((coordinate[0], vertex))
        if on_segment:
            middle = min(on_segment)[1]
            first_half = self.constrain_edge(first, middle)
            second_half = self.constrain_edge(middle, second)
            return first_half[:-1] + second_half
        # Split every crossed edge at the local tangent-plane intersection.
        # Consecutive crossings bound one triangle, so conforming edge splits
        # create the desired constrained chain without requiring a convex
        # four-vertex edge-flip neighborhood.
        crossings = []
        # This pass does not mutate triangles until all crossings are found.
        # Index component membership once, rather than scanning every face
        # again for every edge (quadratic work at full curve sampling).
        component_edges = {
            tuple(sorted((triangle[local], triangle[(local + 1) % 3])))
            for index, triangle in enumerate(self.triangles)
            if self.component_ids[index] == target_component
            for local in range(3)
        }
        for edge in self.edges():
            if first in edge or second in edge:
                continue
            if edge not in component_edges:
                continue
            first_point, second_point = project(edge[0]), project(edge[1])
            denominator = first_point[1] - second_point[1]
            if abs(denominator) <= tolerance:
                continue
            fraction = first_point[1] / denominator
            if not tolerance < fraction < 1.0 - tolerance:
                continue
            crossing_x = first_point[0] + fraction * (second_point[0] - first_point[0])
            if tolerance < crossing_x < length - tolerance and local_crossing(
                edge, fraction, crossing_x
            ):
                crossings.append((crossing_x, edge, fraction))
        chain = [int(first)]
        for _, edge, fraction in sorted(crossings):
            if tuple(sorted(map(int, edge))) in self.constrained_edges:
                # Splitting an already inserted cut edge would destroy it and
                # leave the recorded cut dangling. Leave it whole; the chain
                # check below then fails and the Dijkstra fallback routes
                # around the existing chain instead.
                continue
            edge_first = np.asarray(self.points[edge[0]], dtype=float)
            point = edge_first + fraction * _periodic_delta(
                edge_first, self.points[edge[1]], self.period
            )
            point[2] %= self.period
            inherited = self.tags[edge[0]] & self.tags[edge[1]]
            edge_scale = float(
                np.linalg.norm(
                    _periodic_delta(
                        self.points[edge[0]], self.points[edge[1]], self.period
                    )
                )
            )
            chain.append(self._split_edge(edge, point, fraction, inherited, edge_scale))
        chain.append(int(second))
        current_edges = self.edges()
        if all(
            tuple(sorted((a, b))) in current_edges
            for a, b in zip(chain[:-1], chain[1:])
        ):
            return chain
        maximum_flips = max(1, 2 * len(self.edges()))
        flipped_edges = set()
        for _ in range(maximum_flips):
            if target in self.edges():
                return [int(first), int(second)]
            crossing = []
            allowed_edges = {
                tuple(sorted((triangle[local], triangle[(local + 1) % 3])))
                for index, triangle in enumerate(self.triangles)
                if self.component_ids[index] == target_component
                for local in range(3)
            }
            for edge in self.edges():
                if first in edge or second in edge:
                    continue
                if edge not in allowed_edges:
                    continue
                first_point, second_point = project(edge[0]), project(edge[1])
                denominator = first_point[1] - second_point[1]
                if abs(denominator) <= tolerance:
                    continue
                fraction = first_point[1] / denominator
                if not tolerance < fraction < 1.0 - tolerance:
                    continue
                crossing_x = first_point[0] + fraction * (
                    second_point[0] - first_point[0]
                )
                if tolerance < crossing_x < length - tolerance and local_crossing(
                    edge, fraction, crossing_x
                ):
                    crossing.append((crossing_x, edge))
            if not crossing:
                break
            progress = False
            for _, edge in sorted(crossing):
                if edge in flipped_edges:
                    continue
                if self._flip_crossing_edge(edge, project):
                    flipped_edges.add(edge)
                    progress = True
                    break
            if not progress:
                break
        if getattr(self, "intrinsic_insertion", False):
            return self._constrain_corridor(first, second)
        return self.constrained_path(
            first, second, self.points[first], self.points[second]
        )

    def _constrain_corridor(self, first, second):
        """Route one constraint through adjacent faces, including folded charts.

        A dual path selects an actual connected strip of triangles; overlapping
        projections cannot introduce unrelated edge crossings. Each crossed
        edge is split once, omitting no original cell or its unresolved
        action measure. Component provenance is retained. Existing constraints and physical boundaries
        are barriers. This is the mesh-aligned insertion allowed by §10.3.
        """
        owners = {}
        starts, goals = set(), set()
        for index, triangle in enumerate(self.triangles):
            if first in triangle:
                starts.add(index)
            if second in triangle:
                goals.add(index)
            for a, b in zip(triangle, np.roll(triangle, -1)):
                owners.setdefault(tuple(sorted((int(a), int(b)))), []).append(index)
        origin = np.asarray(self.points[first])
        chord = _periodic_delta(origin, self.points[second], self.period)
        length2 = float(np.dot(chord, chord))
        crossings = {}
        adjacency = {}
        for edge, incident in owners.items():
            if len(incident) != 2 or edge in self.constrained_edges:
                continue
            if first in edge or second in edge:
                continue
            a, b = incident
            if self.component_ids[a] != self.component_ids[b]:
                continue
            start = origin + _periodic_delta(origin, self.points[edge[0]], self.period)
            delta = _periodic_delta(
                self.points[edge[0]], self.points[edge[1]], self.period
            )
            # Closest points on the two supporting lines. A vertex hit uses
            # an edge midpoint to keep the path in face interiors rather than
            # branching at that vertex. The existing geometric allowance is
            # checked below; this does not enlarge the resolution bound.
            solution = np.linalg.lstsq(
                np.column_stack((delta, -chord)), origin - start, rcond=None
            )[0]
            fraction = float(solution[0])
            if (
                not self.config.snap_tolerance
                < fraction
                < 1 - self.config.snap_tolerance
            ):
                fraction = 0.5
            point = start + fraction * delta
            parameter = float(np.clip(np.dot(point - origin, chord) / length2, 0, 1))
            gap = float(np.linalg.norm(point - origin - parameter * chord))
            scale = float(np.linalg.norm(delta))
            if gap > self.config.max_surface_distance_ratio * scale:
                continue
            crossings[edge] = (point, fraction, scale)
            weight = scale + 100 * gap * gap / max(
                np.sqrt(length2), self.config.snap_tolerance
            )
            adjacency.setdefault(a, []).append((b, edge, weight))
            adjacency.setdefault(b, []).append((a, edge, weight))
        distances = {index: 0.0 for index in starts}
        queue = [(0.0, index) for index in sorted(starts)]
        previous = {}
        reached = None
        while queue:
            cost, index = heappop(queue)
            if cost != distances[index]:
                continue
            if index in goals:
                reached = index
                break
            for neighbor, edge, weight in adjacency.get(index, ()):
                candidate = cost + weight
                if candidate < distances.get(neighbor, np.inf):
                    distances[neighbor] = candidate
                    previous[neighbor] = (index, edge)
                    heappush(queue, (candidate, neighbor))
        if reached is None:
            raise _TransitionCutConflict(
                "no bounded face corridor joins the mapped transition samples"
            )
        edges = []
        while reached not in starts:
            reached, edge = previous[reached]
            edges.append(edge)
        if len(edges) + 1 > self.config.max_corridor_faces:
            raise _TransitionCutConflict(
                f"constrained insertion needs {len(edges)+1} faces, above the local corridor budget {self.config.max_corridor_faces}"
            )
        chain = [int(first)]
        for edge in reversed(edges):
            point, fraction, scale = crossings[edge]
            inherited = self.tags[edge[0]] & self.tags[edge[1]]
            chain.append(self._split_edge(edge, point, fraction, inherited, scale))
        chain.append(int(second))
        current = self.edges()
        if any(
            tuple(sorted((a, b))) not in current for a, b in zip(chain[:-1], chain[1:])
        ):
            raise _TransitionCutConflict(
                "a face corridor did not produce one connected constrained chain"
            )
        self.corridor_count += 1
        self.max_corridor_faces_used = max(self.max_corridor_faces_used, len(edges) + 1)
        return chain

    def tagged_path(self, first, second, tag):
        adjacency = {}
        for edge in self.edges():
            if all((self.tags[vertex] & tag) != 0 for vertex in edge):
                weight = float(
                    np.linalg.norm(
                        _periodic_delta(
                            self.points[edge[0]], self.points[edge[1]], self.period
                        )
                    )
                )
                adjacency.setdefault(edge[0], []).append((edge[1], weight))
                adjacency.setdefault(edge[1], []).append((edge[0], weight))
        queue = [(0.0, int(first))]
        distance = {int(first): 0.0}
        previous = {}
        while queue:
            cost, vertex = heappop(queue)
            if vertex == second:
                break
            if cost != distance[vertex]:
                continue
            for neighbor, weight in adjacency.get(vertex, []):
                candidate = cost + weight
                if candidate < distance.get(neighbor, np.inf):
                    distance[neighbor] = candidate
                    previous[neighbor] = vertex
                    heappush(queue, (candidate, neighbor))
        if second not in distance:
            raise ConstrainedCutError("critical port samples are not joined on G_ZERO")
        path = [int(second)]
        while path[-1] != first:
            path.append(previous[path[-1]])
        return path[::-1]

    def freeze(self, triangles):
        return SurfaceMesh(
            level=self.level,
            period=self.period,
            points=np.asarray(self.points, dtype=np.float64).reshape(-1, 3),
            triangles=np.asarray(triangles, dtype=np.int64).reshape(-1, 3),
            B=np.asarray(self.B),
            g=np.asarray(self.g),
            boundary_tags=np.asarray(self.tags, dtype=np.int64),
            point_parent_edges=np.asarray(self.parent_edges, dtype=np.int64).reshape(
                -1, 2
            ),
            triangle_parent_tetrahedra=np.asarray(
                self.parent_tetrahedra, dtype=np.int64
            ),
            component_ids=np.asarray(self.component_ids, dtype=np.int64),
        )


@dataclass
class _InsertedTransition:
    transition: TransitionCurve
    sample_ids: np.ndarray
    gamma_ids: np.ndarray
    path_edges: set[tuple[int, int]]
    vertex_u: dict[int, float]
    gamma_vertex_u: dict[int, float]
    event_endpoint_indices: tuple[int, ...] = ()


def _curve_is_closed(transition):
    if len(transition.u) < 2:
        return False
    scale = max(1.0, transition.total_u_length)
    return transition.total_u_length - transition.u[-1] > (
        64.0 * np.finfo(float).eps * scale
    )


def _insert_curve(
    mesh,
    points,
    u,
    total_length,
    closed,
    *,
    tagged=False,
    snap_open_ends=False,
    event_endpoint_indices=(),
    adaptive_anchors=False,
):
    tag = SurfaceMesh.G_ZERO if tagged else 0
    sample_ids = np.array(
        [
            (
                mesh.insert_tagged_point(point, tag)
                if tagged
                else mesh.insert_point(point, preserve=True)
            )
            for point in points
        ],
        dtype=np.int64,
    )
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ConstrainedCutError(
            "distinct transition samples snapped to one mesh vertex"
        )
    path_edges = set()
    vertex_u = {int(vertex): float(value) for vertex, value in zip(sample_ids, u)}
    pairs = list(zip(range(len(sample_ids) - 1), range(1, len(sample_ids))))
    if closed:
        pairs.append((len(sample_ids) - 1, 0))
    for first_index, second_index in pairs:
        first, second = int(sample_ids[first_index]), int(sample_ids[second_index])
        if tagged:
            chain = mesh.tagged_path(first, second, SurfaceMesh.G_ZERO)
        else:
            first_point = np.asarray(points[first_index], dtype=float)
            delta = _periodic_delta(
                first_point, np.asarray(points[second_index], dtype=float), mesh.period
            )
            anchor_ids = [first]
            anchor_points = [first_point]
            anchor_count = mesh.config.path_anchor_count
            if adaptive_anchors:
                # Contact bisection already puts many samples inside a single
                # face. Additional anchors are useful only for longer chords;
                # the same constrained-edge routine still visits every crossed
                # triangle, including when no extra anchor is needed.
                scale = mesh._nearest_location(first_point)[-1]
                anchor_count = min(
                    anchor_count,
                    max(
                        1,
                        int(
                            np.ceil(
                                np.linalg.norm(delta)
                                / max(scale, mesh.config.snap_tolerance)
                            )
                        ),
                    ),
                )
            for anchor_index in range(1, anchor_count):
                fraction = anchor_index / anchor_count
                anchor_point = first_point + fraction * delta
                anchor_point[2] %= mesh.period
                anchor_id = mesh.insert_point(anchor_point, preserve=False)
                if anchor_id != anchor_ids[-1]:
                    anchor_ids.append(anchor_id)
                    anchor_points.append(anchor_point)
            if second != anchor_ids[-1]:
                anchor_ids.append(second)
                anchor_points.append(np.asarray(points[second_index], dtype=float))
            chain = [anchor_ids[0]]
            for anchor_number, (anchor_first, anchor_second) in enumerate(
                zip(anchor_ids[:-1], anchor_ids[1:])
            ):
                try:
                    section = mesh.constrain_edge(anchor_first, anchor_second)
                except ConstrainedCutError as error:
                    raise type(error)(
                        "transition segment "
                        f"{first_index}->{second_index}, anchor {anchor_number} "
                        f"could not be constrained: {error}"
                    ) from error
                mesh.constrained_edges.update(
                    tuple(sorted((int(a), int(b))))
                    for a, b in zip(section[:-1], section[1:])
                )
                chain.extend(section[1:])
            # Adjacent local constraints can briefly choose overlapping
            # triangle corridors. Erase the resulting closed detour before
            # it becomes a branched cut graph; mapped sample endpoints remain
            # fixed and authoritative.
            simple_chain = []
            positions = {}
            for vertex in chain:
                if vertex in positions:
                    keep = positions[vertex]
                    for removed in simple_chain[keep + 1 :]:
                        positions.pop(removed, None)
                    simple_chain = simple_chain[: keep + 1]
                else:
                    positions[vertex] = len(simple_chain)
                    simple_chain.append(vertex)
            chain = simple_chain
        lengths = np.array(
            [
                np.linalg.norm(
                    _periodic_delta(mesh.points[a], mesh.points[b], mesh.period)
                )
                for a, b in zip(chain[:-1], chain[1:])
            ]
        )
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        end_u = total_length if closed and second_index == 0 else float(u[second_index])
        start_u = float(u[first_index])
        if cumulative[-1] > 0.0:
            values = start_u + (end_u - start_u) * cumulative / cumulative[-1]
            for vertex, value in zip(chain[1:-1], values[1:-1]):
                # A path between two samples may pass through a third sample's
                # vertex when the mesh's own chain order disagrees with the
                # certified sample order (ADR 0005); the authoritative sample
                # parameter must win over a passing path's interpolation.
                vertex_u.setdefault(int(vertex), float(value))
        segment_edges = {
            tuple(sorted((int(a), int(b)))) for a, b in zip(chain[:-1], chain[1:])
        }
        path_edges.update(segment_edges)
        mesh.constrained_edges.update(segment_edges)
    if snap_open_ends and not closed:
        # An open T terminates on the EDGE boundary, but its PL endpoint
        # generically stops a fraction of one edge short of the boundary
        # polyline. Extend the cut to the nearest EDGE boundary edge, splitting
        # it so the cut terminates on the surface edge; the extension carries
        # the endpoint's clamped parameter (ADR 0004).
        for endpoint_index in (0, len(sample_ids) - 1):
            if endpoint_index in event_endpoint_indices:
                continue
            endpoint_id = int(sample_ids[endpoint_index])
            if (mesh.tags[endpoint_id] & SurfaceMesh.EDGE) != 0:
                continue
            component = None
            incident_lengths = []
            origin = np.asarray(mesh.points[endpoint_id], dtype=float)
            for triangle_index, triangle in enumerate(mesh.triangles):
                if endpoint_id not in triangle:
                    continue
                component = int(mesh.component_ids[triangle_index])
                for vertex in triangle:
                    if vertex != endpoint_id:
                        incident_lengths.append(
                            float(
                                np.linalg.norm(
                                    _periodic_delta(
                                        origin, mesh.points[vertex], mesh.period
                                    )
                                )
                            )
                        )
            gap, snapped = _nearest_edge_boundary_point(
                mesh, mesh.points[endpoint_id], component
            )
            if snapped is None:
                raise ConstrainedCutError(
                    "surface has no EDGE boundary to terminate an open companion T"
                )
            # The pre-insertion screen bounded this gap against the coarser
            # pre-split mesh; the allowance the snap actually applies must be
            # the one that holds where it is used, on the mutated mesh.
            allowance = mesh.config.max_surface_distance_ratio * max(
                max(incident_lengths, default=0.0), mesh.config.snap_tolerance
            )
            if gap > allowance:
                raise _TransitionCutConflict(
                    f"open T endpoint {endpoint_index} is {gap:.3e} from its "
                    f"component's EDGE boundary on the refined mesh (allowed "
                    f"{allowance:.3e}); the snapped extension would exceed the "
                    "local resolution"
                )
            # The snap is on a known PL EDGE, whose nonlinear field-line
            # chart need not contain that Euclidean point. Preserve the
            # boundary provenance instead of searching chart interiors again.
            try:
                boundary_id = (
                    mesh.insert_tagged_point(
                        snapped, SurfaceMesh.EDGE, component=component
                    )
                    if getattr(mesh, "normal_insertion", False)
                    else mesh.insert_point(snapped, preserve=True)
                )
            except ConstrainedCutError as error:
                raise _TransitionCutConflict(
                    f"open T endpoint {endpoint_index} boundary insertion: {error}"
                ) from error
            if boundary_id == endpoint_id:
                continue
            try:
                chain = mesh.constrain_edge(endpoint_id, boundary_id)
            except ConstrainedCutError as error:
                raise _TransitionCutConflict(
                    f"open T endpoint {endpoint_index} could not be extended "
                    f"to the EDGE boundary: {error}"
                ) from error
            for vertex in chain[1:]:
                vertex_u[int(vertex)] = float(u[endpoint_index])
            extension_edges = {
                tuple(sorted((int(a), int(b)))) for a, b in zip(chain[:-1], chain[1:])
            }
            path_edges.update(extension_edges)
            mesh.constrained_edges.update(extension_edges)
    degree = {}
    for first, second in path_edges:
        degree[first] = degree.get(first, 0) + 1
        degree[second] = degree.get(second, 0) + 1
    ends = set() if closed else {int(sample_ids[0]), int(sample_ids[-1])}
    # Open ends may have been extended to EDGE by ADR 0004.
    if (snap_open_ends or tagged) and not closed:
        ends = {
            vertex
            for vertex, count in degree.items()
            if count == 1
            and (
                (mesh.tags[vertex] & SurfaceMesh.EDGE)
                # A staged critical port can cover only part of G_ZERO;
                # its supplied endpoints are already boundary vertices, not
                # dangling companion cuts. Reordered full curves can instead
                # end at an interior sample carrying EDGE (ADR 0005).
                or (tagged and vertex in (int(sample_ids[0]), int(sample_ids[-1])))
                or vertex
                in {int(sample_ids[index]) for index in event_endpoint_indices}
            )
        }
    invalid = [
        vertex
        for vertex, count in degree.items()
        if count != (1 if vertex in ends else 2)
    ]
    if (
        invalid
        or (not closed and len(ends) != 2)
        or any(int(vertex) not in degree for vertex in sample_ids)
    ):
        raise _TransitionCutConflict(
            f"inserted {'critical' if tagged else 'companion'} curve branches or terminates "
            "away from an authorized endpoint; "
            f"invalid vertices {[(v,degree[v],mesh.tags[v],np.flatnonzero(sample_ids==v).tolist()) for v in invalid[:8]]}"
        )
    return sample_ids, path_edges, vertex_u


def _triangle_components(triangles, blocked_edges=()):
    blocked = {tuple(sorted(map(int, edge))) for edge in blocked_edges}
    union = _UnionFind(len(triangles))
    owner = {}
    for triangle_id, triangle in enumerate(triangles):
        for index in range(3):
            edge = tuple(sorted((triangle[index], triangle[(index + 1) % 3])))
            if edge in blocked:
                continue
            if edge in owner:
                union.union(triangle_id, owner[edge])
            else:
                owner[edge] = triangle_id
    return union.labels()


def _incident_components(triangles, labels, vertex):
    return sorted(
        {
            int(labels[index])
            for index, triangle in enumerate(triangles)
            if int(vertex) in triangle
        }
    )


def _traced_side_action(mesh, component, triangle_labels, sample_ids):
    """Trace one well just inside ``component`` next to the inserted cut.

    Returns ``(sample_index, action)`` from the first adjacent triangle whose
    projected centroid yields a regular trace, or ``None``. This generates the
    same half-bounce action that ``evaluate_surface_data`` would have put on
    the surface, so side assignment stays data-driven even when the caller
    skipped the surface-wide traces (ADR 0004).
    """
    if mesh.field is None:
        return None
    for sample_index, vertex in enumerate(sample_ids):
        for triangle_id, triangle in enumerate(mesh.triangles):
            if triangle_labels[triangle_id] != component or vertex not in triangle:
                continue
            origin = np.asarray(mesh.points[int(vertex)], dtype=float)
            corners = np.asarray(
                [
                    origin + _periodic_delta(origin, mesh.points[item], mesh.period)
                    for item in triangle
                ]
            )
            edge_scale = max(
                float(np.linalg.norm(corners[i] - corners[(i + 1) % 3]))
                for i in range(3)
            )
            probe = corners.mean(axis=0)
            probe[2] %= mesh.period
            projected = _project_to_level_near(
                probe,
                mesh.field,
                mesh.level,
                SurfaceExtractionConfig(B_tolerance=mesh.config.B_tolerance),
                mesh.config.max_surface_distance_ratio * edge_scale,
            )
            if projected is None:
                continue
            s = float(projected[0] ** 2 + projected[1] ** 2)
            theta = float(np.arctan2(projected[1], projected[0])) if s > 0.0 else 0.0
            trace = trace_regular_well(
                mesh.field,
                mesh.level,
                np.array([s, theta, float(projected[2])]),
                mesh.trace_config or WellTraceConfig(),
            )
            if np.isfinite(trace.action_length):
                return sample_index, float(trace.action_length)
    return None


def _branch_components(mesh, inserted, triangle_labels, all_cut_vertices):
    transition = inserted.transition
    parent = next(port for port in transition.ports if port.role == "parent")
    child = next(port for port in transition.ports if port.role == "child_1")
    probes = [
        (
            int(vertex),
            float(parent.action_values[index]),
            float(child.action_values[index]),
        )
        for index, vertex in enumerate(inserted.sample_ids)
        if index not in inserted.event_endpoint_indices
    ]
    edge_probes = []
    if not probes:
        # A short arc between two event junctions has no non-event sample,
        # but its constrained chain still separates two sides: probe the
        # triangles flanking each chain edge, with the expected branch
        # actions taken from the same common-u port interpolation the cut
        # assigns along the chain.
        for edge in sorted(inserted.path_edges):
            first, second = map(int, edge)
            if first not in inserted.vertex_u or second not in inserted.vertex_u:
                continue
            middle_u = 0.5 * (inserted.vertex_u[first] + inserted.vertex_u[second])
            edge_probes.append(
                (
                    (first, second),
                    _interpolate_port_action(transition, parent, middle_u),
                    _interpolate_port_action(transition, child, middle_u),
                )
            )
    if probes:
        adjacent = sorted(
            {
                component
                for vertex, _, _ in probes
                for component in _incident_components(
                    mesh.triangles, triangle_labels, vertex
                )
            }
        )
    else:
        adjacent = sorted(
            {
                int(triangle_labels[triangle_id])
                for (first, second), _, _ in edge_probes
                for triangle_id, triangle in enumerate(mesh.triangles)
                if first in triangle and second in triangle
            }
        )
    if len(adjacent) != 2:
        if len(adjacent) == 1:
            # A slit can leave the surface globally connected — its two local
            # sides meet around its ends — so no trustworthy parent/child
            # sheet assignment exists at this background resolution. Typical
            # of an under-resolved junction complex; the remedy is background
            # refinement, never a guessed one-sheet duplication.
            raise _TransitionCutConflict(
                "companion cut does not separate the surface at this "
                "resolution; refine the background mesh to resolve the "
                "junction complex"
            )
        raise _TransitionCutConflict(
            "a generic companion cut must have exactly two incident triangle sides; "
            f"found {adjacent}"
        )
    costs = np.zeros((2, 2))
    for component_index, component in enumerate(adjacent):
        parent_errors = []
        child_errors = []
        for vertex, parent_expected, child_expected in probes:
            neighbor_values = []
            for triangle_id, triangle in enumerate(mesh.triangles):
                if triangle_labels[triangle_id] != component or vertex not in triangle:
                    continue
                neighbor_values.extend(
                    mesh.action[item]
                    for item in triangle
                    if item not in all_cut_vertices and np.isfinite(mesh.action[item])
                )
            if neighbor_values:
                value = float(np.mean(neighbor_values))
                parent_errors.append(abs(value - parent_expected))
                child_errors.append(abs(value - child_expected))
        for (first, second), parent_expected, child_expected in edge_probes:
            neighbor_values = []
            for triangle_id, triangle in enumerate(mesh.triangles):
                if (
                    triangle_labels[triangle_id] != component
                    or first not in triangle
                    or second not in triangle
                ):
                    continue
                neighbor_values.extend(
                    mesh.action[item]
                    for item in triangle
                    if item not in all_cut_vertices and np.isfinite(mesh.action[item])
                )
            if neighbor_values:
                value = float(np.mean(neighbor_values))
                parent_errors.append(abs(value - parent_expected))
                child_errors.append(abs(value - child_expected))
        if not parent_errors:
            # No finite pre-cut action neighbors this side (e.g. the caller
            # skipped the surface-wide traces). Generate one datum by tracing
            # a well just inside the side instead of guessing (ADR 0004).
            traced = _traced_side_action(
                mesh, component, triangle_labels, inserted.sample_ids
            )
            if traced is None:
                raise _TransitionCutConflict(
                    "a cut side has no finite neighboring action data and no "
                    "traceable probe point"
                )
            sample_index, value = traced
            parent_errors.append(abs(value - parent.action_values[sample_index]))
            child_errors.append(abs(value - child.action_values[sample_index]))
        costs[component_index] = (np.mean(parent_errors), np.mean(child_errors))
    direct = costs[0, 0] + costs[1, 1]
    swapped = costs[0, 1] + costs[1, 0]
    # The assignment must be decided by the data, not by which side of a
    # near-tie a stray trace landed on: the two candidate costs must differ by
    # a decisive fraction of the physical parent/child action jump, or the
    # transition is reported unresolved (docs/STATUS.md hardening item).
    jump = np.abs(parent.action_values - child.action_values)
    jump = jump[np.isfinite(jump)]
    jump_scale = float(np.mean(jump)) if len(jump) else 0.0
    margin = mesh.config.side_assignment_margin_ratio * jump_scale
    if jump_scale <= 0.0 or abs(direct - swapped) < margin:
        raise _TransitionCutConflict(
            f"parent/child side assignment is not decisive: costs "
            f"{direct:.6e} and {swapped:.6e} differ by less than "
            f"{mesh.config.side_assignment_margin_ratio} of the mean "
            f"parent/child action jump {jump_scale:.6e}"
        )
    return (
        (adjacent[0], adjacent[1]) if direct <= swapped else (adjacent[1], adjacent[0])
    )


def _interpolate_port_action(transition, port, parameter):
    if parameter <= transition.u[-1]:
        return float(np.interp(parameter, transition.u, port.action_values))
    return float(
        np.interp(
            parameter,
            [transition.u[-1], transition.total_u_length],
            [port.action_values[-1], port.action_values[0]],
        )
    )


def _is_resolvable(transition):
    return (
        transition.status is TransitionStatus.REGULAR
        and transition.sampling_certified
        and len(transition.u) >= 2
        and not len(transition.contact_sample_pairs)
        and all(
            status is TransitionStatus.REGULAR for status in transition.sample_status
        )
        and all(
            np.all(np.isfinite(port.points)) and np.all(np.isfinite(port.action_values))
            for port in transition.ports
        )
    )


def _nearest_edge_boundary_point(mesh, point, component=None):
    """Return the distance to the PL ``EDGE`` boundary and its closest point.

    With ``component`` given, only boundary edges owned by a triangle of that
    surface component are candidates: an endpoint must never be screened or
    snapped against a disconnected component that happens to be nearby in
    Euclidean coordinates (§21.2).
    """
    allowed = None
    if component is not None:
        allowed = set()
        for triangle_id, triangle in enumerate(mesh.triangles):
            if mesh.component_ids[triangle_id] != component:
                continue
            for index in range(3):
                allowed.add(tuple(sorted((triangle[index], triangle[(index + 1) % 3]))))
    best_distance = np.inf
    best_point = None
    for edge in mesh.edges():
        if allowed is not None and edge not in allowed:
            continue
        if not all((mesh.tags[vertex] & SurfaceMesh.EDGE) != 0 for vertex in edge):
            continue
        first = np.asarray(mesh.points[edge[0]], dtype=float)
        delta = _periodic_delta(first, mesh.points[edge[1]], mesh.period)
        query = first + _periodic_delta(first, point, mesh.period)
        denominator = float(np.dot(delta, delta))
        fraction = (
            0.0
            if denominator == 0.0
            else float(np.clip(np.dot(query - first, delta) / denominator, 0.0, 1.0))
        )
        closest = first + fraction * delta
        distance = float(np.linalg.norm(query - closest))
        if distance < best_distance:
            best_distance = distance
            best_point = closest
    return best_distance, best_point


def _consistent_curve_component(mesh, points):
    """Locate one whole curve on a single surface component, or explain why not.

    Component-provenance enforcement (DESIGN.md §23 milestone 10.3): an
    off-surface projection may be Euclidean-nearest to a disconnected sheet,
    but a transition curve lives on one connected component.  The curve's
    component is the unique component containing *every* sample within the
    local surface-distance allowance.  No unique component means the curve
    cannot be trustworthily placed: cutting anyway would bridge disconnected
    geometry (§21.2), so the caller must keep the transition explicitly
    unresolved.
    """
    components = set()
    for point in points:
        _, triangle_id, *_ = mesh._nearest_location(point)
        components.add(int(mesh.component_ids[triangle_id]))
    if len(components) == 1:
        return next(iter(components)), None
    viable = []
    for component in sorted(components):
        for point in points:
            distance, _, _, _, _, edge_scale = mesh._nearest_location(
                point, component=component
            )
            allowed = mesh.config.max_surface_distance_ratio * max(
                edge_scale, mesh.config.snap_tolerance
            )
            if distance > allowed:
                break
        else:
            viable.append(component)
    if len(viable) == 1:
        return viable[0], None
    if not viable:
        return None, (
            f"companion T samples locate on disconnected surface components "
            f"{sorted(components)} and no single component contains the whole "
            "curve within the surface-distance allowance"
        )
    return None, (
        f"companion T locates within the surface-distance allowance on "
        f"multiple disconnected components {viable}; the assignment is "
        "ambiguous"
    )


def _geometry_resolution_issue(mesh, transition, event_endpoint_indices=()):
    """Return why the current surface cannot represent ``T``, or ``None``."""
    parent = next(port for port in transition.ports if port.role == "parent")
    closed = _curve_is_closed(transition)
    edge_scales = []
    curve_component, component_issue = _consistent_curve_component(mesh, parent.points)
    if component_issue is not None:
        return component_issue
    for index, point in enumerate(parent.points):
        distance, triangle_id, ids, _, barycentric, edge_scale = mesh._nearest_location(
            point, component=curve_component
        )
        allowed = mesh.config.max_surface_distance_ratio * max(
            edge_scale, mesh.config.snap_tolerance
        )
        if distance > allowed:
            return (
                f"companion T is {distance:.3e} from the nearest surface "
                f"triangle (allowed {allowed:.3e})"
            )
        distances = np.array(
            [
                np.linalg.norm(_periodic_delta(point, existing, mesh.period))
                for existing in mesh.points
            ]
        )
        nearest = int(np.argmin(distances))
        if distances[nearest] <= mesh.config.snap_tolerance:
            on_edge_boundary = (mesh.tags[nearest] & SurfaceMesh.EDGE) != 0
        else:
            active = np.flatnonzero(barycentric > mesh.config.snap_tolerance)
            on_edge_boundary = (
                len(active) == 1
                and (mesh.tags[ids[int(active[0])]] & SurfaceMesh.EDGE) != 0
            ) or (
                len(active) == 2
                and all(
                    (mesh.tags[ids[int(local)]] & SurfaceMesh.EDGE) != 0
                    for local in active
                )
            )
        endpoint = not closed and index in (0, len(parent.points) - 1)
        if on_edge_boundary and not endpoint:
            return (
                "companion T collapses onto the EDGE boundary at an interior "
                "sample; refine the background mesh to resolve the intervening sheet"
            )
        if endpoint and not on_edge_boundary and index not in event_endpoint_indices:
            # An open T terminates on the domain boundary (its endpoints share
            # the marginal endpoints' flux surface), but a PL endpoint
            # generically lands a fraction of one edge short of the PL EDGE
            # polyline. Within the local surface-distance allowance the
            # insertion extends the cut to the boundary (ADR 0004); beyond it
            # the terminal segment is genuinely unresolved.
            gap, _ = _nearest_edge_boundary_point(mesh, point, curve_component)
            if gap > allowed:
                return (
                    f"an open companion T endpoint is {gap:.3e} from the EDGE "
                    f"boundary (snap tolerance {allowed:.3e}), so its cut "
                    "cannot terminate on the surface edge; refine the "
                    "background mesh until the endpoint reaches EDGE"
                )
        if not endpoint:
            boundary_distance, _ = _nearest_edge_boundary_point(
                mesh, point, curve_component
            )
            required = mesh.config.min_transition_strip_edge_ratio * edge_scale
            if boundary_distance < required:
                return (
                    f"companion T-to-EDGE strip width {boundary_distance:.3e} is "
                    f"below the local resolution requirement {required:.3e}"
                )
        edge_scales.append(edge_scale)
    # A polyline that doubles back on itself with sub-resolution strand
    # separation bounds a strip the mesh cannot represent; inserting it would
    # overlap its own constrained chain and branch the cut graph. Real
    # GAMMA_MAX mesh-edge chains do this where the extracted curve zigzags at
    # sub-triangle scale, and the companion inherits it (ADR 0005).
    count = len(parent.points)
    triples = [(i - 1, i, i + 1) for i in range(1, count - 1)]
    if closed and count >= 3:
        triples += [(count - 2, count - 1, 0), (count - 1, 0, 1)]
    for before, middle, after in triples:
        first = np.asarray(parent.points[before], dtype=float)
        second = first + _periodic_delta(first, parent.points[middle], mesh.period)
        third = second + _periodic_delta(
            parent.points[middle], parent.points[after], mesh.period
        )
        incoming = second - first
        outgoing = third - second
        if float(np.dot(incoming, outgoing)) >= 0.0:
            continue
        incoming_length = float(np.linalg.norm(incoming))
        if incoming_length <= 0.0:
            continue
        direction = incoming / incoming_length
        offset = third - first
        separation = float(
            np.linalg.norm(offset - np.dot(offset, direction) * direction)
        )
        required = mesh.config.min_transition_strip_edge_ratio * edge_scales[middle]
        if separation < required:
            return (
                f"companion T doubles back on itself at samples "
                f"{before}->{middle}->{after} with strand separation "
                f"{separation:.3e}, below the local resolution requirement "
                f"{required:.3e}; the strip between the strands cannot be "
                "represented at this sampling"
            )
    return None


def _refresh_inserted_actions(mesh, triangles, labels, copy_id, assigned):
    """Refresh off-cut insertion descendants after branch actions are assigned.

    A helper vertex can have been interpolated from a not-yet-assigned point
    on ``T``. Keeping that stale value blends parent and child actions even
    after the topology is correctly cut (DESIGN.md §§10.3 and 25). Evaluate
    the insertion DAG again, using the copy of each stencil vertex on the
    descendant's own sheet. Authoritative port values are never overwritten.
    A stencil that genuinely crosses a final cut cannot interpolate a
    single-valued action and remains explicit ``NaN`` instead of a plausible
    blend; downstream stages must account for that unresolved action (§21.2).
    """
    incidence = [set() for _ in mesh.points]
    for triangle, label in zip(triangles, labels):
        for vertex in triangle:
            incidence[int(vertex)].add(int(label))
    for vertex in sorted(mesh.action_stencils):
        if vertex in assigned or not incidence[vertex]:
            continue
        if len(incidence[vertex]) != 1:
            mesh.action[vertex] = np.nan
            continue
        component = next(iter(incidence[vertex]))
        source_ids, weights = mesh.action_stencils[vertex]
        active = np.flatnonzero(weights != 0.0)
        source_copies = [
            copy_id.get((source_ids[index], component), source_ids[index])
            for index in active
        ]
        if any(component not in incidence[source] for source in source_copies):
            mesh.action[vertex] = np.nan
            continue
        mesh.action[vertex] = float(
            np.dot(weights[active], [mesh.action[source] for source in source_copies])
        )


def cut_surface_at_transitions(
    surface: SurfaceMesh,
    action_values: FloatArray,
    transitions: tuple[TransitionCurve, ...] | list[TransitionCurve],
    *,
    field: BoozerFieldLike | None = None,
    config: ConstrainedCutConfig | None = None,
    trace_config: WellTraceConfig | None = None,
    _event_endpoints=None,
    _event_marginal_points=(),
    _arc_unresolved_reasons=None,
) -> CutSurface:
    """Insert transition polylines, duplicate ``T``, and assign sheets.

    ``action_values`` are pre-cut half-bounce actions in length units. With a
    supplied field, helper vertices are projected to the local ``B=b`` sheet;
    otherwise they lie on the existing PL level surface for analytic tests.
    ``trace_config`` configures the probe well traced for side assignment
    when no finite pre-cut action neighbors a cut side; it should match the
    configuration ``evaluate_surface_data`` was (or would have been) called
    with.  Failed or bracketed nongeneric curves remain explicit unresolved
    ports, never ordinary missing connectivity (DESIGN.md §21.2), and a
    transition whose insertion or side assignment cannot be completed
    trustworthily is demoted to the same explicit unresolved form instead of
    aborting every other transition on the slice.
    """
    config = config or ConstrainedCutConfig()
    action = np.asarray(action_values, dtype=np.float64)
    if action.shape != (len(surface.points),):
        raise ValueError("action_values must have one value per surface point")
    if len({transition.transition_id for transition in transitions}) != len(
        transitions
    ):
        raise ValueError("transition IDs must be unique within one cut surface")
    mesh = _MutableMesh(surface, action, field, config, trace_config)
    event_endpoints = {} if _event_endpoints is None else _event_endpoints
    mesh.normal_insertion = bool(event_endpoints)
    mesh.intrinsic_insertion = bool(event_endpoints)
    preflight_issues = {
        transition.transition_id: _geometry_resolution_issue(
            mesh, transition, event_endpoints.get(transition.transition_id, ())
        )
        for transition in transitions
        if event_endpoints and _is_resolvable(transition)
    }
    # Component-provenance enforcement: every insertion of one curve is
    # restricted to the single component that holds the whole curve within
    # the surface-distance allowance (§23 milestone 10.3, §21.2).
    curve_components = {}
    for transition in transitions:
        if not _is_resolvable(transition) or preflight_issues.get(
            transition.transition_id
        ):
            continue
        assignments = {}
        for role in ("parent", "child_3"):
            port = next(port for port in transition.ports if port.role == role)
            component, issue = _consistent_curve_component(mesh, port.points)
            if issue is not None:
                preflight_issues[transition.transition_id] = (
                    issue if role == "parent" else f"{role}: {issue}"
                )
                break
            assignments[role] = component
        else:
            curve_components[transition.transition_id] = assignments
    if event_endpoints:
        # All event anchors enter before constraints, so later incidence at a
        # true junction does not split/destroy an already constrained chain.
        # A marginal anchor that cannot insert is skipped: without its shared
        # vertex, an incident arc either still terminates consistently or
        # fails its own degree checks and is demoted explicitly — never a
        # silently branched junction.
        for points in _event_marginal_points:
            for point in points:
                try:
                    mesh.insert_tagged_point(point, SurfaceMesh.G_ZERO)
                except ConstrainedCutError:
                    continue
        for role in ("child_3", "parent"):
            for transition in transitions:
                if not _is_resolvable(transition) or preflight_issues.get(
                    transition.transition_id
                ):
                    continue
                port = next(port for port in transition.ports if port.role == role)
                mesh.active_component = curve_components[transition.transition_id][
                    "parent" if role == "parent" else "child_3"
                ]
                try:
                    for point in port.points:
                        if port.role == "child_3":
                            mesh.insert_tagged_point(point, SurfaceMesh.G_ZERO)
                        else:
                            mesh.insert_point(point, preserve=True)
                except ConstrainedCutError as error:
                    # This transition's own anchors cannot enter the surface;
                    # it is demoted with the recorded reason instead of
                    # aborting every other transition on the slice.
                    preflight_issues[transition.transition_id] = (
                        f"{role} anchor insertion: {error}"
                    )
                finally:
                    mesh.active_component = None
    inserted_transitions = []
    unresolved = []
    unresolved_reasons = []
    unresolved_ports = []
    blocked_edges = set()
    for transition in transitions:
        endpoint_indices = event_endpoints.get(transition.transition_id, ())
        if not _is_resolvable(transition):
            unresolved.append(transition.transition_id)
            if transition.status is TransitionStatus.BUDGET_INSUFFICIENT:
                unresolved_reasons.append(transition.sampling_reason)
            else:
                unresolved_reasons.append(
                    (_arc_unresolved_reasons or {}).get(transition.transition_id)
                    or "transition has failed samples or a bracketed nongeneric event"
                )
            for port in transition.ports:
                unresolved_ports.append(
                    CutTransitionPort(
                        transition.transition_id,
                        port.role,
                        -1,
                        np.full(len(port.points), -1, dtype=np.int64),
                        port.action_values,
                    )
                )
            continue
        geometry_issue = preflight_issues.get(
            transition.transition_id
        ) or _geometry_resolution_issue(mesh, transition, endpoint_indices)
        if geometry_issue is not None:
            unresolved.append(transition.transition_id)
            unresolved_reasons.append(geometry_issue)
            for port in transition.ports:
                unresolved_ports.append(
                    CutTransitionPort(
                        transition.transition_id,
                        port.role,
                        -1,
                        np.full(len(port.points), -1, dtype=np.int64),
                        port.action_values,
                    )
                )
            continue
        parent = next(port for port in transition.ports if port.role == "parent")
        child_1 = next(port for port in transition.ports if port.role == "child_1")
        child_3 = next(port for port in transition.ports if port.role == "child_3")
        if not np.allclose(
            parent.points,
            child_1.points,
            atol=config.snap_tolerance,
            rtol=0.0,
        ):
            raise ConstrainedCutError("parent and child-1 do not share one companion T")
        closed = _curve_is_closed(transition)
        components = curve_components.get(transition.transition_id, {})
        try:
            mesh.active_component = components.get("parent")
            sample_ids, path_edges, vertex_u = _insert_curve(
                mesh,
                parent.points,
                transition.u,
                transition.total_u_length,
                closed,
                snap_open_ends=True,
                event_endpoint_indices=endpoint_indices,
                adaptive_anchors=bool(event_endpoints),
            )
            mesh.active_component = components.get("child_3")
            gamma_ids, _, gamma_vertex_u = _insert_curve(
                mesh,
                child_3.points,
                transition.u,
                transition.total_u_length,
                closed,
                tagged=True,
                event_endpoint_indices=endpoint_indices,
            )
        except ConstrainedCutError as conflict:
            # Vertices this transition already inserted are harmless
            # on-surface refinements; its partially constrained chain stays
            # protected but never becomes a cut. The transition itself is an
            # explicit unresolved hyperedge, not a dead pitch slice. This
            # covers projection and location failures of this transition's
            # own insertion as well as chain conflicts: all are recorded
            # per-transition reasons, never an aborted slice (§21.2).
            unresolved.append(transition.transition_id)
            unresolved_reasons.append(str(conflict))
            for port in transition.ports:
                unresolved_ports.append(
                    CutTransitionPort(
                        transition.transition_id,
                        port.role,
                        -1,
                        np.full(len(port.points), -1, dtype=np.int64),
                        port.action_values,
                    )
                )
            continue
        finally:
            mesh.active_component = None
        inserted_transitions.append(
            _InsertedTransition(
                transition,
                sample_ids,
                gamma_ids,
                path_edges,
                vertex_u,
                gamma_vertex_u,
                endpoint_indices,
            )
        )
        blocked_edges.update(path_edges)

    surviving_edges = mesh.edges()
    destroyed = [edge for edge in blocked_edges if edge not in surviving_edges]
    if destroyed:
        # A stale blocked edge would leave a silent gap in the cut and a
        # plausible, wrong sheet graph (§21.2). The split/flip guards make
        # this unreachable; if it fires anyway the cut cannot be trusted.
        raise ConstrainedCutError(
            f"{len(destroyed)} constrained cut edges were destroyed by later "
            "insertions; the cut cannot be trusted"
        )
    while True:
        pre_duplicate_labels = _triangle_components(mesh.triangles, blocked_edges)
        all_cut_vertices = {vertex for edge in blocked_edges for vertex in edge}
        branch_components = {}
        demoted = None
        for inserted in inserted_transitions:
            try:
                branch_components[inserted.transition.transition_id] = (
                    _branch_components(
                        mesh, inserted, pre_duplicate_labels, all_cut_vertices
                    )
                )
            except _TransitionCutConflict as conflict:
                demoted = (inserted, str(conflict))
                break
        if demoted is None:
            break
        # Side assignment for this transition is not trustworthy: withdraw
        # its blocked edges so it splits nothing, report it unresolved, and
        # relabel for the surviving transitions.
        inserted, reason = demoted
        inserted_transitions.remove(inserted)
        unresolved.append(inserted.transition.transition_id)
        unresolved_reasons.append(reason)
        for port in inserted.transition.ports:
            unresolved_ports.append(
                CutTransitionPort(
                    inserted.transition.transition_id,
                    port.role,
                    -1,
                    np.full(len(port.points), -1, dtype=np.int64),
                    port.action_values,
                )
            )
        blocked_edges = set()
        for survivor in inserted_transitions:
            blocked_edges.update(survivor.path_edges)

    copy_id = {}
    for vertex in sorted(all_cut_vertices):
        components = _incident_components(mesh.triangles, pre_duplicate_labels, vertex)
        for index, component in enumerate(components):
            if index == 0:
                new_id = vertex
            else:
                new_id = len(mesh.points)
                mesh.points.append(np.asarray(mesh.points[vertex]).copy())
                mesh.B.append(mesh.B[vertex])
                mesh.g.append(mesh.g[vertex])
                mesh.tags.append(mesh.tags[vertex])
                mesh.parent_edges.append(np.asarray(mesh.parent_edges[vertex]).copy())
                mesh.action.append(mesh.action[vertex])
            copy_id[(vertex, component)] = new_id
    triangles = []
    for triangle_id, triangle in enumerate(mesh.triangles):
        component = int(pre_duplicate_labels[triangle_id])
        triangles.append(
            [copy_id.get((vertex, component), vertex) for vertex in triangle]
        )

    ports = list(unresolved_ports)
    assigned_action_vertices = set()
    for inserted in inserted_transitions:
        transition = inserted.transition
        parent_component, child_component = branch_components[transition.transition_id]
        parent = next(port for port in transition.ports if port.role == "parent")
        child_1 = next(port for port in transition.ports if port.role == "child_1")
        child_3 = next(port for port in transition.ports if port.role == "child_3")
        for vertex, parameter in inserted.vertex_u.items():
            if (vertex, parent_component) not in copy_id or (
                vertex,
                child_component,
            ) not in copy_id:
                raise ConstrainedCutError(
                    f"arc {transition.transition_id} vertex {vertex} at {mesh.points[vertex]} "
                    f"u={parameter} lacks sides {parent_component,child_component}; "
                    f"incident={_incident_components(mesh.triangles,pre_duplicate_labels,vertex)}, "
                    f"cut degree={sum(vertex in edge for edge in blocked_edges)}, "
                    f"arc degree={sum(vertex in edge for edge in inserted.path_edges)}, "
                    f"sample indices={np.flatnonzero(inserted.sample_ids == vertex).tolist()}, "
                    f"tags={mesh.tags[vertex]}"
                )
            parent_copy = copy_id[(vertex, parent_component)]
            child_copy = copy_id[(vertex, child_component)]
            mesh.action[parent_copy] = _interpolate_port_action(
                transition, parent, parameter
            )
            mesh.action[child_copy] = _interpolate_port_action(
                transition, child_1, parameter
            )
            assigned_action_vertices.update((parent_copy, child_copy))
        parent_ids = np.array(
            [copy_id[(int(vertex), parent_component)] for vertex in inserted.sample_ids]
        )
        child_ids = np.array(
            [copy_id[(int(vertex), child_component)] for vertex in inserted.sample_ids]
        )
        gamma_components = {
            component
            for index, vertex in enumerate(inserted.gamma_ids)
            if index not in inserted.event_endpoint_indices
            for component in _incident_components(
                mesh.triangles, pre_duplicate_labels, int(vertex)
            )
        }
        if len(gamma_components) != 1:
            raise ConstrainedCutError(
                "child-3 curve is incident to more than one sheet"
            )
        gamma_component = gamma_components.pop()
        gamma_ids = np.array(
            [
                copy_id.get((int(vertex), gamma_component), int(vertex))
                for vertex in inserted.gamma_ids
            ]
        )
        for vertex, parameter in inserted.gamma_vertex_u.items():
            target = copy_id.get((vertex, gamma_component), vertex)
            mesh.action[target] = _interpolate_port_action(
                transition, child_3, parameter
            )
            assigned_action_vertices.add(target)
        ports.extend(
            (
                CutTransitionPort(
                    transition.transition_id,
                    "parent",
                    parent_component,
                    parent_ids,
                    parent.action_values,
                ),
                CutTransitionPort(
                    transition.transition_id,
                    "child_1",
                    child_component,
                    child_ids,
                    child_1.action_values,
                ),
                CutTransitionPort(
                    transition.transition_id,
                    "child_3",
                    gamma_component,
                    gamma_ids,
                    child_3.action_values,
                ),
            )
        )

    # In a partially cut arrangement, an uncut incident arc can leave two
    # different one-sided event limits on one mesh vertex. Neither limit is
    # the value of that unresolved vertex. Keep both on their ports, mark the
    # vertex unknown, and propagate its NaN through the insertion stencils.
    # Ordinary (non-event) port samples must still have a unique action.
    event_vertices = {
        int(port.polyline_vertex_ids[index])
        for port in ports
        if port.sheet_id >= 0
        for index in event_endpoints.get(port.transition_id, ())
    }
    controls = {item.transition_id: item.controls for item in transitions}
    values_by_vertex = {}
    for port in ports:
        if port.sheet_id < 0:
            continue
        control = controls[port.transition_id]
        for vertex, value in zip(port.polyline_vertex_ids, port.action_values):
            tolerance = control.additivity_atol + control.additivity_rtol * abs(value)
            values_by_vertex.setdefault(int(vertex), []).append((value, tolerance))
    event_unknown = []
    for vertex, values in values_by_vertex.items():
        value, tolerance = values[0]
        if any(abs(other - value) > tolerance + error for other, error in values[1:]):
            if vertex not in event_vertices:
                raise ConstrainedCutError(
                    "incompatible action limits at a non-event port vertex"
                )
            mesh.action[vertex] = np.nan
            event_unknown.append(vertex)
        else:
            mesh.action[vertex] = value

    _refresh_inserted_actions(
        mesh,
        triangles,
        pre_duplicate_labels,
        copy_id,
        assigned_action_vertices,
    )

    final_labels = _triangle_components(triangles)
    label_map = {}
    for old, new in zip(pre_duplicate_labels, final_labels):
        if old in label_map and label_map[old] != new:
            raise ConstrainedCutError("vertex duplication did not separate a cut sheet")
        label_map[int(old)] = int(new)
    ports = [
        CutTransitionPort(
            port.transition_id,
            port.role,
            label_map.get(port.sheet_id, -1),
            port.polyline_vertex_ids,
            port.action_values,
        )
        for port in ports
    ]
    duplicated_edges = []
    for edge in blocked_edges:
        components = set(
            _incident_components(mesh.triangles, pre_duplicate_labels, edge[0])
        )
        components &= set(
            _incident_components(mesh.triangles, pre_duplicate_labels, edge[1])
        )
        for component in components:
            duplicated_edges.append(
                [copy_id[(edge[0], component)], copy_id[(edge[1], component)]]
            )
    frozen = mesh.freeze(triangles)
    return CutSurface(
        frozen,
        np.asarray(mesh.action),
        final_labels,
        np.asarray(duplicated_edges, dtype=np.int64).reshape(-1, 2),
        tuple(ports),
        np.asarray(unresolved, dtype=np.int64),
        tuple(unresolved_reasons),
        corridor_count=mesh.corridor_count,
        max_corridor_faces_used=mesh.max_corridor_faces_used,
        unresolved_event_action_vertex_ids=np.asarray(
            sorted(event_unknown), dtype=np.int64
        ),
    )


def save_cut_surface(path: str | Path, cut: CutSurface) -> None:
    """Serialize authoritative cut topology without Python object pickles."""
    offsets = [0]
    vertex_ids = []
    port_actions = []
    for port in cut.ports:
        vertex_ids.extend(port.polyline_vertex_ids)
        port_actions.extend(port.action_values)
        offsets.append(len(vertex_ids))
    surface = cut.surface
    event_offsets = np.concatenate(
        ([0], np.cumsum([len(event.port_indices) for event in cut.events]))
    )
    event_point_offsets = np.concatenate(
        ([0], np.cumsum([len(event.marginal_points) for event in cut.events]))
    )
    with Path(path).open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.array([3], dtype=np.int64),
            level=np.array([surface.level]),
            period=np.array([surface.period]),
            points=surface.points,
            triangles=surface.triangles,
            B=surface.B,
            g=surface.g,
            boundary_tags=surface.boundary_tags,
            point_parent_edges=surface.point_parent_edges,
            triangle_parent_tetrahedra=surface.triangle_parent_tetrahedra,
            component_ids=surface.component_ids,
            action_values=cut.action_values,
            sheet_ids=cut.sheet_ids,
            cut_edges=cut.cut_edges,
            unresolved_transition_ids=cut.unresolved_transition_ids,
            unresolved_transition_reasons=np.asarray(
                cut.unresolved_transition_reasons, dtype=str
            ),
            corridor_count=np.array([cut.corridor_count], dtype=np.int64),
            max_corridor_faces_used=np.array(
                [cut.max_corridor_faces_used], dtype=np.int64
            ),
            unresolved_event_action_vertex_ids=cut.unresolved_event_action_vertex_ids,
            port_transition_ids=np.array([port.transition_id for port in cut.ports]),
            port_roles=np.array([port.role for port in cut.ports], dtype="U16"),
            port_sheet_ids=np.array([port.sheet_id for port in cut.ports]),
            port_offsets=np.asarray(offsets, dtype=np.int64),
            port_vertex_ids=np.asarray(vertex_ids, dtype=np.int64),
            port_action_values=np.asarray(port_actions, dtype=np.float64),
            event_ids=np.array(
                [event.event_id for event in cut.events], dtype=np.int64
            ),
            event_offsets=event_offsets,
            event_point_offsets=event_point_offsets,
            event_port_indices=np.array(
                [index for event in cut.events for index in event.port_indices],
                dtype=np.int64,
            ),
            event_sample_indices=np.array(
                [index for event in cut.events for index in event.sample_indices],
                dtype=np.int64,
            ),
            event_marginal_points=np.array(
                [point for event in cut.events for point in event.marginal_points],
                dtype=np.float64,
            ).reshape(-1, 3),
            event_unresolved=np.array(
                [event.unresolved for event in cut.events], dtype=bool
            ),
        )


def load_cut_surface(path: str | Path) -> CutSurface:
    """Load a cut topology written by :func:`save_cut_surface`."""
    with np.load(Path(path), allow_pickle=False) as payload:
        version = payload["schema_version"].tolist()
        if version not in ([2], [3]):
            raise ValueError("unsupported cut-surface schema version")
        surface = SurfaceMesh(
            level=float(payload["level"][0]),
            period=float(payload["period"][0]),
            points=payload["points"],
            triangles=payload["triangles"],
            B=payload["B"],
            g=payload["g"],
            boundary_tags=payload["boundary_tags"],
            point_parent_edges=payload["point_parent_edges"],
            triangle_parent_tetrahedra=payload["triangle_parent_tetrahedra"],
            component_ids=payload["component_ids"],
        )
        offsets = payload["port_offsets"]
        ports = []
        for index in range(len(offsets) - 1):
            start, stop = map(int, offsets[index : index + 2])
            ports.append(
                CutTransitionPort(
                    int(payload["port_transition_ids"][index]),
                    str(payload["port_roles"][index]),
                    int(payload["port_sheet_ids"][index]),
                    payload["port_vertex_ids"][start:stop],
                    payload["port_action_values"][start:stop],
                )
            )
        events = []
        if version == [3]:
            for index, event_id in enumerate(payload["event_ids"]):
                start, stop = payload["event_offsets"][index : index + 2]
                point_start, point_stop = payload["event_point_offsets"][
                    index : index + 2
                ]
                events.append(
                    CutEvent(
                        int(event_id),
                        payload["event_marginal_points"][point_start:point_stop],
                        payload["event_port_indices"][start:stop],
                        payload["event_sample_indices"][start:stop],
                        bool(payload["event_unresolved"][index]),
                    )
                )
        return CutSurface(
            surface,
            payload["action_values"],
            payload["sheet_ids"],
            payload["cut_edges"],
            tuple(ports),
            payload["unresolved_transition_ids"],
            tuple(map(str, payload["unresolved_transition_reasons"])),
            tuple(events),
            int(payload["corridor_count"][0]) if "corridor_count" in payload else 0,
            (
                int(payload["max_corridor_faces_used"][0])
                if "max_corridor_faces_used" in payload
                else 0
            ),
            (
                payload["unresolved_event_action_vertex_ids"]
                if "unresolved_event_action_vertex_ids" in payload
                else np.empty(0, dtype=np.int64)
            ),
        )


def cut_surface_at_transition_arcs(surface, action_values, arrangement, **kwargs):
    """Cut regular arcs while retaining arbitrary-port nongeneric event nodes.

    Event endpoints are inserted jointly; global sheet incidence is determined
    after all regular arc constraints exist. An event is not permission to
    guess missing branches: any arc that cannot be cut retains unresolved
    ports, and each event preserves every incident arc's port endpoints.
    """
    endpoints = {
        arc.curve.transition_id: tuple(
            index
            for index, event_id in zip(
                (0, len(arc.curve.u) - 1), arc.endpoint_event_ids
            )
            if event_id >= 0
        )
        for arc in arrangement.arcs
        if any(event_id >= 0 for event_id in arc.endpoint_event_ids)
    }
    cut = cut_surface_at_transitions(
        surface,
        action_values,
        tuple(arc.curve for arc in arrangement.arcs),
        _event_endpoints=endpoints,
        _event_marginal_points=tuple(
            event.marginal_points for event in arrangement.events
        ),
        _arc_unresolved_reasons={
            arc.curve.transition_id: arc.unresolved_reason for arc in arrangement.arcs
        },
        **kwargs,
    )
    events = []
    for event in arrangement.events:
        port_indices = []
        sample_indices = []
        for arc in arrangement.arcs:
            for end, event_id in zip((0, -1), arc.endpoint_event_ids):
                if event_id != event.event_id:
                    continue
                for index, port in enumerate(cut.ports):
                    if port.transition_id == arc.curve.transition_id:
                        port_indices.append(index)
                        sample_indices.append(
                            0 if end == 0 else len(port.action_values) - 1
                        )
        events.append(
            CutEvent(
                event.event_id,
                event.marginal_points,
                np.array(port_indices, dtype=np.int64),
                np.array(sample_indices, dtype=np.int64),
                event.unresolved,
            )
        )
    return replace(cut, events=tuple(events))
