"""Pitch-surface coarsening and projection (DESIGN.md \u00a78.4).

The authoritative mesh remains plain NumPy arrays.  Downsampling uses
topology-preserving shortest-edge collapses, then projects every moved vertex
back to the requested ``B=b`` level before any bounce data are evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from .field import BoozerFieldLike
from .surface_extract import (
    SurfaceExtractionConfig,
    SurfaceExtractionError,
    SurfaceMesh,
    _canonicalize_point,
    _component_ids,
    _evaluate_B,
    _orient_triangles,
    _physical_g,
    _project_to_level_near,
)


class SurfaceDownsamplingError(RuntimeError):
    """A surface could not be coarsened without violating its invariants."""


@dataclass(frozen=True)
class SurfaceDownsamplingConfig:
    """Controls topology-preserving pitch-surface downsampling.

    ``target_reduction`` is the requested fraction of triangles to remove;
    for example, ``0.8`` requests about 20 percent of the input triangles.
    The target is a request rather than a guarantee because boundary
    preservation, topology, projection, and triangle-quality checks take
    precedence.

    Edge lengths and triangle quality use the logical ``(x, y, zeta)``
    coordinates of DESIGN.md \u00a73.2, with periodic zeta differences locally
    unwrapped.  ``B_tolerance`` and ``g_tolerance`` have the field's units and
    the units of physical ``b dot grad B``, respectively.  Moved vertices may
    travel at most ``max_projection_distance_ratio`` times the collapsed edge
    length while being projected to ``B=b``.
    """

    target_reduction: float = 0.8
    B_tolerance: float = 1.0e-10
    g_tolerance: float = 1.0e-10
    merge_tolerance: float = 1.0e-10
    max_projection_distance_ratio: float = 0.5
    max_normal_deviation_degrees: float = 75.0
    min_triangle_quality: float = 1.0e-4

    def __post_init__(self) -> None:
        if not np.isfinite(self.target_reduction) or not (
            0.0 <= self.target_reduction < 1.0
        ):
            raise ValueError("target_reduction must be finite and in [0, 1)")
        for name in (
            "B_tolerance",
            "g_tolerance",
            "merge_tolerance",
            "max_projection_distance_ratio",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.min_triangle_quality) or not (
            0.0 < self.min_triangle_quality <= 1.0
        ):
            raise ValueError("min_triangle_quality must be finite and in (0, 1]")
        if not np.isfinite(self.max_normal_deviation_degrees) or not (
            0.0 < self.max_normal_deviation_degrees < 90.0
        ):
            raise ValueError(
                "max_normal_deviation_degrees must be finite and in (0, 90)"
            )


def downsample_surface(
    surface: SurfaceMesh,
    field: BoozerFieldLike,
    config: SurfaceDownsamplingConfig | None = None,
) -> SurfaceMesh:
    """Reduce a pitch surface before evaluating bounce integrals.

    The shortest legal edge is collapsed first, preferentially removing tiny
    triangles and narrowing the edge-length distribution.  A simplicial link
    check preserves component topology.  Vertices carrying any boundary tag
    (``EDGE``, ``AXIS``, periodic seam, ``G_ZERO``, or ``G_JUMP``) are never
    moved or removed, so downsampling cannot erase a physical or diagnostic
    boundary.  Each interior replacement point is projected locally to
    ``B=surface.level`` and is rejected if it flips the physical sign of ``g``
    or makes an incident triangle invert or become degenerate.

    The returned mesh may contain more triangles than requested when no more
    safe collapse exists.  Parent-edge and parent-tetrahedron provenance are
    retained only where geometry and connectivity are unchanged; changed
    entries are set to ``-1`` rather than assigned a false parent.
    """
    config = config or SurfaceDownsamplingConfig()
    n_triangles = len(surface.triangles)
    if n_triangles == 0 or config.target_reduction == 0.0:
        return surface

    original_signature = _topology_signature(surface.triangles)
    target_triangles = max(
        1, int(np.ceil((1.0 - config.target_reduction) * n_triangles))
    )
    points = np.asarray(surface.points, dtype=np.float64).copy()
    triangles = np.asarray(surface.triangles, dtype=np.int64).copy()
    g_values = np.asarray(surface.g, dtype=np.float64).copy()
    tags = np.asarray(surface.boundary_tags, dtype=np.int64).copy()
    parent_edges = np.asarray(surface.point_parent_edges, dtype=np.int64).copy()
    parent_tetrahedra = np.asarray(
        surface.triangle_parent_tetrahedra, dtype=np.int64
    ).copy()

    n_points = len(points)
    point_alive = np.ones(n_points, dtype=bool)
    point_moved = np.zeros(n_points, dtype=bool)
    face_alive = np.ones(n_triangles, dtype=bool)
    vertex_faces = [set() for _ in range(n_points)]
    face_owner: dict[tuple[int, int, int], int] = {}
    for face_id, triangle in enumerate(triangles):
        key = _face_key(triangle)
        if key in face_owner:
            raise SurfaceDownsamplingError("surface contains duplicate triangles")
        face_owner[key] = face_id
        for vertex in triangle:
            vertex_faces[int(vertex)].add(face_id)

    versions = np.zeros(n_points, dtype=np.int64)
    candidates: list[tuple[float, int, int, int, int]] = []

    def push_edge(first: int, second: int) -> None:
        first, second = sorted((int(first), int(second)))
        if first == second or not point_alive[first] or not point_alive[second]:
            return
        if not (vertex_faces[first] & vertex_faces[second]):
            return
        heapq.heappush(
            candidates,
            (
                _periodic_edge_length(points[first], points[second], surface.period),
                first,
                second,
                int(versions[first]),
                int(versions[second]),
            ),
        )

    for triangle in triangles:
        push_edge(int(triangle[0]), int(triangle[1]))
        push_edge(int(triangle[1]), int(triangle[2]))
        push_edge(int(triangle[2]), int(triangle[0]))

    alive_triangles = n_triangles
    extraction_config = SurfaceExtractionConfig(
        B_tolerance=config.B_tolerance,
        g_tolerance=config.g_tolerance,
        merge_tolerance=config.merge_tolerance,
    )
    minimum_normal_cosine = np.cos(np.deg2rad(config.max_normal_deviation_degrees))

    while candidates and alive_triangles > target_triangles:
        edge_length, first, second, first_version, second_version = heapq.heappop(
            candidates
        )
        if not point_alive[first] or not point_alive[second]:
            continue
        if versions[first] != first_version or versions[second] != second_version:
            push_edge(first, second)
            continue
        # All provenance-bearing vertices are fixed.  In particular, this
        # leaves the physical edge, g=0 boundary, and quotient seam unchanged.
        if tags[first] != 0 or tags[second] != 0:
            continue
        shared_faces = vertex_faces[first] & vertex_faces[second]
        if len(shared_faces) not in (1, 2):
            continue
        if not _satisfies_link_condition(
            first, second, shared_faces, triangles, vertex_faces
        ):
            continue

        keep, remove = first, second
        seed = _periodic_midpoint(points[keep], points[remove], surface.period)
        maximum_displacement = max(
            config.max_projection_distance_ratio * edge_length,
            config.merge_tolerance,
        )
        replacement = _project_to_level_near(
            seed,
            field,
            surface.level,
            extraction_config,
            maximum_displacement,
        )
        if replacement is None:
            continue
        replacement = _canonicalize_point(
            replacement, surface.period, config.merge_tolerance
        )
        if np.sum(replacement[:2] ** 2) > 1.0 + config.merge_tolerance:
            continue
        replacement_B = float(_evaluate_B(field, replacement[np.newaxis, :])[0])
        if abs(replacement_B - surface.level) > config.B_tolerance:
            continue
        try:
            replacement_g = float(_physical_g(field, replacement[np.newaxis, :])[0])
        except SurfaceExtractionError:
            continue
        if not _preserves_g_sign(
            g_values[keep], g_values[remove], replacement_g, config.g_tolerance
        ):
            continue

        affected_faces = vertex_faces[keep] | vertex_faces[remove]
        if not _valid_replacement_faces(
            keep,
            remove,
            replacement,
            affected_faces,
            triangles,
            face_alive,
            face_owner,
            points,
            surface.period,
            minimum_normal_cosine,
            config.min_triangle_quality,
        ):
            continue

        old_keep_point = points[keep].copy()
        old_faces = {face_id: triangles[face_id].copy() for face_id in affected_faces}
        for face_id, triangle in old_faces.items():
            face_owner.pop(_face_key(triangle), None)
            for vertex in triangle:
                vertex_faces[int(vertex)].discard(face_id)

        points[keep] = replacement
        g_values[keep] = replacement_g
        point_moved[keep] = True
        parent_edges[keep] = -1
        point_alive[remove] = False
        versions[keep] += 1
        versions[remove] += 1

        for face_id, old_triangle in old_faces.items():
            if keep in old_triangle and remove in old_triangle:
                face_alive[face_id] = False
                parent_tetrahedra[face_id] = -1
                alive_triangles -= 1
                continue
            new_triangle = old_triangle.copy()
            new_triangle[new_triangle == remove] = keep
            triangles[face_id] = new_triangle
            parent_tetrahedra[face_id] = -1
            face_owner[_face_key(new_triangle)] = face_id
            for vertex in new_triangle:
                vertex_faces[int(vertex)].add(face_id)

        # Moving ``keep`` invalidates parent-tetrahedron provenance for its
        # entire one-ring, including faces whose vertex IDs did not change.
        if not np.array_equal(old_keep_point, replacement):
            for face_id in vertex_faces[keep]:
                parent_tetrahedra[face_id] = -1

        neighboring_vertices: set[int] = set()
        for face_id in vertex_faces[keep]:
            neighboring_vertices.update(map(int, triangles[face_id]))
        for vertex in neighboring_vertices:
            if vertex != keep:
                push_edge(keep, vertex)
            for face_id in vertex_faces[vertex]:
                triangle = triangles[face_id]
                push_edge(int(triangle[0]), int(triangle[1]))
                push_edge(int(triangle[1]), int(triangle[2]))
                push_edge(int(triangle[2]), int(triangle[0]))

    active_faces = np.flatnonzero(face_alive)
    result_triangles = triangles[active_faces]
    used_points = (
        np.unique(result_triangles)
        if len(result_triangles)
        else np.empty(0, dtype=np.int64)
    )
    protected_points = np.flatnonzero((tags != 0) & point_alive)
    if np.any(~np.isin(protected_points, used_points)):
        raise SurfaceDownsamplingError("downsampling isolated a tagged boundary vertex")
    point_remap = np.full(n_points, -1, dtype=np.int64)
    point_remap[used_points] = np.arange(len(used_points), dtype=np.int64)
    result_triangles = point_remap[result_triangles]
    result_points = points[used_points]
    result_B = _evaluate_B(field, result_points)
    if len(result_B) and np.max(np.abs(result_B - surface.level)) > config.B_tolerance:
        raise SurfaceDownsamplingError("a downsampled vertex is not on B=b")
    result_g = _physical_g(field, result_points)
    result_triangles = _orient_triangles(
        result_points, result_triangles, field, surface.period
    )
    result_components = _component_ids(result_triangles)
    result_signature = _topology_signature(result_triangles)
    if result_signature != original_signature:
        raise SurfaceDownsamplingError(
            "downsampling changed a component's Euler characteristic or boundary count"
        )

    result_parent_edges = parent_edges[used_points]
    result_parent_edges[point_moved[used_points]] = -1
    return SurfaceMesh(
        level=surface.level,
        period=surface.period,
        points=result_points,
        triangles=result_triangles,
        B=result_B,
        g=result_g,
        boundary_tags=tags[used_points],
        point_parent_edges=result_parent_edges,
        triangle_parent_tetrahedra=parent_tetrahedra[active_faces],
        component_ids=result_components,
    )


def _face_key(triangle: np.ndarray) -> tuple[int, int, int]:
    return tuple(sorted(map(int, triangle)))


def _neighbors(vertex: int, triangles: np.ndarray, vertex_faces) -> set[int]:
    result: set[int] = set()
    for face_id in vertex_faces[vertex]:
        result.update(map(int, triangles[face_id]))
    result.discard(vertex)
    return result


def _satisfies_link_condition(
    first: int,
    second: int,
    shared_faces: set[int],
    triangles: np.ndarray,
    vertex_faces,
) -> bool:
    common_neighbors = _neighbors(first, triangles, vertex_faces) & _neighbors(
        second, triangles, vertex_faces
    )
    edge_link: set[int] = set()
    for face_id in shared_faces:
        edge_link.update(map(int, triangles[face_id]))
    edge_link.discard(first)
    edge_link.discard(second)
    return common_neighbors == edge_link


def _valid_replacement_faces(
    keep: int,
    remove: int,
    replacement: np.ndarray,
    affected_faces: set[int],
    triangles: np.ndarray,
    face_alive: np.ndarray,
    face_owner: dict[tuple[int, int, int], int],
    points: np.ndarray,
    period: float,
    minimum_normal_cosine: float,
    minimum_quality: float,
) -> bool:
    new_keys: set[tuple[int, int, int]] = set()
    for face_id in affected_faces:
        if not face_alive[face_id]:
            continue
        old_triangle = triangles[face_id]
        if keep in old_triangle and remove in old_triangle:
            continue
        new_triangle = old_triangle.copy()
        new_triangle[new_triangle == remove] = keep
        key = _face_key(new_triangle)
        if key in new_keys:
            return False
        owner = face_owner.get(key)
        if owner is not None and owner not in affected_faces:
            return False
        new_keys.add(key)

        old_vertices = points[old_triangle]
        new_vertices = points[new_triangle].copy()
        new_vertices[new_triangle == keep] = replacement
        old_normal, _ = _triangle_normal_and_quality(old_vertices, period)
        new_normal, new_quality = _triangle_normal_and_quality(new_vertices, period)
        old_norm = float(np.linalg.norm(old_normal))
        new_norm = float(np.linalg.norm(new_normal))
        if old_norm == 0.0 or new_norm == 0.0 or new_quality < minimum_quality:
            return False
        normal_cosine = float(np.dot(old_normal, new_normal) / (old_norm * new_norm))
        if normal_cosine < minimum_normal_cosine:
            return False
    return True


def _triangle_normal_and_quality(
    vertices: np.ndarray, period: float
) -> tuple[np.ndarray, float]:
    local = np.asarray(vertices, dtype=np.float64).copy()
    for index in (1, 2):
        difference = local[index, 2] - local[0, 2]
        local[index, 2] -= period * np.round(difference / period)
    first = local[1] - local[0]
    second = local[2] - local[0]
    third = local[2] - local[1]
    normal = np.cross(first, second)
    edge_square_sum = float(
        np.dot(first, first) + np.dot(second, second) + np.dot(third, third)
    )
    quality = (
        2.0 * np.sqrt(3.0) * float(np.linalg.norm(normal)) / edge_square_sum
        if edge_square_sum > 0.0
        else 0.0
    )
    return normal, quality


def _periodic_midpoint(
    first: np.ndarray, second: np.ndarray, period: float
) -> np.ndarray:
    local_second = np.asarray(second, dtype=np.float64).copy()
    difference = local_second[2] - first[2]
    local_second[2] -= period * np.round(difference / period)
    return 0.5 * (np.asarray(first, dtype=np.float64) + local_second)


def _periodic_edge_length(
    first: np.ndarray, second: np.ndarray, period: float
) -> float:
    difference = np.asarray(second, dtype=np.float64) - np.asarray(
        first, dtype=np.float64
    )
    difference[2] -= period * np.round(difference[2] / period)
    return float(np.linalg.norm(difference))


def _preserves_g_sign(
    first: float, second: float, replacement: float, tolerance: float
) -> bool:
    def sign(value: float) -> int:
        if value > tolerance:
            return 1
        if value < -tolerance:
            return -1
        return 0

    endpoint_signs = {sign(first), sign(second)} - {0}
    if len(endpoint_signs) > 1:
        return False
    return not endpoint_signs or sign(replacement) in endpoint_signs


def _topology_signature(triangles: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return sorted ``(Euler characteristic, boundary-loop count)`` pairs."""
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if not len(triangles):
        return ()
    component_ids = _component_ids(triangles)
    signature = []
    for component_id in np.unique(component_ids):
        component = triangles[component_ids == component_id]
        vertices = np.unique(component)
        edge_counts: dict[tuple[int, int], int] = {}
        for triangle in component:
            for first, second in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ):
                edge = tuple(sorted((int(first), int(second))))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        if any(count > 2 for count in edge_counts.values()):
            raise SurfaceDownsamplingError("surface has a non-manifold edge")
        euler = len(vertices) - len(edge_counts) + len(component)
        boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
        boundary_components = _edge_component_count(boundary_edges)
        signature.append((euler, boundary_components))
    return tuple(sorted(signature))


def _edge_component_count(edges: list[tuple[int, int]]) -> int:
    if not edges:
        return 0
    parent: dict[int, int] = {}

    def find(vertex: int) -> int:
        parent.setdefault(vertex, vertex)
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    for first, second in edges:
        union(first, second)
    return len({find(vertex) for vertex in parent})
