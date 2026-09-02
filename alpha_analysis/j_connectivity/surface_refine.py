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
    _physical_g,
    _project_to_level_near,
)


class SurfaceDownsamplingError(RuntimeError):
    """A surface could not be coarsened without violating its invariants."""


@dataclass(frozen=True)
class SurfaceDownsamplingReport:
    """Triangle counts and candidate-rejection diagnostics for one coarsening.

    Rejection counts describe candidate heap entries rejected by each safety
    check. Stale entries are not counted, and the same geometric edge can be
    reconsidered after a neighboring collapse.
    """

    input_triangle_count: int
    target_triangle_count: int
    output_triangle_count: int
    tag_protected_rejections: int = 0
    link_condition_rejections: int = 0
    projection_rejections: int = 0
    g_sign_rejections: int = 0
    face_validity_rejections: int = 0
    flux_budget_rejections: int = 0

    @property
    def achieved_reduction(self) -> float:
        """Fraction of input triangles removed (zero for an empty input)."""
        if self.input_triangle_count == 0:
            return 0.0
        return 1.0 - self.output_triangle_count / self.input_triangle_count

    @property
    def reached_target(self) -> bool:
        """Whether the requested triangle-count target was reached."""
        return self.output_triangle_count <= self.target_triangle_count


@dataclass(frozen=True)
class SurfaceDownsamplingResult:
    """A downsampled pitch surface together with its diagnostics."""

    surface: SurfaceMesh
    report: SurfaceDownsamplingReport


@dataclass(frozen=True)
class SurfaceDownsamplingConfig:
    """Controls topology-preserving pitch-surface downsampling.

    ``target_reduction`` is the requested fraction of triangles to remove;
    for example, ``0.8`` requests removal of about 80 percent of the input.
    The target is a request rather than a guarantee because boundary
    preservation, topology, projection, surface-flux, and triangle-quality
    checks take precedence. ``max_flux_relative_error`` bounds drift in the
    axis-regular ``|ds wedge d alpha|`` measure from DESIGN.md \u00a74.4 both
    globally and separately on every original connected component.

    Edge lengths and triangle quality use the logical ``(x, y, zeta)``
    coordinates of DESIGN.md \u00a73.2, with periodic zeta differences locally
    unwrapped.  ``B_tolerance`` and ``g_tolerance`` have the field's units and
    the units of physical ``b dot grad B``, respectively.  Moved vertices may
    travel at most ``max_projection_distance_ratio`` times the collapsed edge
    length while being projected to ``B=b``.
    """

    target_reduction: float = 0.5
    B_tolerance: float = 1.0e-10
    g_tolerance: float = 1.0e-10
    merge_tolerance: float = 1.0e-10
    max_flux_relative_error: float = 5.0e-3
    max_projection_distance_ratio: float = 0.5
    max_normal_deviation_degrees: float = 30.0
    min_triangle_quality: float = 5.0e-2

    def __post_init__(self) -> None:
        if not np.isfinite(self.target_reduction) or not (
            0.0 <= self.target_reduction < 1.0
        ):
            raise ValueError("target_reduction must be finite and in [0, 1)")
        for name in (
            "B_tolerance",
            "g_tolerance",
            "merge_tolerance",
            "max_flux_relative_error",
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
) -> SurfaceDownsamplingResult:
    """Reduce a pitch surface before evaluating bounce integrals.

    The shortest legal edge is collapsed first, preferentially removing tiny
    triangles and narrowing the edge-length distribution.  A simplicial link
    check preserves component topology.  Vertices carrying any boundary tag
    (``EDGE``, ``AXIS``, periodic seam, ``G_ZERO``, or ``G_JUMP``) are never
    moved or removed, so downsampling cannot erase a physical or diagnostic
    boundary.  Each interior replacement point is projected locally to
    ``B=surface.level`` and is rejected if it flips the physical sign of ``g``
    or makes an incident triangle invert or become degenerate. Each collapse
    must also keep the global and each connected component's axis-regular
    surface measure within ``max_flux_relative_error`` of the input mesh.

    The result report makes a shortfall from the requested reduction and the
    binding rejection checks visible. Parent-edge and parent-tetrahedron
    provenance are retained only where geometry and connectivity are
    unchanged; changed entries are set to ``-1`` rather than assigned a false
    parent.
    """
    config = config or SurfaceDownsamplingConfig()
    n_triangles = len(surface.triangles)
    target_triangles = (
        max(1, int(np.ceil((1.0 - config.target_reduction) * n_triangles)))
        if n_triangles
        else 0
    )
    rejection_counts = {
        "tag_protected_rejections": 0,
        "link_condition_rejections": 0,
        "projection_rejections": 0,
        "g_sign_rejections": 0,
        "face_validity_rejections": 0,
        "flux_budget_rejections": 0,
    }

    def make_result(result_surface: SurfaceMesh) -> SurfaceDownsamplingResult:
        return SurfaceDownsamplingResult(
            surface=result_surface,
            report=SurfaceDownsamplingReport(
                input_triangle_count=n_triangles,
                target_triangle_count=target_triangles,
                output_triangle_count=len(result_surface.triangles),
                **rejection_counts,
            ),
        )

    if n_triangles == 0 or config.target_reduction == 0.0:
        return make_result(surface)

    original_signature = _topology_signature(surface.triangles)
    points = np.asarray(surface.points, dtype=np.float64).copy()
    triangles = np.asarray(surface.triangles, dtype=np.int64).copy()
    g_values = np.asarray(surface.g, dtype=np.float64).copy()
    tags = np.asarray(surface.boundary_tags, dtype=np.int64).copy()
    parent_edges = np.asarray(surface.point_parent_edges, dtype=np.int64).copy()
    parent_tetrahedra = np.asarray(
        surface.triangle_parent_tetrahedra, dtype=np.int64
    ).copy()
    reference_normals, reference_qualities = _triangle_normals_and_qualities(
        points[triangles], surface.period
    )
    face_flux = _triangle_flux_measures(points[triangles], field, surface.period)
    original_flux = float(np.sum(face_flux))
    if not np.isfinite(original_flux) or original_flux <= 0.0:
        raise SurfaceDownsamplingError(
            "a nonempty surface must have positive finite |ds wedge d alpha| measure"
        )
    current_flux = original_flux
    _, face_component_indices = np.unique(
        np.asarray(surface.component_ids, dtype=np.int64), return_inverse=True
    )
    original_component_flux = np.bincount(face_component_indices, weights=face_flux)
    if np.any(~np.isfinite(original_component_flux)) or np.any(
        original_component_flux <= 0.0
    ):
        raise SurfaceDownsamplingError(
            "each nonempty component must have positive finite "
            "|ds wedge d alpha| measure"
        )
    current_component_flux = original_component_flux.copy()

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
            rejection_counts["tag_protected_rejections"] += 1
            continue
        shared_faces = vertex_faces[first] & vertex_faces[second]
        if len(shared_faces) not in (1, 2):
            rejection_counts["link_condition_rejections"] += 1
            continue
        if not _satisfies_link_condition(
            first, second, shared_faces, triangles, vertex_faces
        ):
            rejection_counts["link_condition_rejections"] += 1
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
            rejection_counts["projection_rejections"] += 1
            continue
        replacement = _canonicalize_point(
            replacement, surface.period, config.merge_tolerance
        )
        if np.sum(replacement[:2] ** 2) > 1.0 + config.merge_tolerance:
            rejection_counts["projection_rejections"] += 1
            continue
        replacement_B = float(_evaluate_B(field, replacement[np.newaxis, :])[0])
        if abs(replacement_B - surface.level) > config.B_tolerance:
            rejection_counts["projection_rejections"] += 1
            continue
        try:
            replacement_g = float(_physical_g(field, replacement[np.newaxis, :])[0])
        except SurfaceExtractionError:
            rejection_counts["projection_rejections"] += 1
            continue
        if not _preserves_g_sign(
            g_values[keep], g_values[remove], replacement_g, config.g_tolerance
        ):
            rejection_counts["g_sign_rejections"] += 1
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
            reference_normals,
            reference_qualities,
            surface.period,
            minimum_normal_cosine,
            config.min_triangle_quality,
        ):
            rejection_counts["face_validity_rejections"] += 1
            continue
        flux_updates = _replacement_flux_updates(
            keep,
            remove,
            replacement,
            affected_faces,
            triangles,
            face_alive,
            points,
            field,
            surface.period,
        )
        old_local_flux = float(
            np.sum([face_flux[face_id] for face_id in affected_faces])
        )
        candidate_flux = (
            current_flux - old_local_flux + float(np.sum(list(flux_updates.values())))
        )
        candidate_component_flux = current_component_flux.copy()
        for face_id in affected_faces:
            if face_alive[face_id]:
                candidate_component_flux[face_component_indices[face_id]] -= face_flux[
                    face_id
                ]
        for face_id, value in flux_updates.items():
            candidate_component_flux[face_component_indices[face_id]] += value
        if abs(
            candidate_flux - original_flux
        ) > config.max_flux_relative_error * original_flux or np.any(
            np.abs(candidate_component_flux - original_component_flux)
            > config.max_flux_relative_error * original_component_flux
        ):
            rejection_counts["flux_budget_rejections"] += 1
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
                face_flux[face_id] = 0.0
                parent_tetrahedra[face_id] = -1
                alive_triangles -= 1
                continue
            new_triangle = old_triangle.copy()
            new_triangle[new_triangle == remove] = keep
            triangles[face_id] = new_triangle
            face_flux[face_id] = flux_updates[face_id]
            parent_tetrahedra[face_id] = -1
            face_owner[_face_key(new_triangle)] = face_id
            for vertex in new_triangle:
                vertex_faces[int(vertex)].add(face_id)
        current_flux = candidate_flux
        current_component_flux = candidate_component_flux

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
    _validate_reference_geometry(
        points,
        triangles,
        active_faces,
        reference_normals,
        reference_qualities,
        surface.period,
        minimum_normal_cosine,
        config.min_triangle_quality,
    )
    result_face_flux = _triangle_flux_measures(
        result_points[result_triangles], field, surface.period
    )
    result_flux = float(np.sum(result_face_flux))
    if (
        abs(result_flux - original_flux)
        > config.max_flux_relative_error * original_flux
    ):
        raise SurfaceDownsamplingError(
            "downsampled |ds wedge d alpha| measure exceeds its error budget"
        )
    result_component_flux = np.bincount(
        face_component_indices[active_faces],
        weights=result_face_flux,
        minlength=len(original_component_flux),
    )
    if np.any(
        np.abs(result_component_flux - original_component_flux)
        > config.max_flux_relative_error * original_component_flux
    ):
        raise SurfaceDownsamplingError(
            "a downsampled component's |ds wedge d alpha| measure exceeds "
            "its error budget"
        )
    result_components = _component_ids(result_triangles)
    result_signature = _topology_signature(result_triangles)
    if result_signature != original_signature:
        raise SurfaceDownsamplingError(
            "downsampling changed a component's Euler characteristic or boundary count"
        )

    result_parent_edges = parent_edges[used_points]
    result_parent_edges[point_moved[used_points]] = -1
    result_surface = SurfaceMesh(
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
    return make_result(result_surface)


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
    reference_normals: np.ndarray,
    reference_qualities: np.ndarray,
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

        new_vertices = points[new_triangle].copy()
        new_vertices[new_triangle == keep] = replacement
        new_normal, new_quality = _triangle_normal_and_quality(new_vertices, period)
        reference_normal = reference_normals[face_id]
        reference_norm = float(np.linalg.norm(reference_normal))
        new_norm = float(np.linalg.norm(new_normal))
        quality_floor = min(minimum_quality, reference_qualities[face_id])
        if reference_norm == 0.0 or new_norm == 0.0 or new_quality < quality_floor:
            return False
        normal_cosine = float(
            np.dot(reference_normal, new_normal) / (reference_norm * new_norm)
        )
        if normal_cosine < minimum_normal_cosine:
            return False
    return True


def _replacement_flux_updates(
    keep: int,
    remove: int,
    replacement: np.ndarray,
    affected_faces: set[int],
    triangles: np.ndarray,
    face_alive: np.ndarray,
    points: np.ndarray,
    field: BoozerFieldLike,
    period: float,
) -> dict[int, float]:
    face_ids = []
    vertices = []
    for face_id in affected_faces:
        if not face_alive[face_id]:
            continue
        old_triangle = triangles[face_id]
        if keep in old_triangle and remove in old_triangle:
            continue
        new_triangle = old_triangle.copy()
        new_triangle[new_triangle == remove] = keep
        new_vertices = points[new_triangle].copy()
        new_vertices[new_triangle == keep] = replacement
        face_ids.append(face_id)
        vertices.append(new_vertices)
    measures = _triangle_flux_measures(np.asarray(vertices), field, period)
    return {face_id: float(value) for face_id, value in zip(face_ids, measures)}


def _validate_reference_geometry(
    points: np.ndarray,
    triangles: np.ndarray,
    active_faces: np.ndarray,
    reference_normals: np.ndarray,
    reference_qualities: np.ndarray,
    period: float,
    minimum_normal_cosine: float,
    minimum_quality: float,
) -> None:
    for face_id in active_faces:
        normal, quality = _triangle_normal_and_quality(
            points[triangles[face_id]], period
        )
        reference = reference_normals[face_id]
        denominator = float(np.linalg.norm(normal) * np.linalg.norm(reference))
        quality_floor = min(minimum_quality, reference_qualities[face_id])
        if denominator == 0.0 or quality < (1.0 - 1.0e-12) * quality_floor:
            raise SurfaceDownsamplingError(
                "downsampling produced a degenerate or low-quality triangle "
                f"at face {face_id}: quality={quality:.6g}, "
                f"required={quality_floor:.6g}"
            )
        if float(np.dot(normal, reference) / denominator) < minimum_normal_cosine:
            raise SurfaceDownsamplingError(
                "downsampling inverted or over-rotated a triangle"
            )


def _triangle_normals_and_qualities(
    vertices: np.ndarray, period: float
) -> tuple[np.ndarray, np.ndarray]:
    local = np.asarray(vertices, dtype=np.float64).copy().reshape(-1, 3, 3)
    for index in (1, 2):
        difference = local[:, index, 2] - local[:, 0, 2]
        local[:, index, 2] -= period * np.round(difference / period)
    first = local[:, 1] - local[:, 0]
    second = local[:, 2] - local[:, 0]
    third = local[:, 2] - local[:, 1]
    normals = np.cross(first, second)
    edge_square_sum = np.einsum("ij,ij->i", first, first)
    edge_square_sum += np.einsum("ij,ij->i", second, second)
    edge_square_sum += np.einsum("ij,ij->i", third, third)
    qualities = np.divide(
        2.0 * np.sqrt(3.0) * np.linalg.norm(normals, axis=1),
        edge_square_sum,
        out=np.zeros(len(local), dtype=np.float64),
        where=edge_square_sum > 0.0,
    )
    return normals, qualities


def _triangle_flux_measures(
    vertices: np.ndarray, field: BoozerFieldLike, period: float
) -> np.ndarray:
    """Return per-triangle ``|ds wedge d alpha|`` midpoint measures (\u00a74.4)."""
    local = np.asarray(vertices, dtype=np.float64).copy().reshape(-1, 3, 3)
    if not len(local):
        return np.empty(0, dtype=np.float64)
    for index in (1, 2):
        difference = local[:, index, 2] - local[:, 0, 2]
        local[:, index, 2] -= period * np.round(difference / period)
    first = local[:, 1] - local[:, 0]
    second = local[:, 2] - local[:, 0]
    centroids = np.mean(local, axis=1)
    x = centroids[:, 0]
    y = centroids[:, 1]
    iota = np.asarray(field.iota(x * x + y * y), dtype=np.float64)
    omega = 2.0 * (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    omega -= iota * (
        (2.0 * x * first[:, 0] + 2.0 * y * first[:, 1]) * second[:, 2]
        - (2.0 * x * second[:, 0] + 2.0 * y * second[:, 1]) * first[:, 2]
    )
    return 0.5 * np.abs(omega)


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


@dataclass(frozen=True)
class LocalRefinementConfig:
    """Controls for local edge-split refinement near companion curves.

    ``radius_edge_ratio`` scales each triangle's own edge length: a triangle
    within that many local edge lengths of a curve sample is refined.  This
    keeps the operation local (DESIGN.md §23 milestone 10.3), never a global
    remesh.  ``B_tolerance`` accepts a projected interior midpoint on the
    ``B=b`` level, in field units.
    """

    radius_edge_ratio: float = 1.5
    B_tolerance: float = 1.0e-9

    def __post_init__(self) -> None:
        if not np.isfinite(self.radius_edge_ratio) or self.radius_edge_ratio <= 0:
            raise ValueError("radius_edge_ratio must be finite and positive")
        if not np.isfinite(self.B_tolerance) or self.B_tolerance <= 0:
            raise ValueError("B_tolerance must be finite and positive")


@dataclass(frozen=True)
class LocalRefinementReport:
    """Split and rejection counts for one local refinement pass.

    ``new_vertex_edges`` records, per created vertex in creation order, the
    original-surface vertex pair whose edge was split, so the caller can
    extend point-aligned arrays.  A refined vertex never invents data the
    edge did not carry (§21.2): the caller must treat any per-vertex quantity
    that differs between the parents as explicitly unknown.
    """

    input_triangle_count: int
    output_triangle_count: int
    edges_split: int
    projection_rejections: int = 0
    boundary_edges_split: int = 0
    seam_edges_skipped: int = 0
    new_vertex_edges: np.ndarray = None

    def __post_init__(self) -> None:
        edges = (
            np.empty((0, 2), dtype=np.int64)
            if self.new_vertex_edges is None
            else np.asarray(self.new_vertex_edges, dtype=np.int64).reshape(-1, 2)
        )
        object.__setattr__(self, "new_vertex_edges", edges)


def _periodic_edge_delta(first, second, period):
    delta = np.asarray(second, dtype=float) - np.asarray(first, dtype=float)
    delta[2] -= period * np.round(delta[2] / period)
    return delta


def refine_surface_near_curves(
    surface: SurfaceMesh,
    curves,
    field: BoozerFieldLike | None = None,
    config: LocalRefinementConfig | None = None,
) -> tuple[SurfaceMesh, LocalRefinementReport]:
    """Split every splittable edge of the triangles near the given curves.

    Local (not global) surface refinement for §23 milestone 10.3: a triangle
    is refined when a curve sample lies within ``radius_edge_ratio`` of its
    own longest edge length, so the halving is confined to the companion
    neighborhood whose ``min_transition_strip_edge_ratio`` requirement failed.

    Interior midpoints are projected to ``B=b`` along the local gradient with
    a bounded displacement (§8.4); a failed projection leaves that edge
    unsplit and is counted, never replaced by a distant root (§21.2).
    A ``g=0`` boundary midpoint is solved onto the true ``B=b, g=0`` curve
    (the boundary polyline converges to the curve carrying the authoritative
    critical vertices); other boundary midpoints — ``EDGE`` and unresolved
    boundaries — stay exactly on the PL boundary polyline, whose geometry the
    refinement must not move, and carry the exact field value at the PL
    midpoint.  Periodic-seam and axis edges are never split here because
    their twins live on the other seam copy; they are counted as skipped.
    Without a field (analytic PL fixtures) every midpoint is the PL midpoint,
    matching the cut's own analytic insertion behavior.
    """
    config = config or LocalRefinementConfig()
    points = [row.copy() for row in np.asarray(surface.points, dtype=float)]
    triangles = np.asarray(surface.triangles, dtype=np.int64)
    tags = np.asarray(surface.boundary_tags, dtype=np.int64)
    period = float(surface.period)
    samples = (
        np.vstack([np.asarray(curve, dtype=float) for curve in curves])
        if len(curves)
        else np.empty((0, 3))
    )

    edge_triangles: dict[tuple[int, int], list[int]] = {}
    for triangle_id, triangle in enumerate(triangles):
        for local in range(3):
            edge = tuple(sorted((int(triangle[local]), int(triangle[(local + 1) % 3]))))
            edge_triangles.setdefault(edge, []).append(triangle_id)

    near_triangles = []
    for triangle_id, triangle in enumerate(triangles):
        vertices = np.asarray([points[index] for index in triangle])
        edge_scale = max(
            np.linalg.norm(
                _periodic_edge_delta(vertices[i], vertices[(i + 1) % 3], period)
            )
            for i in range(3)
        )
        if not len(samples):
            continue
        deltas = samples[:, np.newaxis, :] - vertices[np.newaxis, :, :]
        deltas[:, :, 2] -= period * np.round(deltas[:, :, 2] / period)
        distance = float(np.min(np.linalg.norm(deltas, axis=2)))
        # A point within radius*scale of the triangle is within
        # (radius+1)*scale of one of its vertices, so this vertex test never
        # under-selects the true neighborhood.
        if distance <= (config.radius_edge_ratio + 1.0) * edge_scale:
            near_triangles.append(triangle_id)

    target_edges = set()
    for triangle_id in near_triangles:
        triangle = triangles[triangle_id]
        for local in range(3):
            target_edges.add(
                tuple(sorted((int(triangle[local]), int(triangle[(local + 1) % 3]))))
            )

    B = list(map(float, surface.B))
    g = list(map(float, surface.g))
    tag_list = list(map(int, tags))
    new_vertex_edges = []
    midpoint_ids: dict[tuple[int, int], int] = {}
    seam_skipped = 0
    projection_rejections = 0
    boundary_split = 0
    for edge in sorted(target_edges):
        first, second = edge
        joint = tags[first] & tags[second]
        if joint & (SurfaceMesh.PERIODIC_SEAM | SurfaceMesh.AXIS):
            seam_skipped += 1
            continue
        boundary = len(edge_triangles[edge]) == 1
        if boundary and (joint & (SurfaceMesh.G_JUMP,)[0]):
            # An unresolved sheet-bridging boundary needs background
            # refinement, not a midpoint on an unknown curve (ADR 0001).
            continue
        delta = _periodic_edge_delta(points[first], points[second], period)
        length = float(np.linalg.norm(delta))
        midpoint = np.asarray(points[first], dtype=float) + 0.5 * delta
        midpoint[2] %= period
        if field is not None and boundary and (joint & SurfaceMesh.G_ZERO):
            # A g=0 boundary midpoint goes onto the true B=b, g=0 curve, so
            # the refined boundary polyline converges to the curve the
            # authoritative critical vertices live on; a PL midpoint would
            # freeze the boundary's discretization gap while insertion
            # allowances shrink with the local edge scale.
            from .critical_curves import (
                CriticalCurveConfig,
                _projected_midpoint,
            )

            projected = _projected_midpoint(
                np.asarray(points[first], dtype=float),
                np.asarray(points[second], dtype=float),
                midpoint,
                field,
                float(surface.level),
                period,
                CriticalCurveConfig(),
            )
            if projected is None:
                projection_rejections += 1
                continue
            midpoint = np.asarray(projected, dtype=float)
        elif field is not None and not boundary:
            projected = _project_to_level_near(
                midpoint,
                field,
                float(surface.level),
                SurfaceExtractionConfig(B_tolerance=config.B_tolerance),
                0.5 * length,
            )
            if projected is None:
                projection_rejections += 1
                continue
            midpoint = projected
        if field is not None:
            midpoint_B = float(_evaluate_B(field, midpoint[np.newaxis, :])[0])
            midpoint_g = float(_physical_g(field, midpoint[np.newaxis, :])[0])
        else:
            midpoint_B = 0.5 * (B[first] + B[second])
            midpoint_g = 0.5 * (g[first] + g[second])
        points.append(midpoint)
        B.append(midpoint_B)
        g.append(midpoint_g)
        tag_list.append(int(joint) if boundary else 0)
        midpoint_ids[edge] = len(points) - 1
        new_vertex_edges.append(edge)
        boundary_split += int(boundary)

    new_triangles = []
    new_components = []
    new_parents = []
    for triangle_id, triangle in enumerate(triangles):
        corner_ids = list(map(int, triangle))
        split = [
            midpoint_ids.get(
                tuple(sorted((corner_ids[local], corner_ids[(local + 1) % 3])))
            )
            for local in range(3)
        ]
        present = [local for local in range(3) if split[local] is not None]
        if not present:
            children = [corner_ids]
        elif len(present) == 3:
            children = [
                [corner_ids[0], split[0], split[2]],
                [split[0], corner_ids[1], split[1]],
                [split[2], split[1], corner_ids[2]],
                [split[0], split[1], split[2]],
            ]
        elif len(present) == 2:
            missing = ({0, 1, 2} - set(present)).pop()
            a = (missing + 1) % 3
            b = (missing + 2) % 3
            # Edges a (between corners a, b) and b (between corners b,
            # missing) are split; corner b touches both midpoints.
            children = [
                [corner_ids[a], split[a], corner_ids[missing]],
                [split[a], corner_ids[b], split[b]],
                [split[a], split[b], corner_ids[missing]],
            ]
        else:
            local = present[0]
            a, b, c = local, (local + 1) % 3, (local + 2) % 3
            children = [
                [corner_ids[a], split[local], corner_ids[c]],
                [split[local], corner_ids[b], corner_ids[c]],
            ]
        for child in children:
            new_triangles.append(child)
            new_components.append(int(surface.component_ids[triangle_id]))
            new_parents.append(int(surface.triangle_parent_tetrahedra[triangle_id]))

    parent_edges = np.vstack(
        (
            np.asarray(surface.point_parent_edges, dtype=np.int64),
            np.full((len(new_vertex_edges), 2), -1, dtype=np.int64),
        )
    )
    refined = SurfaceMesh(
        level=float(surface.level),
        period=period,
        points=np.asarray(points, dtype=float),
        triangles=np.asarray(new_triangles, dtype=np.int64),
        B=np.asarray(B, dtype=float),
        g=np.asarray(g, dtype=float),
        boundary_tags=np.asarray(tag_list, dtype=np.int64),
        point_parent_edges=parent_edges,
        triangle_parent_tetrahedra=np.asarray(new_parents, dtype=np.int64),
        component_ids=np.asarray(new_components, dtype=np.int64),
    )
    report = LocalRefinementReport(
        input_triangle_count=len(triangles),
        output_triangle_count=len(new_triangles),
        edges_split=len(new_vertex_edges),
        projection_rejections=projection_rejections,
        boundary_edges_split=boundary_split,
        seam_edges_skipped=seam_skipped,
        new_vertex_edges=np.asarray(new_vertex_edges, dtype=np.int64).reshape(-1, 2),
    )
    return refined, report
