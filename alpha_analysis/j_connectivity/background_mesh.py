"""Axis-regular logical-cylinder background meshes (DESIGN.md §8.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .field import BoozerFieldLike
from .types import BackgroundMesh


@dataclass(frozen=True)
class BackgroundMeshConfig:
    """Resolution of the structured disk extrusion (DESIGN.md §8.2).

    ``n_radial`` counts equal-radius annuli, ``n_poloidal`` counts vertices on
    each ring, and ``n_zeta`` counts intervals over one field period. Logical
    radii and angles are dimensionless; zeta is measured in radians.
    """

    n_radial: int = 4
    n_poloidal: int = 16
    n_zeta: int = 4

    def __post_init__(self) -> None:
        if self.n_radial < 1:
            raise ValueError("n_radial must be at least one")
        if self.n_poloidal < 3:
            raise ValueError("n_poloidal must be at least three")
        if self.n_zeta < 1:
            raise ValueError("n_zeta must be at least one")


class StructuredPrismMeshBackend:
    """Build a deterministic, axis-regular mesh without Gmsh (§§8.1–8.2)."""

    def __init__(self, config: BackgroundMeshConfig | None = None) -> None:
        self.config = config or BackgroundMeshConfig()

    def build(self, field: BoozerFieldLike) -> BackgroundMesh:
        """Triangulate the unit disk and extrude one periodic zeta period.

        Each triangular prism uses a globally indexed Freudenthal split, so
        adjacent prisms choose the same shared-face diagonal. End disks at
        ``zeta=0`` and ``zeta=2*pi/nfp`` remain explicit and pair one-to-one.
        """
        if field.nfp < 1:
            raise ValueError("field.nfp must be positive")
        disk_points, disk_triangles = _triangulate_disk(self.config)
        n_disk = len(disk_points)
        zeta_period = 2.0 * np.pi / field.nfp
        zeta = np.linspace(0.0, zeta_period, self.config.n_zeta + 1)
        points = np.empty(((self.config.n_zeta + 1) * n_disk, 3), dtype=np.float64)
        for plane, zeta_value in enumerate(zeta):
            plane_slice = slice(plane * n_disk, (plane + 1) * n_disk)
            points[plane_slice, :2] = disk_points
            points[plane_slice, 2] = zeta_value

        tetrahedra = _extrude_triangles(disk_triangles, n_disk, self.config.n_zeta)
        volumes = _signed_volumes(points, tetrahedra)
        negative = volumes < 0.0
        saved = tetrahedra[negative, 2].copy()
        tetrahedra[negative, 2] = tetrahedra[negative, 3]
        tetrahedra[negative, 3] = saved
        if np.any(_signed_volumes(points, tetrahedra) <= 0.0):
            raise RuntimeError(
                "structured prism split produced a degenerate tetrahedron"
            )

        tags = np.zeros(len(points), dtype=np.int64)
        radii_squared = np.sum(points[:, :2] ** 2, axis=1)
        tags[radii_squared == 0.0] |= BackgroundMesh.AXIS
        outer = np.isclose(radii_squared, 1.0, rtol=0.0, atol=2e-15)
        tags[outer] |= BackgroundMesh.OUTER
        tags[:n_disk] |= BackgroundMesh.ZETA_MIN
        tags[-n_disk:] |= BackgroundMesh.ZETA_MAX
        pairs = np.column_stack(
            (
                np.arange(n_disk, dtype=np.int64),
                np.arange(n_disk, dtype=np.int64) + self.config.n_zeta * n_disk,
            )
        )

        s = radii_squared
        theta = np.arctan2(points[:, 1], points[:, 0])
        theta[s == 0.0] = 0.0
        field_values = tuple(
            np.asarray(evaluator(s, theta, points[:, 2]), dtype=np.float64)
            for evaluator in (field.B, field.D_B, field.D2_B)
        )
        return BackgroundMesh(
            points=points,
            tetrahedra=tetrahedra,
            periodic_node_pairs=pairs,
            boundary_tags=tags,
            B=field_values[0],
            D_B=field_values[1],
            D2_B=field_values[2],
        )


def signed_tetrahedron_volumes(mesh: BackgroundMesh) -> np.ndarray:
    """Return oriented tetrahedron volumes in logical ``x*y*zeta`` units."""
    return _signed_volumes(mesh.points, mesh.tetrahedra)


def tetrahedron_quality(mesh: BackgroundMesh) -> np.ndarray:
    """Return mean-ratio quality, equal to one for a regular tetrahedron."""
    vertices = mesh.points[mesh.tetrahedra]
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    sum_squared_edges = sum(
        np.sum((vertices[:, first] - vertices[:, second]) ** 2, axis=1)
        for first, second in edge_pairs
    )
    volumes = np.abs(signed_tetrahedron_volumes(mesh))
    return 12.0 * np.cbrt(3.0 * volumes) ** 2 / sum_squared_edges


def _triangulate_disk(config: BackgroundMeshConfig) -> tuple[np.ndarray, np.ndarray]:
    angles = 2.0 * np.pi * np.arange(config.n_poloidal) / config.n_poloidal
    points = [np.array([0.0, 0.0])]
    for radial_index in range(1, config.n_radial + 1):
        radius = radial_index / config.n_radial
        points.extend(
            np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
        )

    def node(ring: int, poloidal: int) -> int:
        return 1 + (ring - 1) * config.n_poloidal + poloidal % config.n_poloidal

    triangles = []
    for poloidal in range(config.n_poloidal):
        triangles.append((0, node(1, poloidal), node(1, poloidal + 1)))
    for ring in range(1, config.n_radial):
        for poloidal in range(config.n_poloidal):
            inner = node(ring, poloidal)
            inner_next = node(ring, poloidal + 1)
            outer = node(ring + 1, poloidal)
            outer_next = node(ring + 1, poloidal + 1)
            triangles.extend(
                ((inner, outer, outer_next), (inner, outer_next, inner_next))
            )
    return np.asarray(points, dtype=np.float64), np.asarray(triangles, dtype=np.int64)


def _extrude_triangles(triangles: np.ndarray, n_disk: int, n_zeta: int) -> np.ndarray:
    tetrahedra = []
    for plane in range(n_zeta):
        offset = plane * n_disk
        for triangle in triangles:
            lower = np.sort(triangle + offset)
            upper = lower + n_disk
            tetrahedra.extend(
                (
                    (lower[0], lower[1], lower[2], upper[2]),
                    (lower[0], lower[1], upper[1], upper[2]),
                    (lower[0], upper[0], upper[1], upper[2]),
                )
            )
    return np.asarray(tetrahedra, dtype=np.int64)


def _signed_volumes(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    vertices = points[tetrahedra]
    matrices = np.stack(
        (
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 0],
            vertices[:, 3] - vertices[:, 0],
        ),
        axis=2,
    )
    return np.linalg.det(matrices) / 6.0
