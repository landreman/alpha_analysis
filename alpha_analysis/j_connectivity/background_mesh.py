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


@dataclass(frozen=True)
class GmshBackgroundMeshConfig:
    """Resolution controls for the production Gmsh backend (§8.2).

    Sizes and radii are dimensionless logical-cylinder lengths. Critical
    points use ``(x, y, zeta)`` with zeta in radians. When both gradient
    controls are set, locations whose logical-coordinate gradient magnitude
    is below ``low_gradient_threshold`` receive ``low_gradient_size``.
    """

    target_size: float = 0.35
    axis_size: float | None = None
    axis_radius: float = 0.2
    critical_points: tuple[tuple[float, float, float], ...] = ()
    critical_size: float | None = None
    critical_radius: float = 0.2
    low_gradient_size: float | None = None
    low_gradient_threshold: float | None = None

    def __post_init__(self) -> None:
        positive = {
            "target_size": self.target_size,
            "axis_radius": self.axis_radius,
            "critical_radius": self.critical_radius,
        }
        positive.update(
            (name, value)
            for name, value in (
                ("axis_size", self.axis_size),
                ("critical_size", self.critical_size),
                ("low_gradient_size", self.low_gradient_size),
            )
            if value is not None
        )
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if bool(self.critical_points) != (self.critical_size is not None):
            raise ValueError(
                "critical_points and critical_size must be provided together"
            )
        if (self.low_gradient_size is None) != (self.low_gradient_threshold is None):
            raise ValueError(
                "low_gradient_size and low_gradient_threshold must be provided together"
            )
        if (
            self.low_gradient_threshold is not None
            and self.low_gradient_threshold < 0.0
        ):
            raise ValueError("low_gradient_threshold must be nonnegative")
        for point in self.critical_points:
            if len(point) != 3 or not np.all(np.isfinite(point)):
                raise ValueError(
                    "critical_points must contain finite (x, y, zeta) points"
                )


class GmshBackgroundMeshBackend:
    """Build a periodic logical-cylinder mesh with optional Gmsh (§8.2)."""

    def __init__(self, config: GmshBackgroundMeshConfig | None = None) -> None:
        self.config = config or GmshBackgroundMeshConfig()

    def build(self, field: BoozerFieldLike) -> BackgroundMesh:
        """Mesh one field period and return plain arrays after closing Gmsh.

        The cylinder is dimensionless in ``x`` and ``y`` and zeta is in
        radians. Its end disks are related by Gmsh's exact periodic-surface
        constraint. An embedded axis curve preserves axis provenance. This
        method owns the global Gmsh session and refuses to disturb a session
        initialized by other code (DESIGN.md §§8.1–8.2 and 19.2).
        """
        if field.nfp < 1:
            raise ValueError("field.nfp must be positive")
        from . import optional_import

        gmsh = optional_import("gmsh", extra="connectivity")
        if gmsh.isInitialized():
            raise RuntimeError(
                "GmshBackgroundMeshBackend requires ownership of the Gmsh session"
            )

        period = 2.0 * np.pi / field.nfp
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            smallest_size = min(
                value
                for value in (
                    self.config.target_size,
                    self.config.axis_size,
                    self.config.critical_size,
                    self.config.low_gradient_size,
                )
                if value is not None
            )
            gmsh.option.setNumber("Mesh.MeshSizeMin", smallest_size)
            gmsh.option.setNumber("Mesh.MeshSizeMax", self.config.target_size)
            gmsh.model.add("j_connectivity_logical_cylinder")
            volume = gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, period, 1.0)
            axis_start = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
            axis_end = gmsh.model.occ.addPoint(0.0, 0.0, period)
            axis_curve = gmsh.model.occ.addLine(axis_start, axis_end)
            critical_point_tags = []
            for x, y, zeta in self.config.critical_points:
                if x * x + y * y >= 1.0 or not 0.0 < zeta < period:
                    raise ValueError(
                        "critical_points must lie strictly inside the logical cylinder"
                    )
                critical_point_tags.append(gmsh.model.occ.addPoint(x, y, zeta))
            gmsh.model.occ.synchronize()

            outer, lower, upper = _classify_cylinder_surfaces(gmsh, volume, period)
            gmsh.model.mesh.embed(0, [axis_start], 2, lower)
            gmsh.model.mesh.embed(0, [axis_end], 2, upper)
            gmsh.model.mesh.embed(1, [axis_curve], 3, volume)
            if critical_point_tags:
                gmsh.model.mesh.embed(0, critical_point_tags, 3, volume)
            translation = [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                period,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
            gmsh.model.mesh.setPeriodic(2, [upper], [lower], translation)
            _set_gmsh_background_fields(gmsh, self.config, period)
            gmsh.model.mesh.setSizeCallback(
                _gmsh_size_callback(self.config, field, period)
            )
            gmsh.model.mesh.generate(3)
            return _extract_gmsh_mesh(gmsh, field, outer, lower, upper, period)
        finally:
            gmsh.finalize()


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


def _classify_cylinder_surfaces(
    gmsh, volume: int, period: float
) -> tuple[int, int, int]:
    surfaces = gmsh.model.getBoundary([(3, volume)], oriented=False)
    outer = lower = upper = None
    for dimension, tag in surfaces:
        center = gmsh.model.occ.getCenterOfMass(dimension, tag)
        if np.isclose(center[2], 0.0, rtol=0.0, atol=1e-12):
            lower = tag
        elif np.isclose(center[2], period, rtol=0.0, atol=1e-12):
            upper = tag
        else:
            outer = tag
    if outer is None or lower is None or upper is None:
        raise RuntimeError("could not classify logical-cylinder boundary surfaces")
    return outer, lower, upper


def _gmsh_size_callback(
    config: GmshBackgroundMeshConfig, field: BoozerFieldLike, period: float
):
    critical_points = np.asarray(config.critical_points, dtype=np.float64)

    def callback(_dim, _tag, x, y, zeta, _default_size):
        size = config.target_size
        radius = np.hypot(x, y)
        if config.axis_size is not None and radius < config.axis_radius:
            fraction = radius / config.axis_radius
            size = min(
                size,
                config.axis_size + fraction * (config.target_size - config.axis_size),
            )
        if config.critical_size is not None:
            differences = critical_points - np.array([x, y, zeta])
            differences[:, 2] = np.minimum(
                np.abs(differences[:, 2]), period - np.abs(differences[:, 2])
            )
            distances = np.linalg.norm(differences, axis=1)
            nearest = float(np.min(distances))
            if nearest < config.critical_radius:
                fraction = nearest / config.critical_radius
                size = min(
                    size,
                    config.critical_size
                    + fraction * (config.target_size - config.critical_size),
                )
        if config.low_gradient_size is not None:
            s = x * x + y * y
            theta = 0.0 if s == 0.0 else np.arctan2(y, x)
            dB_ds = float(field.dB_ds(s, theta, zeta))
            dB_dtheta = float(field.dB_dtheta(s, theta, zeta))
            dB_dzeta = float(field.dB_dzeta(s, theta, zeta))
            if s == 0.0:
                gradient_squared = dB_dzeta**2
            else:
                dB_dx = 2.0 * x * dB_ds - y * dB_dtheta / s
                dB_dy = 2.0 * y * dB_ds + x * dB_dtheta / s
                gradient_squared = dB_dx**2 + dB_dy**2 + dB_dzeta**2
            if np.sqrt(gradient_squared) <= config.low_gradient_threshold:
                size = min(size, config.low_gradient_size)
        return size

    return callback


def _set_gmsh_background_fields(
    gmsh, config: GmshBackgroundMeshConfig, period: float
) -> None:
    """Install geometric Gmsh fields so isolated refinement regions are sampled."""
    fields = []
    if config.axis_size is not None:
        axis = gmsh.model.mesh.field.add("Cylinder")
        for name, value in (
            ("VIn", config.axis_size),
            ("VOut", config.target_size),
            ("XCenter", 0.0),
            ("YCenter", 0.0),
            ("ZCenter", 0.0),
            ("XAxis", 0.0),
            ("YAxis", 0.0),
            ("ZAxis", period),
            ("Radius", config.axis_radius),
        ):
            gmsh.model.mesh.field.setNumber(axis, name, value)
        fields.append(axis)
    if config.critical_size is not None:
        for x, y, zeta in config.critical_points:
            for periodic_zeta in (zeta - period, zeta, zeta + period):
                ball = gmsh.model.mesh.field.add("Ball")
                for name, value in (
                    ("VIn", config.critical_size),
                    ("VOut", config.target_size),
                    ("XCenter", x),
                    ("YCenter", y),
                    ("ZCenter", periodic_zeta),
                    ("Radius", config.critical_radius),
                ):
                    gmsh.model.mesh.field.setNumber(ball, name, value)
                fields.append(ball)
    if len(fields) == 1:
        gmsh.model.mesh.field.setAsBackgroundMesh(fields[0])
    elif fields:
        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)


def _extract_gmsh_mesh(
    gmsh,
    field: BoozerFieldLike,
    outer_surface: int,
    lower_surface: int,
    upper_surface: int,
    period: float,
) -> BackgroundMesh:
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    points = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
    order = np.argsort(node_tags)
    node_tags = node_tags[order]
    points = points[order]

    element_types, _, element_nodes = gmsh.model.mesh.getElements(3)
    tetrahedron_blocks = []
    for element_type, nodes in zip(element_types, element_nodes):
        _, dimension, order, number_of_nodes, _, _ = (
            gmsh.model.mesh.getElementProperties(int(element_type))
        )
        if dimension != 3 or order != 1 or number_of_nodes != 4:
            raise RuntimeError("Gmsh backend requires first-order tetrahedra")
        tags = np.asarray(nodes, dtype=np.int64).reshape(-1, 4)
        indices = np.searchsorted(node_tags, tags)
        if np.any(node_tags[indices] != tags):
            raise RuntimeError("tetrahedron references an unknown Gmsh node")
        tetrahedron_blocks.append(indices)
    if not tetrahedron_blocks:
        raise RuntimeError("Gmsh generated no tetrahedra")
    tetrahedra = np.vstack(tetrahedron_blocks).astype(np.int64, copy=False)
    negative = _signed_volumes(points, tetrahedra) < 0.0
    tetrahedra[negative, 2:4] = tetrahedra[negative, 3:1:-1]
    if np.any(_signed_volumes(points, tetrahedra) <= 0.0):
        raise RuntimeError("Gmsh generated a degenerate tetrahedron")
    tags = np.zeros(len(points), dtype=np.int64)
    for surface, boundary_tag in (
        (outer_surface, BackgroundMesh.OUTER),
        (lower_surface, BackgroundMesh.ZETA_MIN),
        (upper_surface, BackgroundMesh.ZETA_MAX),
    ):
        surface_nodes, _, _ = gmsh.model.mesh.getNodes(
            2, surface, includeBoundary=True, returnParametricCoord=False
        )
        indices = np.searchsorted(node_tags, np.asarray(surface_nodes, dtype=np.int64))
        tags[indices] |= boundary_tag
    axis = np.linalg.norm(points[:, :2], axis=1) <= 1e-13
    tags[axis] |= BackgroundMesh.AXIS

    _, upper_nodes, lower_nodes, _ = gmsh.model.mesh.getPeriodicNodes(2, upper_surface)
    lower_indices = np.searchsorted(node_tags, np.asarray(lower_nodes, dtype=np.int64))
    upper_indices = np.searchsorted(node_tags, np.asarray(upper_nodes, dtype=np.int64))
    if not len(lower_indices):
        raise RuntimeError("Gmsh did not export periodic end-surface node pairs")
    pairs = np.column_stack((lower_indices, upper_indices)).astype(np.int64, copy=False)
    pair_order = np.argsort(pairs[:, 0])
    pairs = pairs[pair_order]
    tags[pairs[:, 0]] |= BackgroundMesh.ZETA_MIN
    tags[pairs[:, 1]] |= BackgroundMesh.ZETA_MAX
    points[pairs[:, 0], 2] = 0.0
    points[pairs[:, 1], :2] = points[pairs[:, 0], :2]
    points[pairs[:, 1], 2] = period

    s = np.sum(points[:, :2] ** 2, axis=1)
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
