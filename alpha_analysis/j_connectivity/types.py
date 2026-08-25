"""Shared records and explicit result statuses (DESIGN.md §§14 and 21.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class TraceStatus(Enum):
    """Outcome of one well trace; non-regular outcomes remain explicit."""

    REGULAR = auto()
    NO_WELL = auto()
    MAX_PERIODS = auto()
    ROOT_FAILURE = auto()
    QUADRATURE_FAILURE = auto()
    TANGENT_OR_TRANSITION = auto()
    AXIS_UNRESOLVED = auto()
    DEGENERATE = auto()


class SurfaceStatus(Enum):
    """Outcome of a level-surface extraction."""

    REGULAR = auto()
    ROOT_FAILURE = auto()
    PERIODIC_MISMATCH = auto()
    DEGENERATE = auto()
    UNRESOLVED = auto()


class TransitionStatus(Enum):
    """Outcome of mapping a split or merge transition."""

    REGULAR = auto()
    MULTIWAY = auto()
    TANGENT = auto()
    MATCH_FAILURE = auto()
    UNRESOLVED = auto()


class FloodFillStatus(Enum):
    """Outcome of bounded contour reachability classification."""

    RESOLVED = auto()
    UNRESOLVED = auto()
    INVALID_TRANSITION = auto()


class QuadratureStatus(Enum):
    """Outcome of a surface or outer numerical quadrature."""

    CONVERGED = auto()
    MAX_REFINEMENT = auto()
    UNRESOLVED = auto()


@dataclass(frozen=True)
class RunMetadata:
    """Reproducibility context saved with a J-connectivity result (§13.4).

    Paths and hashes identify the equilibrium input; ``created_at`` is UTC so
    serialized results have an unambiguous creation time.
    """

    equilibrium_path: str | None = None
    equilibrium_hash: str | None = None
    code_commit: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class BackgroundMesh:
    """Authoritative logical-cylinder tetrahedral mesh (DESIGN.md §§8.1–8.2).

    Points are dimensionless ``(x, y, zeta)`` with ``zeta`` in radians.
    ``boundary_tags`` is a point-located bit mask. Magnetic-field arrays are
    point-located and retain the field implementation's units. The two end
    disks remain explicit and are related by lower/upper periodic node pairs.
    """

    INTERIOR: ClassVar[int] = 0
    AXIS: ClassVar[int] = 1
    OUTER: ClassVar[int] = 2
    ZETA_MIN: ClassVar[int] = 4
    ZETA_MAX: ClassVar[int] = 8

    points: FloatArray
    tetrahedra: IntArray
    periodic_node_pairs: IntArray
    boundary_tags: IntArray
    B: FloatArray
    D_B: FloatArray
    D2_B: FloatArray

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        tetrahedra = np.asarray(self.tetrahedra, dtype=np.int64)
        pairs = np.asarray(self.periodic_node_pairs, dtype=np.int64)
        tags = np.asarray(self.boundary_tags, dtype=np.int64)
        point_data = tuple(
            np.asarray(values, dtype=np.float64)
            for values in (self.B, self.D_B, self.D2_B)
        )
        n_points = points.shape[0] if points.ndim == 2 else 0
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("points must be a finite array with shape (n_points, 3)")
        if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
            raise ValueError("tetrahedra must have shape (n_tetrahedra, 4)")
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("periodic_node_pairs must have shape (n_pairs, 2)")
        if tags.shape != (n_points,):
            raise ValueError("boundary_tags must have one entry per point")
        if any(values.shape != (n_points,) for values in point_data):
            raise ValueError("B, D_B, and D2_B must have one value per point")
        if not all(np.all(np.isfinite(values)) for values in point_data):
            raise ValueError("background field arrays must be finite")
        if tetrahedra.size and (tetrahedra.min() < 0 or tetrahedra.max() >= n_points):
            raise ValueError("tetrahedron point index is outside the mesh")
        if pairs.size and (pairs.min() < 0 or pairs.max() >= n_points):
            raise ValueError("periodic point index is outside the mesh")
        for name, values in (
            ("points", points),
            ("tetrahedra", tetrahedra),
            ("periodic_node_pairs", pairs),
            ("boundary_tags", tags),
            ("B", point_data[0]),
            ("D_B", point_data[1]),
            ("D2_B", point_data[2]),
        ):
            object.__setattr__(self, name, values)

    def to_pyvista(self):
        """Return an optional PyVista view while NumPy stays authoritative (§17.1)."""
        from . import optional_import

        pyvista = optional_import("pyvista", extra="connectivity")
        cells = np.column_stack(
            (np.full(len(self.tetrahedra), 4, dtype=np.int64), self.tetrahedra)
        ).ravel()
        cell_types = np.full(
            len(self.tetrahedra), int(pyvista.CellType.TETRA), dtype=np.uint8
        )
        grid = pyvista.UnstructuredGrid(cells, cell_types, self.points)
        grid.point_data["B [field units]"] = self.B
        grid.point_data["D_parallel B [field units/rad]"] = self.D_B
        grid.point_data["D_parallel^2 B [field units/rad^2]"] = self.D2_B
        grid.point_data["boundary tag [bit mask]"] = self.boundary_tags
        return grid
