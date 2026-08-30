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
    MAX_PERIODS = auto()
    UNRESOLVED = auto()
    BUDGET_INSUFFICIENT = auto()


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
class WellTrace:
    """One first-return well trace; values follow DESIGN.md §§9.1 and 9.4.

    Coordinates and lifted ``zeta`` are in radians. ``action_length`` and
    ``bounce_time_length`` store the half-bounce ``A`` and ``K`` of §4.2 in
    length units; they are never the legacy normalized ``J``. Extrema kinds
    are ``-1`` for maxima and ``+1`` for minima; nearly tangent contacts are
    recorded separately for §17.4 diagnostics. Non-regular statuses retain
    ``NaN`` rather than a plausible zero action.
    """

    status: TraceStatus
    b: float
    q_in: FloatArray
    q_out_reduced: FloatArray
    zeta_out_unwrapped: float
    field_period_count: int
    action_length: float
    bounce_time_length: float
    extrema_zeta_unwrapped: FloatArray
    extrema_B: FloatArray
    extrema_kind: IntArray
    tangent_zeta_unwrapped: FloatArray
    tangent_B: FloatArray
    n_internal_maxima: int
    itinerary_hash: np.uint64
    B_residual_in: float
    B_residual_out: float
    quadrature_error_A: float
    quadrature_error_K: float

    def __post_init__(self) -> None:
        q_in = np.asarray(self.q_in, dtype=np.float64)
        q_out = np.asarray(self.q_out_reduced, dtype=np.float64)
        extrema_zeta = np.asarray(self.extrema_zeta_unwrapped, dtype=np.float64)
        extrema_B = np.asarray(self.extrema_B, dtype=np.float64)
        extrema_kind = np.asarray(self.extrema_kind, dtype=np.int64)
        tangent_zeta = np.asarray(self.tangent_zeta_unwrapped, dtype=np.float64)
        tangent_B = np.asarray(self.tangent_B, dtype=np.float64)
        if q_in.shape != (3,) or not np.all(np.isfinite(q_in)):
            raise ValueError("q_in must be a finite reduced (s, theta, zeta) point")
        if q_out.shape != (3,):
            raise ValueError("q_out_reduced must have shape (3,)")
        if extrema_zeta.ndim != 1 or extrema_B.shape != extrema_zeta.shape:
            raise ValueError("extrema positions and values must be equal-length arrays")
        if extrema_kind.shape != extrema_zeta.shape:
            raise ValueError("extrema_kind must have one entry per extremum")
        if np.any((extrema_kind != -1) & (extrema_kind != 1)):
            raise ValueError("extrema_kind entries must be -1 or +1")
        if not np.all(np.isfinite(extrema_zeta)) or not np.all(np.isfinite(extrema_B)):
            raise ValueError("extrema arrays must be finite")
        if tangent_zeta.ndim != 1 or tangent_B.shape != tangent_zeta.shape:
            raise ValueError("tangent positions and values must be equal-length arrays")
        if not np.all(np.isfinite(tangent_zeta)) or not np.all(np.isfinite(tangent_B)):
            raise ValueError("tangent candidate arrays must be finite")
        if self.field_period_count < 0:
            raise ValueError("field_period_count must be nonnegative")
        if self.n_internal_maxima != int(np.count_nonzero(extrema_kind == -1)):
            raise ValueError("n_internal_maxima disagrees with extrema_kind")
        if self.status is TraceStatus.REGULAR:
            regular_scalars = (
                self.zeta_out_unwrapped,
                self.action_length,
                self.bounce_time_length,
                self.B_residual_out,
                self.quadrature_error_A,
                self.quadrature_error_K,
            )
            if not np.all(np.isfinite(regular_scalars)) or not np.all(
                np.isfinite(q_out)
            ):
                raise ValueError("a regular trace must contain finite outputs")
            if self.action_length < 0.0 or self.bounce_time_length < 0.0:
                raise ValueError("regular A and K must be nonnegative")
        for name, values in (
            ("q_in", q_in),
            ("q_out_reduced", q_out),
            ("extrema_zeta_unwrapped", extrema_zeta),
            ("extrema_B", extrema_B),
            ("extrema_kind", extrema_kind),
            ("tangent_zeta_unwrapped", tangent_zeta),
            ("tangent_B", tangent_B),
        ):
            object.__setattr__(self, name, values)
        object.__setattr__(self, "itinerary_hash", np.uint64(self.itinerary_hash))


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
