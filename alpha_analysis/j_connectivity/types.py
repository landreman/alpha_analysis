"""Shared records and explicit result statuses (DESIGN.md §§14 and 21.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto


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
