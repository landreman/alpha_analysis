"""Configuration shared by J-connectivity stages (DESIGN.md §§14 and 21)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConnectivityConfig:
    """Global numerical safeguards for the future J-connectivity pipeline.

    ``root_tolerance`` is the absolute tolerance in magnetic-field strength used
    to validate roots of ``B - b``. ``max_field_periods`` bounds a field-line
    scan; reaching it is reported as ``TraceStatus.MAX_PERIODS`` rather than
    silently classifying a trajectory (DESIGN.md §§9.1 and 21.2).
    """

    root_tolerance: float = 1.0e-10
    max_field_periods: int = 128

    def __post_init__(self) -> None:
        if self.root_tolerance <= 0.0:
            raise ValueError("root_tolerance must be positive")
        if self.max_field_periods < 1:
            raise ValueError("max_field_periods must be at least one")
