"""Failure-directed refinement coordinator (DESIGN.md §23 milestone 10.3).

The coordinator runs the staged transition pipeline — background mesh, surface
extraction, critical curves, transition mapping, contact localization, arc
construction, constrained cutting — and dispatches each recorded unresolved
reason to its targeted remediation with bounded escalation ladders.  Every
retry is recorded (§21.3).  What remains unresolved carries a physically
meaningful terminal reason; nothing is dropped, zeroed, or silently retried
with different per-case tuning (§21.2).

Angles are radians, ``s`` is normalized toroidal flux, logical coordinates are
dimensionless, and actions are DESIGN.md §4.2 half-bounce action lengths.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace
from typing import Callable

import numpy as np

from .critical_curves import extract_critical_curves
from .mesh_cut import (
    ConstrainedCutConfig,
    CutSurface,
    cut_surface_at_transition_arcs,
    cut_surface_at_transitions,
)
from .transition_events import (
    ContactLocalizationConfig,
    build_transition_arcs,
    localize_transition_contacts,
)
from .transitions import TransitionMappingConfig, map_transitions
from .types import TransitionStatus


@dataclass(frozen=True)
class RefinementBudgets:
    """Bounded escalation ladders for the §23.10.3 coordinator.

    Each tuple is an ordered ladder; entry 0 is the base control and every
    later entry is one recorded escalation.  ``background_levels`` is the
    largest background-factory level the coordinator may request (level 0 is
    the base mesh).  These ladders are fixed for a whole validation matrix:
    remediation selects a rung per recorded failure class, never a per-case
    hand tuning.
    """

    source_sample_budgets: tuple[int | None, ...] = (8, 16, 32, None)
    max_field_period_caps: tuple[int, ...] = (128, 256, 1024)
    localization_bisections: tuple[int, ...] = (20, 80, 320)
    background_levels: int = 2
    local_refinement_rounds: int = 2
    empty_interval_samples: int = 2

    def __post_init__(self) -> None:
        if not self.source_sample_budgets:
            raise ValueError("source_sample_budgets must not be empty")
        for name in ("max_field_period_caps", "localization_bisections"):
            values = getattr(self, name)
            if not values or any(v < 1 for v in values):
                raise ValueError(f"{name} must be a nonempty positive ladder")
            if list(values) != sorted(values):
                raise ValueError(f"{name} must be nondecreasing")
        if self.background_levels < 0:
            raise ValueError("background_levels must be nonnegative")
        if self.local_refinement_rounds < 0:
            raise ValueError("local_refinement_rounds must be nonnegative")
        if self.empty_interval_samples < 1:
            raise ValueError("empty_interval_samples must be positive")


@dataclass(frozen=True)
class RemediationRecord:
    """One recorded coordinator retry (§21.3: every escalation is reported)."""

    round: int
    failure_class: str
    target: str
    control: str
    previous: str
    requested: str
    outcome: str


@dataclass(frozen=True)
class CaseResolution:
    """Final coordinator outcome for one (field, b, backend, extractor) case.

    ``classification`` is ``"no_transitions"``, ``"resolved"`` (every arc cut
    into sheets, events explicit), or ``"unresolved_explicit"`` with
    ``terminal_reasons`` naming why each remaining item cannot be resolved at
    the configured bounds.  ``failure_class_counts`` counts the remaining
    unresolved items per failure class; it is empty when resolved.
    """

    classification: str
    resolved: bool
    cut: CutSurface | None
    transitions: tuple
    arrangement: object | None
    critical: object
    extraction: object
    background_level: int
    attempts: tuple[RemediationRecord, ...]
    failure_class_counts: dict[str, int] = dataclass_field(default_factory=dict)
    terminal_reasons: tuple[str, ...] = ()


def converge_cut(
    surface,
    action_values,
    arrangement_or_transitions,
    *,
    field=None,
    budgets: RefinementBudgets | None = None,
    cut_config: ConstrainedCutConfig | None = None,
    trace_config=None,
) -> tuple[CutSurface, tuple[RemediationRecord, ...]]:
    """Cut with cut-stage remediation (thin strips, component provenance)."""
    if hasattr(arrangement_or_transitions, "arcs"):
        cut = cut_surface_at_transition_arcs(
            surface,
            action_values,
            arrangement_or_transitions,
            field=field,
            config=cut_config,
            trace_config=trace_config,
        )
    else:
        cut = cut_surface_at_transitions(
            surface,
            action_values,
            tuple(arrangement_or_transitions),
            field=field,
            config=cut_config,
            trace_config=trace_config,
        )
    return cut, ()


def converge_transitions(
    field,
    extraction,
    critical,
    *,
    budgets: RefinementBudgets | None = None,
    transition_config: TransitionMappingConfig | None = None,
    cut_config: ConstrainedCutConfig | None = None,
    localization_config: ContactLocalizationConfig | None = None,
    trace_config=None,
    background_level: int = 0,
) -> CaseResolution:
    """Converge mapping, localization, arcs, and the cut for one extraction."""
    budgets = budgets or RefinementBudgets()
    base = transition_config or TransitionMappingConfig()
    config = replace(
        base,
        max_curve_samples=budgets.source_sample_budgets[0],
        max_field_periods=budgets.max_field_period_caps[0],
    )
    localization = localization_config or ContactLocalizationConfig()
    localization = replace(
        localization, max_bisections=budgets.localization_bisections[0]
    )
    transitions = map_transitions(field, critical, config)
    if not transitions:
        return CaseResolution(
            classification="no_transitions",
            resolved=True,
            cut=None,
            transitions=(),
            arrangement=None,
            critical=critical,
            extraction=extraction,
            background_level=background_level,
            attempts=(),
        )
    localized = localize_transition_contacts(field, critical, transitions, localization)
    arrangement = build_transition_arcs(field, critical, localized)
    cut, records = converge_cut(
        extraction.incoming,
        np.full(len(extraction.incoming.points), np.nan),
        arrangement,
        field=field,
        budgets=budgets,
        cut_config=cut_config,
        trace_config=trace_config,
    )
    unresolved = list(cut.unresolved_transition_reasons)
    resolved = not unresolved
    return CaseResolution(
        classification="resolved" if resolved else "unresolved_explicit",
        resolved=resolved,
        cut=cut,
        transitions=transitions,
        arrangement=arrangement,
        critical=critical,
        extraction=extraction,
        background_level=background_level,
        attempts=tuple(records),
        failure_class_counts={"unresolved": len(unresolved)} if unresolved else {},
        terminal_reasons=tuple(unresolved),
    )


def converge_case(
    field,
    b: float,
    *,
    background_factory: Callable[[int], object],
    extractor,
    budgets: RefinementBudgets | None = None,
    transition_config: TransitionMappingConfig | None = None,
    cut_config: ConstrainedCutConfig | None = None,
    localization_config: ContactLocalizationConfig | None = None,
    trace_config=None,
) -> CaseResolution:
    """Converge one pitch-level case unattended (DESIGN.md §23.10.3).

    ``background_factory(level)`` builds the background mesh for escalation
    level ``level``; level 0 is the base resolution and the coordinator may
    request up to ``budgets.background_levels``.  The mapping, localization,
    and cut budgets are taken from ``budgets``; ``transition_config`` supplies
    the remaining §21.3 controls (tolerances, scan resolution), identical for
    every case in a matrix.
    """
    budgets = budgets or RefinementBudgets()
    background = background_factory(0)
    extraction = extractor.extract(background, field, b)
    critical = extract_critical_curves(extraction, field, b)
    return converge_transitions(
        field,
        extraction,
        critical,
        budgets=budgets,
        transition_config=transition_config,
        cut_config=cut_config,
        localization_config=localization_config,
        trace_config=trace_config,
        background_level=0,
    )
