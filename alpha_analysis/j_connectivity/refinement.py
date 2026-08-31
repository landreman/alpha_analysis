"""Failure-directed refinement coordinator (DESIGN.md §23 milestone 10.3).

The coordinator runs the staged transition pipeline — background mesh, surface
extraction, critical curves, transition mapping, contact localization, arc
construction, constrained cutting — and dispatches each recorded unresolved
reason to its targeted remediation with bounded escalation ladders:

- per-sample field-period-cap escalation for ``MAX_PERIODS`` traces;
- transition-mapping work-budget escalation reusing every cached trace;
- contact-localization bisection-budget escalation for unlocalized brackets;
- interval sampling for event intervals holding no regular arc sample;
- local (not global) surface refinement near companion curves whose
  ``T``-to-``EDGE`` strip width fails the resolution requirement;
- background-mesh escalation for unresolved extractions and critical curves
  (dispatched by :func:`converge_case` through its background factory).

Component-provenance enforcement for off-surface projections lives inside the
constrained cut itself (:mod:`mesh_cut`), where every location and insertion
is restricted to the curve's own surface component.

Every retry is recorded (§21.3).  What remains unresolved carries a
physically meaningful terminal reason with per-failure-class counts; nothing
is dropped, zeroed, or retried with per-case hand tuning (§21.2).

Angles are radians, ``s`` is normalized toroidal flux, logical coordinates
are dimensionless, and actions are DESIGN.md §4.2 half-bounce action lengths.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field as dataclass_field, replace
from typing import Callable

import numpy as np

from .critical_curves import CriticalCurveStatus, CriticalKind, extract_critical_curves
from .mesh_cut import (
    ConstrainedCutConfig,
    CutSurface,
    cut_surface_at_transition_arcs,
    cut_surface_at_transitions,
)
from .surface_refine import LocalRefinementConfig, refine_surface_near_curves
from .transition_events import (
    ContactLocalizationConfig,
    build_transition_arcs,
    localize_transition_contacts,
)
from .transitions import (
    TransitionMappingConfig,
    _map_polyline_samples,
    map_transitions,
)
from .types import SurfaceStatus, TransitionStatus

_CAP_TOKENS = ("backward_max_periods", "forward_max_periods")


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


def classify_failure_reason(reason: str) -> str:
    """Map one recorded unresolved reason to its §23.10.3 failure class."""
    if any(token in reason for token in _CAP_TOKENS) or "cap ceiling" in reason:
        return "max_periods"
    if "T-to-EDGE strip width" in reason or "doubles back" in reason:
        return "thin_strip"
    if "projected locally to B=b" in reason:
        return "local_projection"
    if "disconnected" in reason and "component" in reason:
        return "off_component"
    if "from the nearest surface triangle" in reason:
        return "off_component"
    if "collapses onto the EDGE boundary" in reason or "from the EDGE" in reason:
        return "background_geometry"
    if "does not separate the surface" in reason:
        return "background_geometry"
    if "below the authoritative vertex resolution" in reason:
        return "background_geometry"
    if "incident to more than one sheet" in reason:
        return "background_geometry"
    if "sampling budget" in reason:
        return "source_budget"
    if "localization budget" in reason or "geometry remains unresolved" in reason:
        return "contact_localization"
    if "not certified" in reason and "geometry" in reason:
        return "event_geometry"
    if "no regular arc sample" in reason:
        return "empty_arc_interval"
    if (
        "source_classification" in reason
        or "segment_classification" in reason
        or "source_residual" in reason
        or "degenerate" in reason.lower()
    ):
        return "unresolved_critical"
    return "cut_conflict"


def _gamma_max_polylines(critical):
    return [p for p in critical.polylines if p.kind is CriticalKind.GAMMA_MAX]


def _capped_cache_indices(cache):
    return sorted(
        index
        for index, sample in cache.items()
        if any(token in sample.sample_failure_reason[0] for token in _CAP_TOKENS)
    )


def _mentions_cap(*chunks) -> bool:
    return any(any(token in text for token in _CAP_TOKENS) for text in chunks if text)


def _is_strip_reason(reason: str) -> bool:
    return "T-to-EDGE strip width" in reason or "doubles back" in reason


def _is_projection_reason(reason: str) -> bool:
    return "projected locally to B=b" in reason


def _is_locally_refinable_reason(reason: str) -> bool:
    """Cut failures whose recorded remedy is local surface refinement.

    A strip or strand separation below ``min_transition_strip_edge_ratio``
    times the local edge scale passes once the local edges halve, and a
    constrained vertex whose bounded ``B=b`` projection failed converges
    because the PL surface approaches the true level quadratically while the
    displacement allowance shrinks only linearly.
    """
    return _is_strip_reason(reason) or _is_projection_reason(reason)


def _unwrap_near(point, origin, period):
    point = np.asarray(point, dtype=float).copy()
    point[2] -= period * np.round((point[2] - origin[2]) / period)
    return point


def _edge_crosses_curves(first, second, curves, period):
    """Whether the segment crosses any companion polyline segment.

    Both live near the same surface, so a genuine crossing shows as a
    closest approach far below the segment scale at interior parameters of
    both segments.  A segment running alongside the curve (as thin-strip
    edges do) is near-parallel and never spans the coming discontinuity, so
    it is not a crossing.
    """
    first = np.asarray(first, dtype=float)
    direction = _unwrap_near(second, first, period) - first
    length = float(np.linalg.norm(direction))
    if length == 0.0:
        return False
    for curve in curves:
        curve = np.asarray(curve, dtype=float)
        for a, b in zip(curve[:-1], curve[1:]):
            a = _unwrap_near(a, first, period)
            b = _unwrap_near(b, a, period)
            other = b - a
            other_length = float(np.linalg.norm(other))
            if other_length == 0.0:
                continue
            normal = np.cross(direction, other)
            normal_squared = float(np.dot(normal, normal))
            if normal_squared <= (1.0e-9 * length * other_length) ** 2:
                continue
            offset = a - first
            s = float(np.dot(np.cross(offset, other), normal) / normal_squared)
            t = float(np.dot(np.cross(offset, direction), normal) / normal_squared)
            if not (-1.0e-9 <= s <= 1.0 + 1.0e-9 and -1.0e-9 <= t <= 1.0 + 1.0e-9):
                continue
            gap = float(np.linalg.norm(first + s * direction - (a + t * other)))
            if gap <= 0.35 * min(length, other_length):
                return True
    return False


def _perpendicular_curve_offset(point, curves, period):
    """Perpendicular offset of ``point`` from its nearest companion segment."""
    point = np.asarray(point, dtype=float)
    best = None
    for curve in curves:
        curve = np.asarray(curve, dtype=float)
        for a, b in zip(curve[:-1], curve[1:]):
            a = _unwrap_near(a, point, period)
            b = _unwrap_near(b, a, period)
            segment = b - a
            length_squared = float(np.dot(segment, segment))
            if length_squared == 0.0:
                continue
            fraction = float(
                np.clip(np.dot(point - a, segment) / length_squared, 0.0, 1.0)
            )
            perpendicular = point - (a + fraction * segment)
            distance = float(np.linalg.norm(perpendicular))
            if best is None or distance < best[0]:
                best = (distance, perpendicular)
    return None if best is None else best[1]


def _same_side_action(midpoint, endpoints, actions, curves, period):
    """Clamp a crossing-edge midpoint to its same-side endpoint's action.

    An edge that crosses the coming cut may not interpolate across the
    discontinuity (§21.2), but its midpoint lies definitely on one side of
    the companion curve: it takes the finite action of the endpoint whose
    perpendicular offset from the nearest companion segment points the same
    way — a same-branch clamp with the ADR 0004 precedent, never a
    parent/child blend.  Ambiguity stays ``NaN``.
    """
    offset_mid = _perpendicular_curve_offset(midpoint, curves, period)
    if offset_mid is None or float(np.linalg.norm(offset_mid)) <= 1.0e-12:
        return np.nan
    for endpoint, action in zip(endpoints, actions):
        if not np.isfinite(action):
            continue
        offset_end = _perpendicular_curve_offset(endpoint, curves, period)
        if offset_end is None or float(np.linalg.norm(offset_end)) <= 1.0e-12:
            continue
        if float(np.dot(offset_mid, offset_end)) > 0.0:
            return float(action)
    return np.nan


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
    """Cut with bounded local surface refinement near failing thin strips.

    Transitions rejected because the companion-to-``EDGE`` strip width (or a
    double-back strand separation) is below ``min_transition_strip_edge_ratio``
    times the local edge scale are retried after halving the triangles near
    their companion curves — local refinement only, never a global remesh.
    Actions on refined vertices are interpolated only when both parents agree;
    anything else stays explicit ``NaN`` (§21.2), which the cut's probe-based
    side assignment and unresolved-flux accounting already handle.  A strip
    still failing at the refinement bound keeps its ports and gains the
    explicit terminal reason; the check itself is never loosened.
    """
    budgets = budgets or RefinementBudgets()
    records: list[RemediationRecord] = []
    is_arrangement = hasattr(arrangement_or_transitions, "arcs")

    def run(current_surface, current_action):
        if is_arrangement:
            return cut_surface_at_transition_arcs(
                current_surface,
                current_action,
                arrangement_or_transitions,
                field=field,
                config=cut_config,
                trace_config=trace_config,
            )
        return cut_surface_at_transitions(
            current_surface,
            current_action,
            tuple(arrangement_or_transitions),
            field=field,
            config=cut_config,
            trace_config=trace_config,
        )

    if is_arrangement:
        curves = {
            arc.curve.transition_id: next(
                port for port in arc.curve.ports if port.role == "parent"
            ).points
            for arc in arrangement_or_transitions.arcs
        }
    else:
        curves = {
            transition.transition_id: next(
                port for port in transition.ports if port.role == "parent"
            ).points
            for transition in arrangement_or_transitions
        }

    working_surface = surface
    working_action = np.asarray(action_values, dtype=np.float64)
    rounds_used = 0
    while True:
        cut = run(working_surface, working_action)
        strip_ids = [
            int(transition_id)
            for transition_id, reason in zip(
                cut.unresolved_transition_ids, cut.unresolved_transition_reasons
            )
            if _is_locally_refinable_reason(reason)
        ]
        if not strip_ids or rounds_used >= budgets.local_refinement_rounds:
            break
        refined, report = refine_surface_near_curves(
            working_surface,
            tuple(curves[transition_id] for transition_id in strip_ids),
            field,
            LocalRefinementConfig(),
        )
        if report.edges_split == 0:
            break
        parents = report.new_vertex_edges
        first = working_action[parents[:, 0]]
        second = working_action[parents[:, 1]]
        # A midpoint of a same-branch PL edge takes the PL value; an edge
        # that crosses a companion polyline spans the coming action
        # discontinuity, so its midpoint is clamped to its same-side
        # endpoint's value instead of a parent/child blend, and stays
        # explicit NaN when even the side is ambiguous (§21.2).
        period = float(working_surface.period)
        all_curves = tuple(curves.values())
        new_action = np.empty(len(parents))
        new_points = refined.points[len(working_surface.points) :]
        for row, edge in enumerate(parents):
            endpoints = (
                working_surface.points[int(edge[0])],
                working_surface.points[int(edge[1])],
            )
            actions = (float(first[row]), float(second[row]))
            if not (np.isfinite(actions[0]) and np.isfinite(actions[1])):
                new_action[row] = np.nan
            elif _edge_crosses_curves(*endpoints, all_curves, period):
                new_action[row] = _same_side_action(
                    new_points[row], endpoints, actions, all_curves, period
                )
            else:
                new_action[row] = 0.5 * (actions[0] + actions[1])
        working_action = np.concatenate((working_action, new_action))
        working_surface = refined
        rounds_used += 1
        triggered_by_strip = any(
            _is_strip_reason(reason)
            for reason in cut.unresolved_transition_reasons
            if _is_locally_refinable_reason(reason)
        )
        records.append(
            RemediationRecord(
                round=rounds_used,
                failure_class=(
                    "thin_strip" if triggered_by_strip else ("local_projection")
                ),
                target=f"transitions {strip_ids}",
                control="local_refinement_round",
                previous=str(rounds_used - 1),
                requested=str(rounds_used),
                outcome=(
                    f"split {report.edges_split} edges near {len(strip_ids)} "
                    f"companion curves ({report.projection_rejections} "
                    "projection rejections)"
                ),
            )
        )
    if any(
        _is_locally_refinable_reason(reason)
        for reason in cut.unresolved_transition_reasons
    ):
        used = f"({rounds_used} of {budgets.local_refinement_rounds} rounds used)"
        annotated = tuple(
            (
                reason
                + "; genuinely unrepresentable strip at the local refinement "
                + f"bound {used}"
                if _is_strip_reason(reason)
                else (
                    reason
                    + "; local projection still failing at the local "
                    + f"refinement bound {used}"
                    if _is_projection_reason(reason)
                    else reason
                )
            )
            for reason in cut.unresolved_transition_reasons
        )
        cut = replace(cut, unresolved_transition_reasons=annotated)
    return cut, tuple(records)


def _remap_capped_samples(
    field, critical, caches, config, records, round_number, previous_cap
):
    """Retrace only the cache entries whose failure is the field-period cap.

    A regular trace is cap-independent — the cap only binds when it is
    exceeded — so every other cached sample stays exact under the raised cap
    and is reused, which is what makes per-sample escalation affordable.
    """
    maxima = _gamma_max_polylines(critical)
    retraced = False
    for transition_id, cache in enumerate(caches):
        indices = _capped_cache_indices(cache)
        if not indices:
            continue
        for index in indices:
            cache[index] = _map_polyline_samples(
                field,
                critical,
                maxima[transition_id],
                transition_id,
                config,
                np.asarray([index], dtype=np.int64),
            )
        still = _capped_cache_indices(cache)
        records.append(
            RemediationRecord(
                round=round_number,
                failure_class="max_periods",
                target=f"transition {transition_id} source vertices {indices}",
                control="max_field_periods",
                previous=str(previous_cap),
                requested=str(config.max_field_periods),
                outcome=(
                    "retraced; all resolved"
                    if not still
                    else f"retraced; {len(still)} still capped"
                ),
            )
        )
        retraced = True
    return retraced


def _empty_interval_extra_samples(
    field, critical, localized, arrangement, extra_samples, attempts, budgets
):
    """Map extra true-curve samples into event intervals with no regular one."""
    maxima = _gamma_max_polylines(critical)
    added = []
    for arc in arrangement.arcs:
        if arc.unresolved_reason != "event interval has no regular arc sample":
            continue
        if arc.source_interval is None:
            continue
        low, high = arc.source_interval
        key = (arc.source_transition_id, round(float(low), 12), round(float(high), 12))
        if attempts.get(key, 0) >= budgets.empty_interval_samples:
            continue
        polyline = maxima[arc.source_transition_id]
        source = localized.transitions[arc.source_transition_id]
        length = float(polyline.total_length)
        interior = [
            (index, u_value + cycle * length)
            for index, u_value in enumerate(polyline.u)
            for cycle in ((0, 1) if polyline.closed else (0,))
            if low < u_value + cycle * length < high
        ]
        taken = {
            round(float(sample.u[0]), 12)
            for sample in extra_samples.get(arc.source_transition_id, ())
        }
        interior = [
            (index, u_value)
            for index, u_value in interior
            if round(float(u_value), 12) not in taken
        ]
        if not interior:
            continue
        middle = 0.5 * (low + high)
        index, u_value = min(interior, key=lambda item: abs(item[1] - middle))
        sample = _map_polyline_samples(
            field,
            critical,
            polyline,
            arc.source_transition_id,
            source.controls,
            np.asarray([index], dtype=np.int64),
        )
        sample = replace(sample, u=np.asarray([u_value], dtype=float))
        extra_samples.setdefault(arc.source_transition_id, []).append(sample)
        attempts[key] = attempts.get(key, 0) + 1
        added.append(
            (
                arc.source_transition_id,
                (float(low), float(high)),
                float(u_value),
                sample.sample_status[0].name,
            )
        )
    return added


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
    """Converge mapping, localization, arcs, and the cut for one extraction.

    Diagnoses the machine-readable unresolved reasons after each stage and
    applies exactly one bounded escalation at a time — highest-priority
    first: per-sample period caps, then contact-localization bisections, then
    the source sampling budget, then interval samples for empty event
    intervals — recording every retry.  Trace caches persist across
    escalations so a raised budget reuses every existing trace; a regular
    trace is cap-independent, so raising the cap retraces only capped
    samples.  Terminal states keep their ports and reasons (§21.2).
    """
    budgets = budgets or RefinementBudgets()
    base = transition_config or TransitionMappingConfig()
    localization_base = localization_config or ContactLocalizationConfig()
    source_rung = 0
    cap_rung = 0
    localization_rung = 0
    records: list[RemediationRecord] = []
    maxima = _gamma_max_polylines(critical)
    caches = [dict() for _ in maxima]
    extra_samples: dict[int, list] = {}
    interval_attempts: dict[tuple, int] = {}

    if not maxima:
        if critical.status is CriticalCurveStatus.REGULAR:
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
        reason = (
            f"critical-curve status is {critical.status.name} with no mappable "
            "GAMMA_MAX polyline; the marginal classification is unresolved, "
            "not an absence of transitions"
        )
        return CaseResolution(
            classification="unresolved_explicit",
            resolved=False,
            cut=None,
            transitions=(),
            arrangement=None,
            critical=critical,
            extraction=extraction,
            background_level=background_level,
            attempts=(),
            failure_class_counts={"unresolved_critical": 1},
            terminal_reasons=(reason,),
        )

    def config():
        return replace(
            base,
            max_curve_samples=budgets.source_sample_budgets[source_rung],
            max_field_periods=budgets.max_field_period_caps[cap_rung],
        )

    def localization():
        return replace(
            localization_base,
            max_bisections=budgets.localization_bisections[localization_rung],
        )

    transitions = map_transitions(field, critical, config(), _sample_caches=caches)
    localized = None
    arrangement = None
    round_number = 0
    hard_limit = (
        len(budgets.source_sample_budgets)
        + len(budgets.max_field_period_caps)
        + len(budgets.localization_bisections)
        + budgets.empty_interval_samples * 8
        + 4
    )
    while round_number < hard_limit:
        round_number += 1
        # 1. Per-sample field-period-cap escalation: mapped samples first.
        if cap_rung + 1 < len(budgets.max_field_period_caps) and any(
            _capped_cache_indices(cache) for cache in caches
        ):
            previous = budgets.max_field_period_caps[cap_rung]
            cap_rung += 1
            escalated = config()
            _remap_capped_samples(
                field, critical, caches, escalated, records, round_number, previous
            )
            transitions = map_transitions(
                field, critical, escalated, _sample_caches=caches
            )
            localized = None
            arrangement = None
            continue
        if localized is None:
            localized = localize_transition_contacts(
                field, critical, transitions, localization()
            )
            arrangement = None
        if arrangement is None:
            arrangement = build_transition_arcs(
                field,
                critical,
                localized,
                {key: tuple(value) for key, value in extra_samples.items()},
            )
        # 2. A cap binding inside localization probes or arc certification.
        cap_in_probes = any(
            _mentions_cap(occurrence.reason)
            for event in localized.events
            for occurrence in event.occurrences
        ) or any(
            _mentions_cap(arc.unresolved_reason, *arc.curve.sample_failure_reason)
            for arc in arrangement.arcs
        )
        if cap_in_probes and cap_rung + 1 < len(budgets.max_field_period_caps):
            previous = budgets.max_field_period_caps[cap_rung]
            cap_rung += 1
            records.append(
                RemediationRecord(
                    round=round_number,
                    failure_class="max_periods",
                    target="localization probes and arc certification traces",
                    control="max_field_periods",
                    previous=str(previous),
                    requested=str(budgets.max_field_period_caps[cap_rung]),
                    outcome="reran mapping, localization, and arcs at the new cap",
                )
            )
            _remap_capped_samples(
                field, critical, caches, config(), records, round_number, previous
            )
            transitions = map_transitions(
                field, critical, config(), _sample_caches=caches
            )
            localized = None
            arrangement = None
            continue
        # 3. Contact-localization bisection budget.
        unlocalized = [
            (occurrence.source_transition_id, occurrence.source_sample_pair)
            for event in localized.events
            for occurrence in event.occurrences
            if not occurrence.localized
            and "contact localization budget exhausted" in occurrence.reason
        ]
        if unlocalized and localization_rung + 1 < len(budgets.localization_bisections):
            previous = budgets.localization_bisections[localization_rung]
            localization_rung += 1
            records.append(
                RemediationRecord(
                    round=round_number,
                    failure_class="contact_localization",
                    target=f"contact brackets {sorted(set(unlocalized))}",
                    control="max_bisections",
                    previous=str(previous),
                    requested=str(budgets.localization_bisections[localization_rung]),
                    outcome="reran contact localization",
                )
            )
            localized = None
            arrangement = None
            continue
        # 4. Source sampling work budget, reusing all cached traces.
        # An arc whose certification discovered a new nongeneric sample or an
        # unexplained count change needs more retained source vertices: after
        # remapping, the change becomes a source-level contact bracket that
        # localization can turn into an explicit event.
        needs_budget = [
            arc.curve.transition_id
            for arc in arrangement.arcs
            if arc.curve.status is TransitionStatus.BUDGET_INSUFFICIENT
            or "sampling budget exhausted" in (arc.unresolved_reason or "")
            or arc.unresolved_reason
            == "unexplained interior-maximum count change within arc"
            or arc.unresolved_reason == "additional nongeneric or failed sample in arc"
            # A nongeneric sample adjacent to an event (or an interval with
            # no regular sample) also needs more retained source vertices, so
            # the nongeneric structure becomes source-level brackets and
            # sampled events that localization can make explicit.
            or arc.unresolved_reason == "event has no regular one-sided sample"
            or arc.unresolved_reason == "source arc contains a failed trace"
        ] + [
            transition.transition_id
            for transition in transitions
            if transition.status is TransitionStatus.BUDGET_INSUFFICIENT
        ]
        if needs_budget and source_rung + 1 < len(budgets.source_sample_budgets):
            previous = budgets.source_sample_budgets[source_rung]
            source_rung += 1
            records.append(
                RemediationRecord(
                    round=round_number,
                    failure_class="source_budget",
                    target=f"transitions or arcs {sorted(set(needs_budget))}",
                    control="max_curve_samples",
                    previous=str(previous),
                    requested=str(budgets.source_sample_budgets[source_rung]),
                    outcome="remapped with cached traces",
                )
            )
            transitions = map_transitions(
                field, critical, config(), _sample_caches=caches
            )
            localized = None
            arrangement = None
            continue
        # 5. Event intervals holding no regular sample.
        added = _empty_interval_extra_samples(
            field,
            critical,
            localized,
            arrangement,
            extra_samples,
            interval_attempts,
            budgets,
        )
        if added:
            records.append(
                RemediationRecord(
                    round=round_number,
                    failure_class="empty_arc_interval",
                    target=f"event intervals {[item[:2] for item in added]}",
                    control="interval_samples",
                    previous="0",
                    requested=str(len(added)),
                    outcome=f"mapped interval samples with statuses "
                    f"{[item[3] for item in added]}",
                )
            )
            arrangement = None
            continue
        break

    cut, cut_records = converge_cut(
        extraction.incoming,
        np.full(len(extraction.incoming.points), np.nan),
        arrangement,
        field=field,
        budgets=budgets,
        cut_config=cut_config,
        trace_config=trace_config,
    )
    records.extend(cut_records)
    terminal = []
    for reason in cut.unresolved_transition_reasons:
        if _mentions_cap(reason):
            reason = (
                reason + f" (field-period cap ceiling "
                f"{budgets.max_field_period_caps[cap_rung]} reached)"
            )
        terminal.append(reason)
    counts = Counter(classify_failure_reason(reason) for reason in terminal)
    resolved = not terminal
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
        failure_class_counts=dict(counts),
        terminal_reasons=tuple(terminal),
    )


_BACKGROUND_CLASSES = frozenset(
    {"unresolved_critical", "background_geometry", "off_component"}
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
    request up to ``budgets.background_levels``.  Background escalation is
    dispatched for an ``UNRESOLVED`` extraction (ADR 0001 sheet-bridging
    splits), an ``UNRESOLVED`` critical-curve classification, and for
    residual failures whose recorded remedy is a finer background
    (source-classification failures, a companion curve the surface cannot
    hold, an open endpoint that cannot reach ``EDGE``).  All other budgets
    come from ``budgets``; ``transition_config`` supplies the remaining
    §21.3 controls, identical for every case in a matrix.
    """
    budgets = budgets or RefinementBudgets()
    attempts: list[RemediationRecord] = []
    level = 0
    resolution = None
    while True:
        background = background_factory(level)
        extraction = extractor.extract(background, field, b)
        critical = extract_critical_curves(extraction, field, b)
        stage_reason = None
        if extraction.status is not SurfaceStatus.REGULAR:
            stage_reason = (
                "unresolved_extraction",
                f"surface extraction is {extraction.status.name} with "
                f"{extraction.n_unresolved_splits} unresolved sheet-bridging "
                "splits (ADR 0001); the level-set topology needs background "
                "refinement before any cut",
            )
        elif critical.status is CriticalCurveStatus.UNRESOLVED:
            stage_reason = (
                "unresolved_critical",
                "critical-curve classification is UNRESOLVED; ambiguous "
                "segments need background refinement",
            )
        if stage_reason is None:
            resolution = converge_transitions(
                field,
                extraction,
                critical,
                budgets=budgets,
                transition_config=transition_config,
                cut_config=cut_config,
                localization_config=localization_config,
                trace_config=trace_config,
                background_level=level,
            )
            resolution = replace(
                resolution, attempts=tuple(attempts) + resolution.attempts
            )
            trigger = sorted(set(resolution.failure_class_counts) & _BACKGROUND_CLASSES)
            if resolution.resolved or not trigger:
                return resolution
            trigger_class = trigger[0]
            trigger_target = "; ".join(resolution.terminal_reasons)[:200]
        else:
            trigger_class, trigger_target = stage_reason
            resolution = CaseResolution(
                classification="unresolved_explicit",
                resolved=False,
                cut=None,
                transitions=(),
                arrangement=None,
                critical=critical,
                extraction=extraction,
                background_level=level,
                attempts=tuple(attempts),
                failure_class_counts={trigger_class: 1},
                terminal_reasons=(trigger_target,),
            )
        if level >= budgets.background_levels:
            return resolution
        attempts.append(
            RemediationRecord(
                round=level + 1,
                failure_class=trigger_class,
                target=trigger_target,
                control="background_level",
                previous=str(level),
                requested=str(level + 1),
                outcome="rebuilt background, extraction, and critical curves",
            )
        )
        level += 1
