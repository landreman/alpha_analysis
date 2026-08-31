"""Failure-directed refinement coordinator tests (DESIGN.md §23 milestone 10.3).

Every remediation class the coordinator dispatches is exercised on an analytic
synthetic field or a hand-built surface whose correct outcome is derived
independently of the code under test: per-sample period-cap escalation,
source-budget escalation, contact-localization escalation, below-``b`` fold
certification, local surface refinement near thin strips, and
component-provenance enforcement.  Bounded termination keeps §21.2: an
unresolvable case ends with explicit reasons and retained ports, never a cut
by default or a silently absorbed failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from dataclasses import replace
from scipy.optimize import brentq

from alpha_analysis.j_connectivity import (
    BackgroundMeshConfig,
    ConstrainedCutConfig,
    MarchingTetrahedraExtractor,
    StructuredPrismMeshBackend,
    SurfaceMesh,
    TransitionMappingConfig,
    TransitionStatus,
    extract_critical_curves,
    map_transitions,
)
from alpha_analysis.j_connectivity.refinement import (
    CaseResolution,
    RefinementBudgets,
    converge_case,
    converge_cut,
    converge_transitions,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField
from test_mesh_cut import _port, _surface_and_action, _transition


def _fold_field():
    """B = 2 + cos z + (0.35+0.3 s) cos 2z + (0.05+0.03 cos th) sin 5z.

    At ``b=1.5`` the marginal maximum traces one closed ``GAMMA_MAX`` circle.
    The harmonic-5 wiggle creates one interior maximum--minimum pair whose
    fold (annihilation) height stays about ``0.1`` below ``b``: the parent
    well's interior-maximum count changes across two fold events while every
    port action stays continuous (ADR 0003's second, below-``b`` case).
    """
    C0, A0, A1 = 2.0, 0.35, 0.3
    e0, e1, phi0 = 0.05, 0.03, np.pi / 2
    return SyntheticFourierField(
        nfp=1,
        m=np.array([0, 0, 0, 0, 1, 1]),
        n=np.array([0, 1, 2, 5, 5, -5]),
        cosine_coefficients=np.array(
            [
                [C0, 0],
                [1.0, 0],
                [A0, A1],
                [e0 * np.cos(phi0), 0],
                [e1 / 2 * np.cos(phi0), 0],
                [e1 / 2 * np.cos(phi0), 0],
            ]
        ),
        sine_coefficients=np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [-e0 * np.sin(phi0), 0],
                [-e1 / 2 * np.sin(phi0), 0],
                [e1 / 2 * np.sin(phi0), 0],
            ]
        ),
        iota_coefficients=np.array([0.0]),
        G_coefficients=np.array([3.0]),
        I_coefficients=np.array([0.0]),
    )


FOLD_B = 1.5


def _fold_interior_maximum_count(field, theta):
    """Count sub-``b`` interior maxima of the marginal well at ``theta``.

    Derived from the field alone: solve the marginal flux surface s*(theta)
    with plain 1-D root finds, then count derivative sign changes strictly
    inside the well.  No production transition code is used.
    """

    def profile_extrema(s):
        zg = np.linspace(0.0, 2.0 * np.pi, 4001)
        sign = np.sign(field.D_B(s, theta, zg))
        flips = np.flatnonzero(sign[:-1] * sign[1:] < 0)
        out = []
        for j in flips:
            z = brentq(
                lambda zz: float(field.D_B(s, theta, zz)),
                zg[j],
                zg[j + 1],
                xtol=1e-13,
            )
            out.append((z, float(field.B(s, theta, z))))
        return out

    def marginal_gap(s):
        heights = np.array([h for _, h in profile_extrema(s)])
        below_main = heights[heights < heights.max() - 0.5]
        return below_main.max() - FOLD_B

    s_star = brentq(marginal_gap, 0.05, 0.95, xtol=1e-12)
    extrema = profile_extrema(s_star)
    heights = np.array([h for _, h in extrema])
    main = heights.max()
    marginal_z = next(z for z, h in extrema if abs(h - FOLD_B) < 1e-9)
    count = 0
    for z, h in extrema:
        if h >= main - 0.5 or abs(z - marginal_z) < 1e-6 or h >= FOLD_B:
            continue
        if float(field.D2_B(s_star, theta, z)) < 0:
            count += 1
    return count


def _long_well_field():
    """B = 2 + (0.3+0.2 s) cos z + 0.04 cos th cos z with iota = 0.1.

    Along a field line the barrier-top envelope beats over ``2 pi/iota`` — ten
    field periods — so the wells around the marginal tops at ``b=2.4`` span
    several periods: a small ``max_field_periods`` cap genuinely binds and a
    larger one genuinely resolves, which is the per-sample escalation physics
    of the real d23p4 near-``B_max`` cases.
    """
    return SyntheticFourierField(
        nfp=1,
        m=np.array([0, 0, 1, 1]),
        n=np.array([0, 1, 1, -1]),
        cosine_coefficients=np.array([[2.0, 0], [0.3, 0.2], [0.02, 0], [0.02, 0]]),
        sine_coefficients=np.zeros((4, 2)),
        iota_coefficients=np.array([0.1]),
        G_coefficients=np.array([3.0]),
        I_coefficients=np.array([0.0]),
    )


LONG_WELL_B = 2.4


def _two_barrier_field():
    # Same construction as the milestone 10.2 fixture: the lower maxima of
    # cos(3z)+0.2cos(z) exchange heights at theta=0, pi through the
    # sin(theta) sin(zeta) perturbation, giving two equal-height contacts.
    return SyntheticFourierField(
        nfp=1,
        m=np.array([0, 0, 0, 1, 1]),
        n=np.array([0, 3, 1, 1, -1]),
        cosine_coefficients=np.array(
            [[2, 0.2], [1, 0], [0.2, 0], [0, 0.025], [0, -0.025]]
        ),
        sine_coefficients=np.zeros((5, 2)),
        iota_coefficients=np.array([0.0]),
        G_coefficients=np.array([3.0]),
        I_coefficients=np.array([0.0]),
    )


@pytest.fixture(scope="module")
def long_well_pipeline():
    field = _long_well_field()
    background = StructuredPrismMeshBackend(BackgroundMeshConfig(5, 16, 24)).build(
        field
    )
    extraction = MarchingTetrahedraExtractor().extract(background, field, LONG_WELL_B)
    critical = extract_critical_curves(extraction, field, LONG_WELL_B)
    return field, extraction, critical


@pytest.fixture(scope="module")
def fold_pipeline():
    field = _fold_field()
    background = StructuredPrismMeshBackend(BackgroundMeshConfig(4, 12, 24)).build(
        field
    )
    extraction = MarchingTetrahedraExtractor().extract(background, field, FOLD_B)
    critical = extract_critical_curves(extraction, field, FOLD_B)
    return field, extraction, critical


@pytest.fixture(scope="module")
def fold_resolution(fold_pipeline):
    field, extraction, critical = fold_pipeline
    return converge_transitions(
        field,
        extraction,
        critical,
        budgets=RefinementBudgets(
            source_sample_budgets=(8, 16),
            max_field_period_caps=(128,),
            localization_bisections=(20, 60),
            background_levels=0,
            local_refinement_rounds=1,
        ),
        transition_config=TransitionMappingConfig(),
    )


@pytest.fixture(scope="module")
def two_barrier_pipeline():
    field = _two_barrier_field()
    zeta = np.arccos(-np.sqrt(7 / 30))
    b = float(field.B(0.5, 0, zeta))
    background = StructuredPrismMeshBackend(BackgroundMeshConfig(4, 12, 24)).build(
        field
    )
    extraction = MarchingTetrahedraExtractor().extract(background, field, b)
    critical = extract_critical_curves(extraction, field, b)
    return field, b, extraction, critical


@pytest.fixture(scope="module")
def two_barrier_full(two_barrier_pipeline):
    field, b, extraction, critical = two_barrier_pipeline
    return converge_transitions(
        field,
        extraction,
        critical,
        budgets=RefinementBudgets(
            source_sample_budgets=(None,),
            max_field_period_caps=(128,),
            localization_bisections=(20,),
            background_levels=0,
            local_refinement_rounds=1,
        ),
        transition_config=TransitionMappingConfig(),
    )


def _multi_arc_signature(cut):
    """Label-free sheet/port/event incidence signature of a cut surface."""
    order = {}
    for port in cut.ports:
        order.setdefault(int(port.sheet_id), len(order))
    ports = tuple(
        (port.transition_id, port.role, order[int(port.sheet_id)])
        for port in sorted(cut.ports, key=lambda p: (p.transition_id, p.role))
    )
    events = tuple(
        tuple(sorted(order[int(cut.ports[i].sheet_id)] for i in event.port_indices))
        for event in cut.events
    )
    return (
        len(np.unique(cut.sheet_ids)),
        ports,
        events,
        tuple(sorted(map(int, cut.unresolved_transition_ids))),
    )


def test_max_periods_escalation_retraces_failing_samples_and_matches_direct_mapping(
    long_well_pipeline,
):
    field, extraction, critical = long_well_pipeline
    config = TransitionMappingConfig()
    budgets = RefinementBudgets(
        source_sample_budgets=(6,),
        max_field_period_caps=(4, 16),
        localization_bisections=(8,),
        background_levels=0,
        local_refinement_rounds=0,
    )
    # The base cap genuinely binds: this is the failure the coordinator must
    # remediate per sample, not a synthetic monkeypatch.
    base = map_transitions(
        field,
        critical,
        replace(config, max_curve_samples=6, max_field_periods=4),
    )
    assert any(
        reason.endswith("max_periods")
        for transition in base
        for reason in transition.sample_failure_reason
    )

    resolution = converge_transitions(
        field,
        extraction,
        critical,
        budgets=budgets,
        transition_config=config,
    )

    cap_records = [
        record
        for record in resolution.attempts
        if record.failure_class == "max_periods"
    ]
    assert cap_records, "the cap escalation must be recorded (§21.3)"
    assert all(record.control == "max_field_periods" for record in cap_records)
    assert {(record.previous, record.requested) for record in cap_records} == {
        ("4", "16")
    }
    # After escalation no sample failure is a period cap, and the escalated
    # mapping is exactly what direct mapping at the final cap produces: no
    # sample was dropped, zeroed, or relabeled without retracing (§21.2).
    final = resolution.transitions
    assert not any(
        reason.endswith("max_periods")
        for transition in final
        for reason in transition.sample_failure_reason
    )
    direct = map_transitions(
        field,
        critical,
        replace(config, max_curve_samples=6, max_field_periods=16),
    )
    assert len(final) == len(direct)
    for ours, reference in zip(final, direct):
        np.testing.assert_allclose(ours.u, reference.u, rtol=0, atol=1e-12)
        assert ours.sample_status == reference.sample_status
        for port_ours, port_reference in zip(ours.ports, reference.ports):
            np.testing.assert_allclose(
                port_ours.action_values,
                port_reference.action_values,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )


def test_unresolvable_cap_terminates_bounded_with_recorded_attempts(
    long_well_pipeline,
):
    field, extraction, critical = long_well_pipeline
    budgets = RefinementBudgets(
        source_sample_budgets=(6,),
        max_field_period_caps=(2, 4),
        localization_bisections=(8,),
        background_levels=0,
        local_refinement_rounds=0,
    )
    resolution = converge_transitions(
        field,
        extraction,
        critical,
        budgets=budgets,
        transition_config=TransitionMappingConfig(),
    )
    assert resolution.classification == "unresolved_explicit"
    assert not resolution.resolved
    # The ladder is bounded: exactly one cap escalation (2 -> 4) was tried per
    # affected transition, and the terminal state names the cap ceiling.
    cap_records = [r for r in resolution.attempts if r.failure_class == "max_periods"]
    assert cap_records
    assert {(r.previous, r.requested) for r in cap_records} == {("2", "4")}
    assert resolution.failure_class_counts.get("max_periods", 0) > 0
    assert any("cap" in reason for reason in resolution.terminal_reasons)
    # §21.2: capped samples stay explicit MAX_PERIODS with NaN actions; the
    # curve is never cut.
    assert any(
        status is TransitionStatus.MAX_PERIODS
        for transition in resolution.transitions
        for status in transition.sample_status
    )
    if resolution.cut is not None:
        assert len(resolution.cut.cut_edges) == 0


@pytest.mark.slow
def test_source_budget_escalation_certifies_and_matches_full_budget_topology(
    two_barrier_pipeline, two_barrier_full
):
    # This doubles as the milestone 10.1 invariance carried through
    # remediation: the escalated run starts from an insufficient budget of 3
    # and its final sheet graph must equal the full-budget coordinator's —
    # never a silent function of the starting budget.  It is `slow` as a cost
    # control only: the same source-budget escalation keeps fast
    # production-path coverage in the fold test (which certifies through the
    # (8, 16) ladder and records it), and the six-sheet contact topology
    # keeps fast coverage in the localization test below.
    field, b, extraction, critical = two_barrier_pipeline
    escalated = converge_transitions(
        field,
        extraction,
        critical,
        budgets=RefinementBudgets(
            source_sample_budgets=(3, None),
            max_field_period_caps=(128,),
            localization_bisections=(20,),
            background_levels=0,
            local_refinement_rounds=1,
        ),
        transition_config=TransitionMappingConfig(),
    )
    assert escalated.resolved, escalated.terminal_reasons
    budget_records = [
        r for r in escalated.attempts if r.failure_class == "source_budget"
    ]
    assert budget_records, "budget escalation must be recorded"
    assert all(r.control == "max_curve_samples" for r in budget_records)

    full = two_barrier_full
    assert full.resolved, full.terminal_reasons
    assert _multi_arc_signature(escalated.cut) == _multi_arc_signature(full.cut)
    # Two equal-height contacts inside one ordinary well give exactly six
    # limiting trapped branches (the 10.2 acceptance topology).
    assert len(np.unique(escalated.cut.sheet_ids)) == 6
    assert len(escalated.cut.events) == 2
    assert all(event.unresolved for event in escalated.cut.events)


def test_contact_localization_budget_escalates_until_events_localize(
    two_barrier_pipeline,
):
    field, b, extraction, critical = two_barrier_pipeline
    resolution = converge_transitions(
        field,
        extraction,
        critical,
        budgets=RefinementBudgets(
            source_sample_budgets=(None,),
            max_field_period_caps=(128,),
            localization_bisections=(1, 40),
            background_levels=0,
            local_refinement_rounds=1,
        ),
        transition_config=TransitionMappingConfig(),
    )
    records = [
        r for r in resolution.attempts if r.failure_class == "contact_localization"
    ]
    assert records, "localization escalation must be recorded"
    assert all(r.control == "max_bisections" for r in records)
    assert {(r.previous, r.requested) for r in records} == {("1", "40")}
    assert resolution.resolved, resolution.terminal_reasons
    assert len(np.unique(resolution.cut.sheet_ids)) == 6
    assert len(resolution.cut.events) == 2
    # Equal-height contacts must never be certified as folds: each event
    # keeps two marginal maxima and its equal-height kind.
    assert {event.kind for event in resolution.arrangement.events} == {"equal_height"}
    assert all(
        len(event.marginal_points) == 2 for event in resolution.arrangement.events
    )


def test_below_b_fold_event_certifies_and_cuts_with_continuous_limits(
    fold_pipeline, fold_resolution
):
    field, extraction, critical = fold_pipeline
    # Independent fold location: bisect the interior-maximum count computed
    # from the field alone, far from the production scan and solver.
    low, high = 1.0, 2.0
    assert _fold_interior_maximum_count(field, low) == 1
    assert _fold_interior_maximum_count(field, high) == 0
    for _ in range(22):
        middle = 0.5 * (low + high)
        if _fold_interior_maximum_count(field, middle) == 1:
            low = middle
        else:
            high = middle
    theta_fold = 0.5 * (low + high)

    resolution = fold_resolution
    assert resolution.resolved, resolution.terminal_reasons
    arrangement = resolution.arrangement
    # Two fold events on the closed curve (mirror images), each with one
    # marginal point: a fold is not decomposed into a fake equal-height pair.
    assert len(arrangement.events) == 2
    kinds = {event.kind for event in arrangement.events}
    assert kinds == {"fold"}
    assert all(len(event.marginal_points) == 1 for event in arrangement.events)
    angles = sorted(
        np.mod(
            np.arctan2(event.marginal_points[0][1], event.marginal_points[0][0]),
            2 * np.pi,
        )
        for event in arrangement.events
    )
    np.testing.assert_allclose(angles, [theta_fold, 2 * np.pi - theta_fold], atol=2e-3)
    # Across a below-b fold every port action is continuous: the one-sided
    # limits of the two incident arcs agree. A solver that flipped the
    # curvature test or accepted a height at b would move or break this.
    for event in arrangement.events:
        limits = {}
        for arc in arrangement.arcs:
            for end, event_id in zip((0, -1), arc.endpoint_event_ids):
                if event_id != event.event_id:
                    continue
                limits[arc.curve.transition_id] = [
                    port.action_values[end] for port in arc.curve.ports
                ]
        assert len(limits) == 2
        first, second = limits.values()
        np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-8)
    cut = resolution.cut
    assert len(cut.unresolved_transition_ids) == 0
    assert len(cut.events) == 2
    assert all(port.sheet_id >= 0 for port in cut.ports)
    # The (8, 16) ladder is the fast production-path evidence for
    # source-budget escalation: certification needed and used the recorded
    # second rung.
    budget_records = [
        r for r in resolution.attempts if r.failure_class == "source_budget"
    ]
    assert budget_records, "source-budget escalation must be recorded"
    assert all(r.control == "max_curve_samples" for r in budget_records)


def test_thin_strip_local_refinement_enables_the_cut_and_preserves_topology(
    fold_pipeline, fold_resolution
):
    # Under a stricter strip-resolution requirement the fold field's coarse
    # surface genuinely cannot hold the companion-to-EDGE strip, which is the
    # real-matrix failure the §23.10.3 local refinement remedies.  The
    # remediated cut must reproduce, sheet for sheet and event for event, the
    # cut the default requirement accepts directly: refinement may only make
    # the same topology representable, never a different one.
    field, extraction, critical = fold_pipeline
    budgets = RefinementBudgets(
        source_sample_budgets=(8, 16),
        max_field_period_caps=(128,),
        localization_bisections=(20, 60),
        background_levels=0,
        local_refinement_rounds=0,
    )
    strict = ConstrainedCutConfig(min_transition_strip_edge_ratio=0.8)
    base = converge_transitions(
        field,
        extraction,
        critical,
        budgets=budgets,
        transition_config=TransitionMappingConfig(),
        cut_config=strict,
    )
    assert not base.resolved
    assert any("strip width" in reason for reason in base.terminal_reasons)
    assert base.failure_class_counts.get("thin_strip", 0) > 0

    remediated = converge_transitions(
        field,
        extraction,
        critical,
        budgets=replace(budgets, local_refinement_rounds=2),
        transition_config=TransitionMappingConfig(),
        cut_config=strict,
    )
    assert remediated.resolved, remediated.terminal_reasons
    strip_records = [r for r in remediated.attempts if r.failure_class == "thin_strip"]
    assert strip_records, "the strip remediation must be recorded"
    reference = fold_resolution
    assert reference.resolved
    assert _multi_arc_signature(remediated.cut) == _multi_arc_signature(reference.cut)


def test_degenerate_endpoint_arms_certify_their_regular_interior():
    from alpha_analysis import DATA_DIR, BoozerField
    from alpha_analysis.j_connectivity import (
        BoundsConfig,
        CriticalKind,
        find_global_B_bounds,
    )

    # PCA at radially global lambda_n=0.1: every GAMMA_MAX polyline is an
    # open arm whose terminal vertices classify DEGENERATE (the milestone 8
    # junction solves put them on D_parallel^2 B = 0), which used to veto the
    # whole curve.  ADR 0008: the endpoints become explicit events, the
    # 6-vertex arm cuts its regular interior between them, and the 3-vertex
    # arms dissolve entirely into their endpoint events — never a silent
    # truncation and never an invented cut.
    field = BoozerField.from_boozmn(
        Path(DATA_DIR) / "boozmn_20260402-01-038_Ax_PCA_20dofs_allNfp_aspect6_"
        "eval000290_low_resolution.nc"
    )
    bounds = find_global_B_bounds(field, BoundsConfig(17, 32, 32))
    b = bounds.refined_min + 0.1 * (bounds.refined_max - bounds.refined_min)
    background = StructuredPrismMeshBackend(BackgroundMeshConfig(6, 24, 12)).build(
        field
    )
    extraction = MarchingTetrahedraExtractor().extract(background, field, b)
    critical = extract_critical_curves(extraction, field, b)
    maxima = [p for p in critical.polylines if p.kind is CriticalKind.GAMMA_MAX]
    assert maxima and all(not p.closed for p in maxima)
    degenerate_ends = [
        critical.points[p.vertex_ids[index]] for p in maxima for index in (0, -1)
    ]
    assert all(
        critical.point_kind[p.vertex_ids[index]] == CriticalKind.DEGENERATE.value
        for p in maxima
        for index in (0, -1)
    )

    resolution = converge_transitions(
        field,
        extraction,
        critical,
        budgets=RefinementBudgets(
            source_sample_budgets=(8, 16),
            max_field_period_caps=(128,),
            localization_bisections=(20,),
            background_levels=0,
            local_refinement_rounds=1,
        ),
        transition_config=TransitionMappingConfig(),
    )
    arrangement = resolution.arrangement
    endpoint_events = [
        event for event in arrangement.events if event.kind == "degenerate_endpoint"
    ]
    assert len(endpoint_events) == 2 * len(maxima)
    for event in endpoint_events:
        point = event.marginal_points[0]
        s = float(np.sum(point[:2] ** 2))
        theta = float(np.arctan2(point[1], point[0]))
        # The event sits on a solved degenerate marginal point of the field
        # itself: B = b and D_parallel^2 B = 0 to solver accuracy.
        assert abs(float(field.B(s, theta, point[2])) - b) < 1e-8
        assert abs(float(field.D2_B(s, theta, point[2]))) < 1e-6
        assert (
            min(np.linalg.norm(point - end) for end in degenerate_ends) < 1e-9
        ), "endpoint events must sit on the classified terminal vertices"
    # The one arm with interior vertices certifies its regular interior
    # between its two endpoint events — the degenerate endpoints no longer
    # veto the curve.  At this coarse background its slit does not yet
    # separate the surface (the junction complex is under-resolved), so the
    # cut demotes it with the explicit background-refinement reason instead
    # of guessing a one-sheet duplication or truncating silently (§21.2).
    certified_arcs = [
        arc
        for arc in arrangement.arcs
        if arc.curve.status is TransitionStatus.REGULAR and arc.curve.sampling_certified
    ]
    assert certified_arcs
    assert all(
        event_id >= 0 for arc in certified_arcs for event_id in arc.endpoint_event_ids
    )
    assert resolution.failure_class_counts == {"background_geometry": 1}
    assert any(
        "does not separate the surface" in reason
        for reason in resolution.terminal_reasons
    )
    unresolved_ports = [port for port in resolution.cut.ports if port.sheet_id < 0]
    assert unresolved_ports, "the demoted slit keeps its ports (§21.2)"
    # Sources fully dissolved into endpoint events stay visible as events
    # with no incident arc, never as silently absent transitions (§21.2).
    sources_with_arcs = {arc.source_transition_id for arc in arrangement.arcs}
    dissolved = set(range(len(maxima))) - sources_with_arcs
    assert dissolved, "the sub-vertex-resolution arms dissolve into events"
    for source_id in dissolved:
        assert any(
            occurrence.source_transition_id == source_id
            for event in endpoint_events
            for occurrence in event.occurrences
        )


def test_local_refinement_refines_near_the_curve_not_globally():
    from test_mesh_cut import _grid_component
    from alpha_analysis.j_connectivity.surface_refine import (
        refine_surface_near_curves,
    )

    xs = np.linspace(-0.8, 0.8, 17)
    ys = np.linspace(-0.2, 0.2, 5)
    points, triangles, components = _grid_component(
        list(xs), list(ys), point_offset=0, component_id=0
    )
    surface = SurfaceMesh(
        level=1.0,
        period=2.0 * np.pi,
        points=points,
        triangles=triangles,
        B=np.ones(len(points)),
        g=-np.ones(len(points)),
        boundary_tags=np.zeros(len(points), dtype=np.int64),
        point_parent_edges=np.full((len(points), 2), -1, dtype=np.int64),
        triangle_parent_tetrahedra=np.full(len(triangles), -1, dtype=np.int64),
        component_ids=components,
    )
    curve = np.column_stack((np.full(3, -0.7), [-0.1, 0.0, 0.1], np.zeros(3)))
    refined, report = refine_surface_near_curves(surface, (curve,))
    assert report.edges_split > 0
    # Local, not global (§23 10.3): the far side of the surface is untouched
    # — same vertices, same triangles.
    far_original = {
        tuple(np.round(point, 12)) for point in surface.points if point[0] > 0.0
    }
    far_refined = {
        tuple(np.round(point, 12)) for point in refined.points if point[0] > 0.0
    }
    assert far_original == far_refined
    near_refined = [point for point in refined.points if point[0] <= -0.55]
    assert len(near_refined) > np.count_nonzero(surface.points[:, 0] <= -0.55)


def test_local_refinement_actions_never_blend_across_the_companion():
    # Finite pre-cut actions with a jump along the coming cut: refined
    # midpoints of same-branch edges take PL values, midpoints of edges that
    # cross the companion are clamped to their own side or stay NaN — never
    # a parent/child blend (§21.2).  The parent branch spans [2.9, 3.3] and
    # the child branch [0.95, 1.15]; any non-port finite value strictly
    # between the bands is a blend across the discontinuity.
    surface, _ = _surface_and_action()
    transition = _transition()
    companion = np.column_stack(([-0.4, 0.0, 0.4], [0.19, 0.19, 0.19], np.zeros(3)))
    x, y = surface.points[:, 0], surface.points[:, 1]
    action = np.where(
        y < 0.19,
        3.1 + 0.25 * x,
        np.where(np.abs(x) <= 0.4, 1.05 + 0.125 * x, np.nan),
    )
    action[9:] = np.nan
    # Move the child-3 marginal actions out of the parent/child-1 gap so a
    # cross-branch blend (about 2.1 for these branches) cannot hide behind a
    # legitimate child-3 interpolation.
    ports = []
    for port in transition.ports:
        if port.role in {"parent", "child_1"}:
            ports.append(replace(port, points=companion))
        else:
            ports.append(replace(port, action_values=np.array([4.0, 4.05, 4.1])))
    transition = replace(transition, ports=tuple(ports))

    cut, records = converge_cut(
        surface,
        action,
        (transition,),
        budgets=RefinementBudgets(local_refinement_rounds=4),
    )
    assert cut.unresolved_transition_ids.size == 0
    assert len(np.unique(cut.sheet_ids)) == 3
    assert [r.failure_class for r in records].count("thin_strip") >= 1
    values = cut.action_values
    port_vertices = {
        int(vertex) for port in cut.ports for vertex in port.polyline_vertex_ids
    }
    check = np.array(
        [index not in port_vertices for index in range(len(values))]
    ) & np.isfinite(values)
    blended = check & (values > 1.2) & (values < 2.85)
    assert not np.any(blended), values[blended]
    # The crossing-edge midpoints that could not be clamped stay explicit
    # NaN rather than silently absent (§21.2).
    assert np.count_nonzero(~np.isfinite(values)) > 0


def test_zero_width_strip_reports_refinement_bound_not_a_cut():
    surface, action = _surface_and_action()
    transition = _transition()
    # The companion leaves a strip far below any reachable local resolution:
    # bounded refinement must stop at its bound with the explicit terminal
    # reason, never cut, and never loosen the strip check itself.
    companion = np.column_stack(
        ([-0.8, 0.0, 0.8], [0.19999, 0.19999, 0.19999], np.zeros(3))
    )
    action = np.where(surface.points[:, 1] < 0.19999, 3.0, 1.0)
    action[9:] = np.nan
    ports = tuple(
        replace(port, points=companion) if port.role in {"parent", "child_1"} else port
        for port in transition.ports
    )
    transition = replace(transition, ports=ports)

    cut, records = converge_cut(
        surface,
        action,
        (transition,),
        budgets=RefinementBudgets(local_refinement_rounds=2),
    )
    assert np.array_equal(cut.unresolved_transition_ids, [7])
    reason = cut.unresolved_transition_reasons[0]
    assert "refinement bound" in reason
    assert len(cut.cut_edges) == 0
    assert all(port.sheet_id == -1 for port in cut.ports)


def _surface_with_distractor():
    """The two-component cut fixture plus a fine foreign sheet at zeta=0.25.

    The distractor grid sits directly over the companion path but displaced
    in zeta, the geometry the real matrix shows when a nearby ``B=b`` sheet
    is Euclidean-closer than the curve's own component.  Its fine edges make
    its own surface-distance allowance small, so it can never legitimately
    host the whole companion curve.
    """
    from test_mesh_cut import _grid_component

    surface, action = _surface_and_action()
    offset = len(surface.points)
    extra_points, extra_triangles, extra_components = _grid_component(
        [-0.1, 0.0, 0.1, 0.2], [-0.1, 0.0, 0.1], point_offset=offset, component_id=2
    )
    extra_points[:, 2] = 0.25
    points = np.vstack((surface.points, extra_points))
    triangles = np.vstack((surface.triangles, extra_triangles))
    surface = SurfaceMesh(
        level=surface.level,
        period=surface.period,
        points=points,
        triangles=triangles,
        B=np.ones(len(points)),
        g=-np.ones(len(points)),
        boundary_tags=np.concatenate(
            (surface.boundary_tags, np.zeros(len(extra_points), dtype=np.int64))
        ),
        point_parent_edges=np.full((len(points), 2), -1, dtype=np.int64),
        triangle_parent_tetrahedra=np.full(len(triangles), -1, dtype=np.int64),
        component_ids=np.concatenate((surface.component_ids, extra_components)),
    )
    action = np.concatenate((action, np.full(len(extra_points), 2.0)))
    return surface, action


def test_component_enforcement_keeps_curve_on_its_own_component():
    surface, action = _surface_with_distractor()
    transition = _transition()
    # The middle sample is Euclidean-nearer to the foreign zeta=0.25 sheet
    # than to its own component 0.  Nearest-triangle location alone reports a
    # component jump; provenance enforcement must find the one component that
    # holds the whole curve within the surface-distance allowance.
    companion = np.column_stack((np.full(3, 0.1), [-0.2, 0.0, 0.2], [0.0, 0.13, 0.0]))
    ports = tuple(
        replace(port, points=companion) if port.role in {"parent", "child_1"} else port
        for port in transition.ports
    )
    transition = replace(transition, ports=ports)

    cut, records = converge_cut(surface, action, (transition,))
    assert cut.unresolved_transition_ids.size == 0
    parent = next(port for port in cut.ports if port.role == "parent")
    incident_components = set()
    for vertex in parent.polyline_vertex_ids:
        for triangle_id, triangle in enumerate(cut.surface.triangles):
            if int(vertex) in set(map(int, triangle)):
                incident_components.add(int(cut.surface.component_ids[triangle_id]))
    # Cutting on the foreign sheet would merge disconnected components in all
    # but name (§21.2); the cut must live entirely on component 0.
    assert incident_components == {0}


def test_genuinely_off_component_curve_stays_explicitly_unresolved():
    surface, action = _surface_with_distractor()
    transition = _transition()
    # The middle sample is beyond component 0's allowance and the foreign
    # sheet cannot hold the endpoints: no consistent single component exists
    # and enforcement must refuse to cut rather than bridge the gap (§21.2).
    companion = np.column_stack((np.full(3, 0.1), [-0.2, 0.0, 0.2], [0.0, 0.45, 0.0]))
    ports = tuple(
        replace(port, points=companion) if port.role in {"parent", "child_1"} else port
        for port in transition.ports
    )
    transition = replace(transition, ports=ports)

    cut, records = converge_cut(surface, action, (transition,))
    assert np.array_equal(cut.unresolved_transition_ids, [7])
    assert "component" in cut.unresolved_transition_reasons[0]
    assert len(cut.cut_edges) == 0
    assert len(np.unique(cut.sheet_ids)) == 3


def test_degenerate_source_classification_is_terminal_after_background_ladder():
    # B = 2 - cos z + c(s) cos 2z with c(s)=0.2+0.1s: at c=1/4 the marginal
    # maximum's curvature D2_B vanishes identically, so critical-curve
    # vertices near s=0.5 are genuinely DEGENERATE.  Background refinement
    # cannot reclassify physics; the coordinator must record its bounded
    # attempts and terminate with the explicit source-classification reason.
    field = SyntheticFourierField(
        nfp=1,
        m=np.array([0, 0, 0]),
        n=np.array([0, 1, 2]),
        cosine_coefficients=np.array([[2.0, 0.0], [-1.0, 0.0], [0.2, 0.1]]),
        sine_coefficients=np.zeros((3, 2)),
        iota_coefficients=np.array([0.0]),
        G_coefficients=np.array([3.0]),
        I_coefficients=np.array([0.0]),
    )
    b = float(field.B(0.5, 0.0, 0.0))
    levels = [(3, 8, 16), (4, 10, 20)]
    built = []

    def factory(level):
        built.append(level)
        return StructuredPrismMeshBackend(BackgroundMeshConfig(*levels[level])).build(
            field
        )

    resolution = converge_case(
        field,
        b,
        background_factory=factory,
        extractor=MarchingTetrahedraExtractor(),
        budgets=RefinementBudgets(
            source_sample_budgets=(8,),
            max_field_period_caps=(64,),
            localization_bisections=(8,),
            background_levels=1,
            local_refinement_rounds=0,
        ),
        transition_config=TransitionMappingConfig(),
    )
    assert built == [0, 1], "each background escalation rebuilds once, in order"
    assert resolution.classification == "unresolved_explicit"
    background_records = [
        r for r in resolution.attempts if r.failure_class == "unresolved_critical"
    ]
    assert background_records
    assert all(r.control == "background_level" for r in background_records)
    assert resolution.background_level == 1
    assert any(
        "classification" in reason or "degenerate" in reason.lower()
        for reason in resolution.terminal_reasons
    )
    if resolution.cut is not None:
        assert len(resolution.cut.cut_edges) == 0


def test_matrix_report_meets_milestone_10_3_acceptance():
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "validation"
        / "milestone10.3-real-equilibria.json"
    )
    payload = json.loads(path.read_text())
    cases = payload["cases"]
    assert len(cases) == 120
    files = {case["file"] for case in cases.values()}
    assert len(files) == 5
    assert {case["backend"] for case in cases.values()} == {"structured", "gmsh"}
    assert {case["extractor"] for case in cases.values()} == {
        "marching_tetrahedra",
        "pyvista",
    }
    assert len({case["lambda_n"] for case in cases.values()}) == 6

    allowed = {"resolved", "no_transitions", "unresolved_explicit"}
    resolved_count = 0
    for key, case in cases.items():
        assert case["outcome"] == "completed", key
        assert case["classification"] in allowed, key
        if case["classification"] in {"resolved", "no_transitions"}:
            resolved_count += 1
        else:
            # Every unresolved case terminates with a physically meaningful
            # reason and per-failure-class counts, not a silent default.
            assert case["terminal_reasons"], key
            assert case["failure_class_counts"], key
            assert all(case["failure_class_counts"].values()), key
        # No per-case tuning: cases carry recorded attempts, never their own
        # control overrides.  The single controls block is global.
        assert "controls" not in case, key
    assert resolved_count >= 0.95 * len(
        cases
    ), f"only {resolved_count} of {len(cases)} cases resolved"
    assert "controls" in payload
    assert "budgets" in payload["controls"]


def test_local_refinement_diagnostic_writes_png(tmp_path):
    from alpha_analysis.j_connectivity.surface_refine import (
        refine_surface_near_curves,
    )
    from alpha_analysis.j_connectivity.visualization import plot_local_refinement

    surface, _ = _surface_and_action()
    companion = np.column_stack(([-0.4, 0.0, 0.4], [0.19, 0.19, 0.19], np.zeros(3)))
    refined, report = refine_surface_near_curves(surface, (companion,))
    assert len(refined.triangles) > len(surface.triangles)
    path = tmp_path / "local-refinement.png"
    figure = plot_local_refinement(surface, refined, (companion,), path=path)
    assert path.exists() and path.stat().st_size > 0
    # The drawn geometry is the asserted geometry: both triangulations and
    # the companion polyline appear with their real vertex counts.
    axes = figure.axes[0]
    assert any(len(line.get_xdata()) == len(companion) for line in axes.get_lines())
