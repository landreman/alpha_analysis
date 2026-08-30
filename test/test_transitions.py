"""Transition-map physics tests (DESIGN.md §§5.3, 10.2, 20.2, and 23)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from dataclasses import replace
from scipy.integrate import quad
from scipy.optimize import brentq

from alpha_analysis.j_connectivity import (
    CriticalKind,
    CriticalCurveStatus,
    SurfaceCurveMesh,
    SurfaceMesh,
    TransitionCurve,
    TransitionMappingConfig,
    TransitionStatus,
    extract_critical_curves,
    map_transitions,
    map_transitions_budget_sweep,
    trace_regular_well,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField
from alpha_analysis.j_connectivity.visualization import plot_transition_diagnostics


def _field(
    *, double_maximum: bool = False, iota: float = 0.0, C: float = 3.0
) -> SyntheticFourierField:
    if double_maximum:
        # B=2-cos(2 zeta) has two distinct equal-height marginal maxima in
        # one field period, so no generic three-port event can be assigned.
        modes = np.array([0, 2])
        cosine = np.array([[2.0], [-1.0]])
    else:
        # B=2-cos(zeta)+(0.3+0.2s)cos(2zeta).  At b=1.4, s=0.5,
        # zeta=0 is a marginal maximum separating two child wells.  The
        # surrounding regular crossings have cos(zeta)=1/4 exactly.
        modes = np.array([0, 1, 2])
        cosine = np.array([[2.0, 0.0], [-1.0, 0.0], [0.3, 0.2]])
    return SyntheticFourierField(
        nfp=1,
        m=np.zeros(len(modes), dtype=np.int64),
        n=modes,
        cosine_coefficients=cosine,
        sine_coefficients=np.zeros_like(cosine),
        iota_coefficients=np.array([iota]),
        G_coefficients=np.array([C]),
        I_coefficients=np.array([0.0]),
    )


def _critical_circles(
    field: SyntheticFourierField,
    *,
    s: float,
    zeta_values: tuple[float, ...],
    count: int = 8,
    theta_offset: float = 0.0,
):
    points = []
    segments = []
    for zeta in zeta_values:
        offset = len(points)
        theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False) + theta_offset
        points.extend(
            np.column_stack(
                (
                    np.sqrt(s) * np.cos(theta),
                    np.sqrt(s) * np.sin(theta),
                    np.full(count, zeta),
                )
            )
        )
        ids = offset + np.arange(count)
        segments.extend(np.column_stack((ids, np.roll(ids, -1))))
    points = np.asarray(points)
    segments = np.asarray(segments, dtype=np.int64)
    s_values = np.sum(points[:, :2] ** 2, axis=1)
    theta = np.arctan2(points[:, 1], points[:, 0])
    B = np.asarray(field.B(s_values, theta, points[:, 2]))
    C = np.asarray(field.G(s_values) + field.iota(s_values) * field.I(s_values))
    g = B * np.asarray(field.D_B(s_values, theta, points[:, 2])) / C
    curve = SurfaceCurveMesh(
        period=2.0 * np.pi,
        points=points,
        segments=segments,
        B=B,
        g=g,
        boundary_tags=np.full(len(points), SurfaceMesh.G_ZERO, dtype=np.int64),
    )
    return extract_critical_curves(curve, field, b=float(B[0]))


def _port(transition, role: str):
    return next(port for port in transition.ports if port.role == role)


def test_generic_split_recovers_T_lifted_identity_and_additive_actions(tmp_path):
    iota = 0.4
    field = _field(iota=iota)
    critical = _critical_circles(field, s=0.5, zeta_values=(0.0,))
    assert {polyline.kind for polyline in critical.polylines} == {
        CriticalKind.GAMMA_MAX
    }

    coarse = map_transitions(
        field,
        critical,
        TransitionMappingConfig(
            action_quadrature_order=12, max_action_quadrature_order=24
        ),
    )[0]
    fine = map_transitions(
        field,
        critical,
        TransitionMappingConfig(
            action_quadrature_order=48, max_action_quadrature_order=96
        ),
    )[0]

    assert fine.status is TransitionStatus.REGULAR
    legacy_positional_curve = TransitionCurve(
        fine.transition_id,
        fine.b,
        fine.u,
        fine.total_u_length,
        fine.ports,
        fine.marginal_points,
        fine.field_line_identity,
        fine.event_zeta_unwrapped,
        fine.additivity_residual,
        fine.status,
        fine.source_critical_status,
        fine.controls,
    )
    assert legacy_positional_curve.sample_status == (fine.status,) * len(fine.u)
    assert fine.source_critical_status is CriticalCurveStatus.REGULAR
    assert len(fine.ports) == 3
    parent = _port(fine, "parent")
    child_1 = _port(fine, "child_1")
    child_3 = _port(fine, "child_3")
    root = np.arccos(0.25)
    np.testing.assert_allclose(parent.zeta_unwrapped, -root, atol=2.0e-11)
    np.testing.assert_allclose(child_1.zeta_unwrapped, -root, atol=2.0e-11)
    np.testing.assert_allclose(child_3.zeta_unwrapped, 0.0, atol=2.0e-11)
    np.testing.assert_allclose(parent.points, child_1.points, atol=2.0e-11)
    np.testing.assert_allclose(child_3.points, fine.marginal_points, atol=2.0e-11)

    # Identity is derived independently from every lifted port sample.  A
    # nearest-neighbor permutation of either curve therefore cannot pass.
    expected_identity = np.column_stack(
        (
            np.full(len(fine.u), 0.5),
            np.mod(
                np.arctan2(fine.marginal_points[:, 1], fine.marginal_points[:, 0]),
                2.0 * np.pi,
            ),
        )
    )
    np.testing.assert_allclose(fine.field_line_identity, expected_identity, atol=2e-12)
    for port in fine.ports:
        theta = np.arctan2(port.points[:, 1], port.points[:, 0])
        alpha = np.mod(theta - iota * port.zeta_unwrapped, 2.0 * np.pi)
        expected_alpha = np.mod(fine.field_line_identity[:, 1], 2.0 * np.pi)
        periodic_error = np.mod(alpha - expected_alpha + np.pi, 2.0 * np.pi) - np.pi
        np.testing.assert_allclose(
            np.sum(port.points[:, :2] ** 2, axis=1),
            fine.field_line_identity[:, 0],
            atol=2.0e-11,
        )
        np.testing.assert_allclose(periodic_error, 0.0, atol=2.0e-11)
    np.testing.assert_array_equal(
        child_3.source_vertex_ids,
        next(
            polyline.vertex_ids
            for polyline in critical.polylines
            if polyline.kind is CriticalKind.GAMMA_MAX
        ),
    )

    def B(zeta):
        return 2.0 - np.cos(zeta) + 0.4 * np.cos(2.0 * zeta)

    expected_child = quad(
        lambda zeta: 3.0 / B(zeta) * np.sqrt(1.0 - B(zeta) / 1.4),
        -root,
        0.0,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
    )[0]
    np.testing.assert_allclose(child_1.action_values, expected_child, rtol=2.0e-10)
    np.testing.assert_allclose(child_3.action_values, expected_child, rtol=2.0e-10)
    np.testing.assert_allclose(parent.action_values, 2.0 * expected_child, rtol=2.0e-8)
    assert np.all(parent.quadrature_error < 1.0e-10)
    assert np.all(child_1.quadrature_error >= 0.0)
    assert np.all(child_3.quadrature_error >= 0.0)
    assert np.max(np.abs(fine.additivity_residual)) < 2.0e-8
    assert np.max(
        _port(fine, "child_1").quadrature_error
        + _port(fine, "child_3").quadrature_error
    ) < 0.15 * np.max(
        _port(coarse, "child_1").quadrature_error
        + _port(coarse, "child_3").quadrature_error
    )
    underresolved = map_transitions(
        field,
        critical,
        TransitionMappingConfig(
            action_quadrature_order=2, max_action_quadrature_order=4
        ),
    )[0]
    assert underresolved.status is TransitionStatus.UNRESOLVED
    assert np.max(np.abs(underresolved.additivity_residual)) > 1.0e-5

    # The common PL arc-length parameter has an independently known circle
    # limit, and its error decreases when transition-curve sampling doubles.
    finer_critical = _critical_circles(field, s=0.5, zeta_values=(0.0,), count=16)
    sampled_finer = map_transitions(
        field,
        finer_critical,
        TransitionMappingConfig(action_quadrature_order=48),
    )[0]
    exact_length = 2.0 * np.pi * np.sqrt(0.5)
    assert abs(sampled_finer.total_u_length - exact_length) < abs(
        fine.total_u_length - exact_length
    )
    capped = map_transitions(
        field,
        finer_critical,
        TransitionMappingConfig(action_quadrature_order=48, max_curve_samples=5),
    )[0]
    source_ids = _port(capped, "child_3").source_vertex_ids
    assert np.all(np.diff(source_ids) > 0)
    assert set(source_ids).issubset(set(range(16)))
    assert len(capped.u) == 5
    assert capped.controls.max_curve_samples == 5
    assert capped.total_u_length == sampled_finer.total_u_length
    assert capped.status is TransitionStatus.BUDGET_INSUFFICIENT
    assert not capped.sampling_certified
    cached_full, cached_capped = (
        item[0]
        for item in map_transitions_budget_sweep(
            field,
            finer_critical,
            (None, 5),
            TransitionMappingConfig(action_quadrature_order=48),
        )
    )
    for cached, standalone in (
        (cached_capped, capped),
        (cached_full, sampled_finer),
    ):
        assert cached.status is standalone.status
        np.testing.assert_allclose(
            cached.field_line_identity, standalone.field_line_identity, atol=1.0e-14
        )
        np.testing.assert_allclose(
            cached.event_zeta_unwrapped,
            standalone.event_zeta_unwrapped,
            atol=1.0e-14,
        )
        for cached_port, standalone_port in zip(cached.ports, standalone.ports):
            np.testing.assert_allclose(
                cached_port.action_values, standalone_port.action_values, atol=1.0e-14
            )

    # Reversing G+iota*I reverses physical tracing: T moves from the negative
    # lifted root to the positive one while the action remains unchanged.
    reverse_field = _field(iota=iota, C=-3.0)
    reverse = map_transitions(
        reverse_field,
        critical,
        TransitionMappingConfig(action_quadrature_order=48),
    )[0]
    assert reverse.status is TransitionStatus.REGULAR
    np.testing.assert_allclose(
        _port(reverse, "parent").zeta_unwrapped, root, atol=2.0e-11
    )
    np.testing.assert_allclose(
        _port(reverse, "parent").action_values,
        parent.action_values,
        rtol=2.0e-10,
    )

    # Approach the transition from the merged and split sides. The local
    # extremum gap is exactly |b-B_max|=0.2|s-0.5| for this field, and all
    # ordinary first-return actions must approach their mapped limiting port.
    def nearby_actions(delta: float):
        result = []
        for s in (0.5 - delta, 0.5 + delta):
            grid = np.linspace(-np.pi, np.pi, 513)
            values = np.asarray(field.B(s, iota * grid, grid)) - 1.4
            roots = [
                brentq(
                    lambda zeta: float(field.B(s, iota * zeta, zeta)) - 1.4,
                    left,
                    right,
                )
                for left, right, left_value, right_value in zip(
                    grid[:-1], grid[1:], values[:-1], values[1:]
                )
                if left_value * right_value < 0.0
            ]
            incoming = [
                root for root in roots if float(field.D_B(s, iota * root, root)) < 0.0
            ]
            result.append(
                np.array(
                    [
                        trace_regular_well(
                            field,
                            1.4,
                            np.array([s, iota * root, root]),
                        ).action_length
                        for root in incoming
                    ]
                )
            )
        return result

    coarse_nearby = nearby_actions(0.02)
    fine_nearby = nearby_actions(0.005)
    limiting = np.array(
        [
            parent.action_values[0],
            child_1.action_values[0],
            child_3.action_values[0],
        ]
    )
    coarse_error = np.abs(
        np.concatenate((coarse_nearby[0], coarse_nearby[1])) - limiting
    )
    fine_error = np.abs(np.concatenate((fine_nearby[0], fine_nearby[1])) - limiting)
    assert np.all(fine_error < coarse_error)
    assert np.max(fine_error) < 0.35 * np.max(coarse_error)

    output = tmp_path / "transition.png"
    figure, axes = plot_transition_diagnostics(field, fine, output_path=output)
    assert output.exists()
    assert len(axes) == 4
    assert any(line.get_label() == r"$T$" for line in axes[0].lines)
    connector_zeta = axes[0].lines[2].get_data_3d()[2]
    assert np.ptp(connector_zeta) < np.pi
    plt.close(figure)

    # A degenerate curve elsewhere does not erase a locally regular maximum
    # polyline; the slice-global status remains attached as report metadata.
    source_unresolved = map_transitions(
        field,
        replace(critical, status=CriticalCurveStatus.UNRESOLVED),
        TransitionMappingConfig(action_quadrature_order=48),
    )[0]
    assert source_unresolved.status is TransitionStatus.REGULAR
    assert source_unresolved.source_critical_status is CriticalCurveStatus.UNRESOLVED


def test_transition_sampling_budget_is_explicit_when_certification_cannot_finish(
    tmp_path,
):
    """A coarse work budget must not masquerade as a resolved cut curve."""
    field = _field(iota=0.4)
    critical = _critical_circles(field, s=0.5, zeta_values=(0.0,), count=16)

    transition = map_transitions(
        field,
        critical,
        TransitionMappingConfig(
            action_quadrature_order=48,
            max_curve_samples=2,
        ),
    )[0]

    assert transition.status is TransitionStatus.BUDGET_INSUFFICIENT
    assert not transition.sampling_certified
    assert transition.sampling_samples_used == 2
    assert transition.authoritative_sample_count == 16
    assert len(transition.sampling_unresolved_intervals) > 0
    assert "budget" in transition.sampling_reason
    output = tmp_path / "sampling-budget.png"
    figure, _ = plot_transition_diagnostics(field, transition, output_path=output)
    try:
        assert output.stat().st_size > 0
        assert "BUDGET_INSUFFICIENT" in figure._suptitle.get_text()
        assert "sampling uncertified (2/16)" in figure._suptitle.get_text()
    finally:
        plt.close(figure)


def test_transition_sampling_refines_nonlinear_port_actions():
    """Action curvature, not just curve geometry, can exhaust the work budget."""
    # B=2-cos(zeta)+(0.3+0.2s+0.02cos(theta))cos(2zeta), iota=0.
    # At b=1.4, the exact marginal curve is zeta=0,
    # s=0.5-0.1cos(theta), and every well profile has coefficient 0.4.
    # G=1+100s^2 therefore makes each limiting action a known nonlinear
    # function on that curve without introducing a nongeneric event.
    field = SyntheticFourierField(
        nfp=1,
        m=np.array([0, 0, 0, 1, 1]),
        n=np.array([0, 1, 2, 2, -2]),
        cosine_coefficients=np.array(
            [[2.0, 0.0], [-1.0, 0.0], [0.3, 0.2], [0.01, 0.0], [0.01, 0.0]]
        ),
        sine_coefficients=np.zeros((5, 2)),
        iota_coefficients=np.array([0.0]),
        G_coefficients=np.array([1.0, 0.0, 100.0]),
        I_coefficients=np.array([0.0]),
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    s = 0.5 - 0.1 * np.cos(theta)
    points = np.column_stack(
        (np.sqrt(s) * np.cos(theta), np.sqrt(s) * np.sin(theta), np.zeros(32))
    )
    ids = np.arange(32, dtype=np.int64)
    critical = extract_critical_curves(
        SurfaceCurveMesh(
            period=2.0 * np.pi,
            points=points,
            segments=np.column_stack((ids, np.roll(ids, -1))),
            B=np.full(32, 1.4),
            g=np.zeros(32),
            boundary_tags=np.full(32, SurfaceMesh.G_ZERO, dtype=np.int64),
        ),
        field,
        1.4,
    )
    controls = TransitionMappingConfig(
        max_curve_samples=8,
        curve_geometry_rtol=1.0,
        curve_action_rtol=1.0e-3,
    )
    limited, full = (
        item[0]
        for item in map_transitions_budget_sweep(field, critical, (8, None), controls)
    )
    assert limited.status is TransitionStatus.BUDGET_INSUFFICIENT
    assert limited.sampling_max_action_error > 0.1
    assert full.status is TransitionStatus.REGULAR
    root = np.arccos(0.25)

    def B(zeta):
        return 2.0 - np.cos(zeta) + 0.4 * np.cos(2.0 * zeta)

    normalized_child = quad(
        lambda zeta: np.sqrt(1.0 - B(zeta) / 1.4) / B(zeta),
        -root,
        0.0,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
    )[0]
    expected = (1.0 + 100.0 * full.field_line_identity[:, 0] ** 2) * normalized_child
    np.testing.assert_allclose(
        _port(full, "child_1").action_values, expected, rtol=2e-10
    )
    np.testing.assert_allclose(
        _port(full, "parent").action_values, 2.0 * expected, rtol=2e-8
    )


def test_one_degenerate_sample_does_not_erase_a_regular_transition_curve():
    """A nongeneric endpoint stays explicit without discarding valid samples."""
    field = _field(iota=0.4)
    critical = _critical_circles(field, s=0.5, zeta_values=(0.0,))
    maximum = critical.polylines[0]
    point_kind = critical.point_kind.copy()
    point_kind[maximum.vertex_ids[0]] = CriticalKind.DEGENERATE.value
    critical = replace(
        critical,
        point_kind=point_kind,
        status=CriticalCurveStatus.DEGENERATE,
    )

    transition = map_transitions(
        field,
        critical,
        TransitionMappingConfig(action_quadrature_order=48),
    )[0]

    assert transition.status is TransitionStatus.UNRESOLVED
    assert transition.sample_status[0] is TransitionStatus.UNRESOLVED
    assert transition.sample_failure_reason[0] == "source_classification"
    assert all(
        status is TransitionStatus.REGULAR for status in transition.sample_status[1:]
    )
    assert set(transition.sample_failure_reason[1:]) == {"regular"}
    for port in transition.ports:
        assert np.isnan(port.action_values[0])
        assert np.all(np.isfinite(port.action_values[1:]))

    limited = map_transitions(
        field,
        critical,
        TransitionMappingConfig(action_quadrature_order=48, max_curve_samples=3),
    )[0]
    assert limited.status is TransitionStatus.UNRESOLVED
    assert not limited.sampling_certified
    assert len(limited.sampling_unresolved_intervals) > 0
    assert "budget" not in limited.sampling_reason


def test_high_mode_transition_actions_resolve_every_internal_extremum():
    """Internal ripple must not fool either child or parent quadrature."""
    ripple = 2.0e-4
    field = SyntheticFourierField(
        nfp=1,
        m=np.array([0, 0, 0, 0]),
        n=np.array([0, 1, 2, 120]),
        cosine_coefficients=np.array(
            [[2.0 - ripple, 0.0], [-1.0, 0.0], [0.3, 0.2], [ripple, 0.0]]
        ),
        sine_coefficients=np.zeros((4, 2)),
        iota_coefficients=np.array([0.4]),
        G_coefficients=np.array([3.0]),
        I_coefficients=np.array([0.0]),
    )
    critical = _critical_circles(field, s=0.5, zeta_values=(0.0,))

    transition = map_transitions(
        field,
        critical,
        TransitionMappingConfig(action_quadrature_order=32),
    )[0]

    assert transition.status is TransitionStatus.REGULAR
    parent = _port(transition, "parent")
    child_1 = _port(transition, "child_1")
    child_3 = _port(transition, "child_3")
    root = brentq(
        lambda zeta: float(field.B(0.5, 0.4 * zeta, zeta)) - 1.4,
        -np.pi,
        -0.1,
    )
    breakpoints = np.linspace(root, 0.0, 961)
    expected_child = quad(
        lambda zeta: 3.0
        / float(field.B(0.5, 0.4 * zeta, zeta))
        * np.sqrt(max(1.0 - float(field.B(0.5, 0.4 * zeta, zeta)) / 1.4, 0.0)),
        root,
        0.0,
        points=breakpoints[1:-1],
        epsabs=1.0e-11,
        epsrel=1.0e-11,
        limit=1000,
    )[0]
    roundoff = 8.0 * np.finfo(float).eps * abs(expected_child)
    assert np.all(
        np.abs(child_1.action_values - expected_child)
        <= 1.01 * child_1.quadrature_error + roundoff
    )
    assert np.all(
        np.abs(child_3.action_values - expected_child)
        <= 1.01 * child_3.quadrature_error + roundoff
    )
    np.testing.assert_allclose(parent.action_values, 2.0 * expected_child, rtol=2.0e-8)


def test_equal_height_multiway_event_is_explicit_and_never_gets_actions():
    field = _field(double_maximum=True)
    critical = _critical_circles(field, s=0.5, zeta_values=(0.5 * np.pi, 1.5 * np.pi))

    transitions = map_transitions(field, critical)

    assert len(transitions) == 2
    assert all(item.status is TransitionStatus.MULTIWAY for item in transitions)
    assert all(
        np.isnan(port.action_values).all()
        for item in transitions
        for port in item.ports
    )

    duplicate_field = _field()
    duplicate_critical = _critical_circles(
        duplicate_field, s=0.5, zeta_values=(0.0, 0.0)
    )
    duplicate_components = map_transitions(duplicate_field, duplicate_critical)
    assert len(duplicate_components) == 2
    assert all(
        item.status is TransitionStatus.MULTIWAY for item in duplicate_components
    )

    point_kind = duplicate_critical.point_kind.copy()
    for polyline in duplicate_critical.polylines:
        point_kind[polyline.vertex_ids[0]] = CriticalKind.DEGENERATE.value
    partially_degenerate_duplicates = map_transitions(
        duplicate_field,
        replace(
            duplicate_critical,
            point_kind=point_kind,
            status=CriticalCurveStatus.DEGENERATE,
        ),
    )
    assert all(
        item.status is TransitionStatus.MULTIWAY
        for item in partially_degenerate_duplicates
    )

    # A curve demoted whole as a duplicate companion keeps no brackets: every
    # sample is non-regular with NaN actions, so no bracket between two
    # regular samples survives, and milestone 10 is not handed subdivision
    # points on a curve it must not subdivide.  The barrier field would
    # otherwise have brackets here -- the same field brackets twice on a
    # single circle.
    stepped_field = _stepped_over_contact_field()
    single = map_transitions(
        stepped_field,
        _critical_circles(stepped_field, s=0.5, zeta_values=(0.0,), count=8),
    )[0]
    assert len(single.contact_sample_pairs) == 2
    coincident = map_transitions(
        stepped_field,
        _critical_circles(stepped_field, s=0.5, zeta_values=(0.0, 0.0), count=8),
    )
    assert len(coincident) == 2
    for item in coincident:
        assert item.status is TransitionStatus.MULTIWAY
        assert set(item.sample_failure_reason) == {"duplicate_companion"}
        assert len(item.contact_sample_pairs) == 0
        assert np.all(item.interior_maximum_count == -1)
        assert np.all(np.isnan(item.barrier_margin))


def test_transition_trace_cap_has_a_distinct_explicit_status():
    # Along each lifted line B=2-cos(theta), theta=alpha+0.2*zeta. Starting
    # from the global maximum B=b=3, the next equal-height contact is five
    # field periods away, beyond this deliberately one-period cap.
    field = SyntheticFourierField(
        nfp=1,
        m=np.array([0, 1]),
        n=np.array([0, 0]),
        cosine_coefficients=np.array([[2.0], [-1.0]]),
        sine_coefficients=np.zeros((2, 1)),
        iota_coefficients=np.array([0.2]),
        G_coefficients=np.array([3.0]),
        I_coefficients=np.array([0.0]),
    )
    zeta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    points = np.column_stack(
        (
            np.full(len(zeta), -np.sqrt(0.5)),
            np.zeros(len(zeta)),
            zeta,
        )
    )
    ids = np.arange(len(points))
    curve = SurfaceCurveMesh(
        period=2.0 * np.pi,
        points=points,
        segments=np.column_stack((ids, np.roll(ids, -1))),
        B=np.full(len(points), 3.0),
        g=np.zeros(len(points)),
        boundary_tags=np.full(len(points), SurfaceMesh.G_ZERO, dtype=np.int64),
    )
    critical = extract_critical_curves(curve, field, b=3.0)

    transitions = map_transitions(
        field,
        critical,
        TransitionMappingConfig(max_field_periods=1),
    )

    assert len(transitions) == 1
    assert transitions[0].status is TransitionStatus.MAX_PERIODS
    assert transitions[0].controls.max_field_periods == 1
    assert all(np.isnan(port.action_values).all() for port in transitions[0].ports)


def _stepped_over_contact_field(delta_0: float = 0.0530, delta_1: float = 0.0045):
    """Return a field whose barrier crosses ``b`` between critical samples.

    Along a field line (``iota=0``, so ``theta`` is constant) the profile is

        B = 2 - cos(zeta) + (0.5 + 0.2 s) cos(2 zeta)
            + delta(theta) [sin(6 zeta) - 6 sin(zeta)],
        delta(theta) = delta_0 + delta_1 cos(theta).

    The bump vanishes with its first and second derivatives at ``zeta=0``, so
    ``zeta=0, s=0.5`` is a marginal maximum at ``b=1.6`` with curvature
    ``-1.4`` for every ``theta``: the whole circle is ``Gamma_max``.  The
    backward half of the parent well contains one further maximum whose height
    passes through ``b`` at ``delta = 0.054276...``, between the eight sampled
    ``theta`` values.  That contact is a second equal-height maximum: a
    nongeneric multiway event (DESIGN.md §5.4) which no sample lands on.
    """
    m = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
    n = np.array([0, 1, 2, 1, 6, -1, 1, -6, 6])
    cosine = np.zeros((len(m), 2))
    cosine[0, 0] = 2.0
    cosine[1, 0] = -1.0
    cosine[2] = (0.5, 0.2)
    sine = np.zeros((len(m), 2))
    sine[3, 0] = 6.0 * delta_0
    sine[4, 0] = -delta_0
    sine[5, 0] = -3.0 * delta_1
    sine[6, 0] = 3.0 * delta_1
    sine[7, 0] = 0.5 * delta_1
    sine[8, 0] = -0.5 * delta_1
    return SyntheticFourierField(
        nfp=1,
        m=m,
        n=n,
        cosine_coefficients=cosine,
        sine_coefficients=sine,
        iota_coefficients=np.array([0.0]),
        G_coefficients=np.array([3.0]),
        I_coefficients=np.array([0.0]),
    )


def _sampled_interior_maxima(field, b, s, theta, half_width=6.0, n_samples=240001):
    """Count the parent well's other maxima by direct sampling of ``B``."""
    zeta = np.linspace(-half_width, half_width, n_samples)
    B = np.asarray(field.B(s, theta, zeta), dtype=float)
    center = np.argmin(np.abs(zeta))
    forward = np.flatnonzero(B[center + 1 :] >= b) + center + 1
    backward = np.flatnonzero(B[:center] >= b)
    assert len(forward) and len(backward), "the sampled well must be bounded"
    inside = (zeta > zeta[backward[-1]]) & (zeta < zeta[forward[0]])
    B_inside, zeta_inside = B[inside], zeta[inside]
    peaks = (
        np.flatnonzero(
            (B_inside[1:-1] > B_inside[:-2]) & (B_inside[1:-1] >= B_inside[2:])
        )
        + 1
    )
    peaks = peaks[np.abs(zeta_inside[peaks]) > 1.0e-2]
    return len(peaks), (b - np.max(B_inside[peaks]) if len(peaks) else np.inf)


def test_equal_height_contact_between_samples_is_multiway_not_regular():
    # DESIGN.md §5.4: a jump in itinerary larger than the generic one-maximum
    # change must be reported, not mapped as a generic three-port event.  Here
    # every sample is regular and additive to round-off, yet a second maximum
    # of the marginal height sits between two adjacent Gamma_max samples, so
    # the port actions are discontinuous in u inside the hyperedge.
    field = _stepped_over_contact_field()
    s, b = 0.5, 1.6
    critical = _critical_circles(field, s=s, zeta_values=(0.0,), count=8)
    assert {polyline.kind for polyline in critical.polylines} == {
        CriticalKind.GAMMA_MAX
    }

    transition = map_transitions(field, critical, TransitionMappingConfig())[0]

    # Independent physics: the interior-maximum count really does change, and
    # only between samples 1-2 and 6-7.
    theta = np.arctan2(
        transition.marginal_points[:, 1], transition.marginal_points[:, 0]
    )
    sampled = np.array(
        [_sampled_interior_maxima(field, b, s, float(value))[0] for value in theta]
    )
    np.testing.assert_array_equal(sampled, np.array([0, 0, 1, 1, 1, 1, 1, 0]))
    np.testing.assert_array_equal(transition.interior_maximum_count, sampled)
    np.testing.assert_array_equal(
        transition.contact_sample_pairs, np.array([[1, 2], [6, 7]])
    )

    # The samples themselves are regular and the actions remain usable: the
    # curve is not discarded, it is labeled.
    assert transition.status is TransitionStatus.MULTIWAY
    assert all(
        status is TransitionStatus.REGULAR for status in transition.sample_status
    )
    assert np.max(np.abs(transition.additivity_residual)) < 1.0e-12
    assert all(np.all(np.isfinite(port.action_values)) for port in transition.ports)

    limited = map_transitions(
        field, critical, TransitionMappingConfig(max_curve_samples=3)
    )[0]
    assert limited.status is TransitionStatus.MULTIWAY
    assert not limited.sampling_certified
    assert len(limited.sampling_unresolved_intervals) > 0
    assert "budget" not in limited.sampling_reason
    assert all(status is TransitionStatus.REGULAR for status in limited.sample_status)

    # The recorded barrier margin shrinks toward the contact and is infinite
    # where the parent well has no other maximum at all.
    assert np.all(np.isinf(transition.barrier_margin[[0, 1, 7]]))
    assert 0.0 < transition.barrier_margin[2] < transition.barrier_margin[4]
    np.testing.assert_allclose(
        transition.barrier_margin[[2, 3, 4]],
        [_sampled_interior_maxima(field, b, s, float(theta[i]))[1] for i in (2, 3, 4)],
        atol=2.0e-6,
    )

    # The jump the contact produces is far larger than the generic variation
    # of A_W along the rest of the curve.
    parent = _port(transition, "parent")
    steps = np.abs(np.diff(parent.action_values))
    assert steps[1] > 10.0 * np.median(steps)

    # Negating delta mirrors the bump in zeta, moving the barrier from the
    # backward (child-1) half of the parent well to the forward (child-3)
    # half.  The two halves are summed, so the mirrored curve must report the
    # same counts, brackets and margins; without the sum a stepped-over
    # contact on the child-3 side would be invisible and the curve would read
    # REGULAR while its port actions jump (§21.2).
    mirrored_field = _stepped_over_contact_field(delta_0=-0.0530, delta_1=-0.0045)
    mirrored = map_transitions(
        mirrored_field,
        _critical_circles(mirrored_field, s=0.5, zeta_values=(0.0,), count=8),
        TransitionMappingConfig(),
    )[0]
    assert mirrored.status is TransitionStatus.MULTIWAY
    mirrored_theta = np.arctan2(
        mirrored.marginal_points[:, 1], mirrored.marginal_points[:, 0]
    )
    np.testing.assert_array_equal(
        mirrored.interior_maximum_count,
        [
            _sampled_interior_maxima(mirrored_field, b, s, float(value))[0]
            for value in mirrored_theta
        ],
    )
    np.testing.assert_array_equal(
        mirrored.interior_maximum_count, transition.interior_maximum_count
    )
    np.testing.assert_array_equal(
        mirrored.contact_sample_pairs, transition.contact_sample_pairs
    )
    np.testing.assert_allclose(
        mirrored.barrier_margin, transition.barrier_margin, atol=1.0e-12
    )

    # A curve whose barrier never crosses b over the same sampling is still
    # regular: the detector does not fire on ordinary variation.
    quiet = map_transitions(
        _stepped_over_contact_field(delta_0=0.050, delta_1=0.0015),
        _critical_circles(
            _stepped_over_contact_field(delta_0=0.050, delta_1=0.0015),
            s=s,
            zeta_values=(0.0,),
            count=8,
        ),
        TransitionMappingConfig(),
    )[0]
    assert quiet.status is TransitionStatus.REGULAR
    assert len(quiet.contact_sample_pairs) == 0
    assert np.all(quiet.interior_maximum_count == 1)


def _shaded_spans(axis):
    """Return the x-extents of the axvspan patches on one axis, ordered."""
    spans = []
    for patch in axis.patches:
        interval = patch.get_path().get_extents(patch.get_patch_transform()).intervalx
        spans.append((float(interval[0]), float(interval[1])))
    return sorted(spans)


def test_interior_maximum_data_reports_the_highest_barrier_inside_the_well():
    # The margin is b minus the *highest* interior barrier, minima are not
    # barriers, and an extremum recorded at or beyond the crossing is outside
    # the well.  On real equilibria a parent well holds dozens of barriers, so
    # the reduction over them has to be the maximum height, not the first or
    # the lowest.
    from alpha_analysis.j_connectivity.transitions import (
        _DirectionalTrace,
        _interior_maximum_data,
    )

    config = TransitionMappingConfig()
    trace = _DirectionalTrace(
        status=TransitionStatus.REGULAR,
        distance=4.0,
        zeta=4.0,
        extrema_distances=np.array([0.5, 1.0, 1.5, 2.0, 4.5]),
        extrema_curvatures=np.array([-1.0, +2.0, -0.5, -3.0, -1.0]),
        extrema_B_minus_b=np.array([-0.7, -2.0, -0.2, -0.9, -0.001]),
    )

    count, margin = _interior_maximum_data(trace, config)

    # Three interior maxima (the positive curvature is a minimum, and the
    # extremum at 4.5 lies beyond the crossing at 4.0); the highest sits
    # 0.2 below b.
    assert count == 3
    assert margin == pytest.approx(0.2)

    empty = replace(
        trace,
        extrema_distances=np.array([1.0]),
        extrema_curvatures=np.array([+2.0]),
        extrema_B_minus_b=np.array([-2.0]),
    )
    assert _interior_maximum_data(empty, config) == (0, np.inf)


def test_only_a_closed_polyline_compares_its_wraparound_arc():
    # An open GAMMA_MAX polyline has no arc from its last vertex back to its
    # first, so (n-1, 0) there would bracket the whole curve and invent a
    # subdivision point on an arc that does not exist.
    from alpha_analysis.j_connectivity.transitions import _between_sample_contacts

    # counts[0] and counts[-1] differ, so only the wrap arc separates them.
    counts = np.array([0, 1, 1, 1], dtype=np.int64)
    regular = [TransitionStatus.REGULAR] * 4

    np.testing.assert_array_equal(
        _between_sample_contacts(counts, regular, closed=False),
        np.array([[0, 1]]),
    )
    np.testing.assert_array_equal(
        _between_sample_contacts(counts, regular, closed=True),
        np.array([[0, 1], [3, 0]]),
    )


def test_a_closed_curve_brackets_a_contact_in_its_wraparound_arc():
    # A closed GAMMA_MAX polyline has one more arc than it has vertex pairs:
    # the one from the last sample back to the first.  A contact there must be
    # bracketed as (n-1, 0), or the curve reports REGULAR while its port
    # actions jump across it (§21.2).  delta(theta) is even about theta = 0,
    # so an unrotated circle always agrees at its first and last sample;
    # rotating the sampled circle puts the count change in the wrap arc.
    field = _stepped_over_contact_field(delta_0=0.0501, delta_1=0.0045)
    critical = _critical_circles(
        field,
        s=0.5,
        zeta_values=(0.0,),
        count=8,
        theta_offset=np.deg2rad(20.0),
    )
    assert critical.polylines[0].closed

    transition = map_transitions(field, critical, TransitionMappingConfig())[0]

    theta = np.arctan2(
        transition.marginal_points[:, 1], transition.marginal_points[:, 0]
    )
    np.testing.assert_array_equal(
        transition.interior_maximum_count,
        [_sampled_interior_maxima(field, 1.6, 0.5, float(value))[0] for value in theta],
    )
    assert transition.status is TransitionStatus.MULTIWAY
    pairs = transition.contact_sample_pairs.tolist()
    assert [7, 0] in pairs, pairs
    np.testing.assert_array_equal(pairs, [[0, 1], [2, 3], [4, 5], [7, 0]])

    # The same bracket drives the diagnostic, so the wraparound shading is
    # exercised against a row the detector emitted rather than an injected
    # one: the tail span is drawn and the complement is not.
    figure, axes = plot_transition_diagnostics(field, transition)
    try:
        spans = _shaded_spans(axes[1])
        u, total = transition.u, transition.total_u_length
        assert len(spans) == 5
        assert (pytest.approx(u[7]), pytest.approx(total)) in spans
        assert (pytest.approx(u[0]), pytest.approx(u[0])) in spans
        assert all(not (span[0] <= u[1] and span[1] >= u[6]) for span in spans), spans
    finally:
        plt.close(figure)


def test_a_bracket_never_outranks_a_sample_level_failure():
    # docs/STATUS.md: "Cap exhaustion is distinct TransitionStatus.MAX_PERIODS".
    # A stepped-over event is recorded in contact_sample_pairs regardless, so
    # it may only lift a curve that is otherwise fully regular; relabeling a
    # capped or failed curve MULTIWAY would hide why it is unusable (§21.2).
    from alpha_analysis.j_connectivity.transitions import _curve_status

    regular = (TransitionStatus.REGULAR,) * 3
    assert _curve_status(regular, 0) is TransitionStatus.REGULAR
    assert _curve_status(regular, 1) is TransitionStatus.MULTIWAY
    for failure in (
        TransitionStatus.MAX_PERIODS,
        TransitionStatus.UNRESOLVED,
        TransitionStatus.MATCH_FAILURE,
        TransitionStatus.TANGENT,
    ):
        mixed = (TransitionStatus.REGULAR, failure, TransitionStatus.REGULAR)
        assert _curve_status(mixed, 0) is failure
        assert _curve_status(mixed, 2) is failure
    # A sample that is itself a multiway event still aggregates to MULTIWAY.
    sampled = (TransitionStatus.REGULAR, TransitionStatus.MULTIWAY)
    assert _curve_status(sampled, 0) is TransitionStatus.MULTIWAY

    # The same rule on the production path: one degenerate source vertex on a
    # curve that also brackets two stepped-over events. The brackets survive,
    # and the curve still reports why it is unusable rather than MULTIWAY.
    field = _stepped_over_contact_field()
    critical = _critical_circles(field, s=0.5, zeta_values=(0.0,), count=8)
    point_kind = np.array(critical.point_kind, dtype=np.int64)
    point_kind[critical.polylines[0].vertex_ids[4]] = CriticalKind.DEGENERATE.value
    damaged = replace(
        critical,
        point_kind=point_kind,
        status=CriticalCurveStatus.DEGENERATE,
    )

    transition = map_transitions(field, damaged, TransitionMappingConfig())[0]

    np.testing.assert_array_equal(
        transition.interior_maximum_count, [0, 0, 1, 1, -1, 1, 1, 0]
    )
    np.testing.assert_array_equal(
        transition.contact_sample_pairs, np.array([[1, 2], [6, 7]])
    )
    assert transition.sample_status[4] is TransitionStatus.UNRESOLVED
    assert transition.status is TransitionStatus.UNRESOLVED


def test_contact_bracket_is_shaded_where_the_contact_actually_lies():
    # DESIGN.md §17.5: a stepped-over contact must be visible in the plot, not
    # only in the status, so a jump in A_p(u) is not read as a steep slope.
    # The shaded band must be the bracket itself -- for a closed curve's
    # wraparound bracket, the two spans it really occupies, never the
    # complement between them (§24: a plausible-looking wrong picture).
    field = _stepped_over_contact_field()
    critical = _critical_circles(field, s=0.5, zeta_values=(0.0,), count=8)
    transition = map_transitions(field, critical, TransitionMappingConfig())[0]
    pairs = transition.contact_sample_pairs
    assert len(pairs) and np.all(pairs[:, 0] < pairs[:, 1])

    figure, axes = plot_transition_diagnostics(field, transition)
    try:
        labels = [text.get_text() for text in axes[1].get_legend().get_texts()]
        assert labels.count("equal-height contact between samples") == 1
        np.testing.assert_allclose(
            _shaded_spans(axes[1]),
            sorted(tuple(transition.u[pair]) for pair in pairs),
            atol=1.0e-12,
        )
    finally:
        plt.close(figure)

    # The wraparound bracket of a closed curve decreases in u.  It covers
    # [u[-1], total_u_length] and [u[0], u[second]], and nothing in between.
    for second, expected_head in ((1, transition.u[1]), (0, transition.u[0])):
        wrapped = replace(
            transition,
            contact_sample_pairs=np.array([[len(transition.u) - 1, second]]),
        )
        figure, axes = plot_transition_diagnostics(field, wrapped)
        try:
            np.testing.assert_allclose(
                _shaded_spans(axes[1]),
                [
                    (transition.u[0], expected_head),
                    (transition.u[-1], transition.total_u_length),
                ],
                atol=1.0e-12,
            )
        finally:
            plt.close(figure)
    # (n-1, 0) is the row _between_sample_contacts actually emits; its second
    # span is empty because u[0] is zero, and the bracket is the tail.
    assert transition.u[0] == 0.0
    assert transition.u[1] < transition.u[-1] < transition.total_u_length
