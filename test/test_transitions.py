"""Transition-map physics tests (DESIGN.md §§5.3, 10.2, 20.2, and 23)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import replace
from scipy.integrate import quad
from scipy.optimize import brentq

from alpha_analysis.j_connectivity import (
    CriticalKind,
    CriticalCurveStatus,
    SurfaceCurveMesh,
    SurfaceMesh,
    TransitionMappingConfig,
    TransitionStatus,
    extract_critical_curves,
    map_transitions,
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
):
    points = []
    segments = []
    for zeta in zeta_values:
        offset = len(points)
        theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
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
        TransitionMappingConfig(action_quadrature_order=12),
    )[0]
    fine = map_transitions(
        field,
        critical,
        TransitionMappingConfig(action_quadrature_order=48),
    )[0]

    assert fine.status is TransitionStatus.REGULAR
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
    assert np.max(np.abs(fine.additivity_residual)) < 0.15 * np.max(
        np.abs(coarse.additivity_residual)
    )
    underresolved = map_transitions(
        field,
        critical,
        TransitionMappingConfig(action_quadrature_order=2),
    )[0]
    assert underresolved.status is TransitionStatus.UNRESOLVED
    assert np.max(np.abs(underresolved.additivity_residual)) > 0.1

    # The common PL arc-length parameter has an independently known circle
    # limit, and its error decreases when transition-curve sampling doubles.
    sampled_finer = map_transitions(
        field,
        _critical_circles(field, s=0.5, zeta_values=(0.0,), count=16),
        TransitionMappingConfig(action_quadrature_order=48),
    )[0]
    exact_length = 2.0 * np.pi * np.sqrt(0.5)
    assert abs(sampled_finer.total_u_length - exact_length) < abs(
        fine.total_u_length - exact_length
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

    duplicate_components = map_transitions(
        _field(),
        _critical_circles(_field(), s=0.5, zeta_values=(0.0, 0.0)),
    )
    assert len(duplicate_components) == 2
    assert all(
        item.status is TransitionStatus.MULTIWAY for item in duplicate_components
    )


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
