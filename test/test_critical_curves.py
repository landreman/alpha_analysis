"""Critical-curve extraction tests (DESIGN.md §§5.1–5.4, 10.1, and 23)."""

from __future__ import annotations

from dataclasses import replace

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.optimize import brentq

from alpha_analysis import DATA_DIR, BoozerField

from alpha_analysis.j_connectivity import (
    BackgroundMeshConfig,
    CriticalCurveConfig,
    CriticalCurveStatus,
    CriticalKind,
    MarchingTetrahedraExtractor,
    StructuredPrismMeshBackend,
    SurfaceCurveMesh,
    SurfaceMesh,
    SurfaceStatus,
    extract_critical_curves,
)
from alpha_analysis.j_connectivity.critical_curves import (
    CriticalCurveError,
    _degenerate_point,
    _field_values,
    _resolve_by_curve_walk,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField
from alpha_analysis.j_connectivity.denominator import find_global_B_bounds
from alpha_analysis.j_connectivity.visualization import plot_critical_curves


def _radial_field() -> SyntheticFourierField:
    # B = 1.5 + s + 0.25 cos(zeta), with iota=0.  At b=2 its marginal
    # circles are analytic: (s,zeta)=(0.25,0) is a maximum and
    # (s,zeta)=(0.75,pi) is a minimum.
    return SyntheticFourierField(
        nfp=1,
        m=np.array([0, 0]),
        n=np.array([0, 1]),
        cosine_coefficients=np.array([[1.5, 1.0], [0.25, 0.0]]),
        sine_coefficients=np.zeros((2, 2)),
        iota_coefficients=np.array([0.0]),
        G_coefficients=np.array([1.0]),
        I_coefficients=np.array([0.0]),
    )


def _circle_points(s: float, zeta: float, count: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack(
        (
            np.sqrt(s) * np.cos(theta),
            np.sqrt(s) * np.sin(theta),
            np.full(count, zeta),
        )
    )


def test_production_extraction_recovers_analytic_critical_locations():
    field = _radial_field()
    b = 2.013
    background = StructuredPrismMeshBackend(
        BackgroundMeshConfig(n_radial=6, n_poloidal=16, n_zeta=9)
    ).build(field)

    surface = MarchingTetrahedraExtractor().extract(background, field, b)
    result = extract_critical_curves(surface, field, b)

    assert len(result.polylines) == 2
    by_kind = {polyline.kind: polyline for polyline in result.polylines}
    maximum = result.points[by_kind[CriticalKind.GAMMA_MAX].vertex_ids]
    minimum = result.points[by_kind[CriticalKind.GAMMA_MIN].vertex_ids]
    # B=b at zeta=0 gives s=b-1.75; at zeta=pi it gives s=b-1.25.
    np.testing.assert_allclose(np.sum(maximum[:, :2] ** 2, axis=1), b - 1.75)
    np.testing.assert_allclose(maximum[:, 2], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(np.sum(minimum[:, :2] ** 2, axis=1), b - 1.25)
    np.testing.assert_allclose(minimum[:, 2], np.pi, atol=1.0e-12)
    assert all(polyline.closed for polyline in result.polylines)
    unresolved = extract_critical_curves(
        replace(
            surface,
            status=SurfaceStatus.UNRESOLVED,
            n_unresolved_splits=1,
        ),
        field,
        b,
    )
    assert unresolved.status is CriticalCurveStatus.UNRESOLVED
    assert unresolved.report.source_unresolved_split_count == 1


def test_analytic_min_max_curves_are_classified_and_periodically_stitched(tmp_path):
    field = _radial_field()
    period = 2.0 * np.pi
    maximum = _circle_points(0.25, 0.0, 8)
    # Represent the maximum circle as an open chain whose two endpoint copies
    # lie on opposite sides of the quotient seam.  Stitching must close it.
    maximum = np.vstack((maximum, maximum[0] + np.array([0.0, 0.0, period])))
    minimum = _circle_points(0.75, np.pi, 8)
    points = np.vstack((maximum, minimum))
    maximum_segments = np.column_stack((np.arange(8), np.arange(1, 9)))
    minimum_ids = np.arange(9, 17)
    minimum_segments = np.column_stack((minimum_ids, np.roll(minimum_ids, -1)))
    segments = np.vstack((maximum_segments, minimum_segments))
    tags = np.full(len(points), SurfaceMesh.G_ZERO, dtype=np.int64)
    tags[[0, 8]] |= SurfaceMesh.PERIODIC_SEAM
    s = np.sum(points[:, :2] ** 2, axis=1)
    theta = np.arctan2(points[:, 1], points[:, 0])
    curve = SurfaceCurveMesh(
        period=period,
        points=points,
        segments=segments,
        B=np.asarray(field.B(s, theta, points[:, 2])),
        g=np.asarray(
            field.B(s, theta, points[:, 2]) * field.D_B(s, theta, points[:, 2])
        ),
        boundary_tags=tags,
    )

    result = extract_critical_curves(curve, field, b=2.0)

    assert len(result.polylines) == 2
    assert all(polyline.closed for polyline in result.polylines)
    by_kind = {polyline.kind: polyline for polyline in result.polylines}
    assert set(by_kind) == {CriticalKind.GAMMA_MIN, CriticalKind.GAMMA_MAX}
    max_points = result.points[by_kind[CriticalKind.GAMMA_MAX].vertex_ids]
    min_points = result.points[by_kind[CriticalKind.GAMMA_MIN].vertex_ids]
    np.testing.assert_allclose(np.sum(max_points[:, :2] ** 2, axis=1), 0.25)
    np.testing.assert_allclose(np.mod(max_points[:, 2], period), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.sum(min_points[:, :2] ** 2, axis=1), 0.75)
    np.testing.assert_allclose(min_points[:, 2], np.pi)
    assert np.all(result.D2_B[by_kind[CriticalKind.GAMMA_MAX].vertex_ids] < 0.0)
    assert np.all(result.D2_B[by_kind[CriticalKind.GAMMA_MIN].vertex_ids] > 0.0)

    output = tmp_path / "critical-curves.png"
    figure, axis = plot_critical_curves(result, output_path=output)
    assert output.exists()
    assert len(axis.lines) == 2
    plt.close(figure)


def _sign_changing_field() -> SyntheticFourierField:
    # B = 1.5 + s + 0.25 cos(theta) cos(zeta).  With iota=0, the zeta=0
    # marginal curve has D_parallel^2 B = -0.25 cos(theta), so its type
    # changes at theta=pi/2 through an exactly degenerate point.
    return SyntheticFourierField(
        nfp=1,
        m=np.array([0, 1, 1]),
        n=np.array([0, 1, -1]),
        cosine_coefficients=np.array([[1.5, 1.0], [0.125, 0.0], [0.125, 0.0]]),
        sine_coefficients=np.zeros((3, 2)),
        iota_coefficients=np.array([0.0]),
        G_coefficients=np.array([1.0]),
        I_coefficients=np.array([0.0]),
    )


def _sign_changing_point(theta: float) -> np.ndarray:
    s = 0.5 - 0.25 * np.cos(theta)
    return np.array([np.sqrt(s) * np.cos(theta), np.sqrt(s) * np.sin(theta), 0.0])


def test_ambiguous_segment_refines_to_analytic_degenerate_point():
    field = _sign_changing_field()
    input_theta = np.array([np.pi / 6, 3 * np.pi / 4, 7 * np.pi / 6, 7 * np.pi / 4])
    points = np.vstack([_sign_changing_point(value) for value in input_theta])
    theta = np.arctan2(points[:, 1], points[:, 0])
    s = np.sum(points[:, :2] ** 2, axis=1)
    curve = SurfaceCurveMesh(
        period=2.0 * np.pi,
        points=points,
        segments=np.column_stack((np.arange(4), np.roll(np.arange(4), -1))),
        B=np.asarray(field.B(s, theta, points[:, 2])),
        g=np.asarray(
            field.B(s, theta, points[:, 2]) * field.D_B(s, theta, points[:, 2])
        ),
        boundary_tags=np.full(4, SurfaceMesh.G_ZERO, dtype=np.int64),
    )

    result = extract_critical_curves(
        curve,
        field,
        b=2.0,
        config=CriticalCurveConfig(D2_tolerance=1.0e-10, max_refinement_levels=4),
    )

    degenerate = result.point_kind == CriticalKind.DEGENERATE
    assert np.count_nonzero(degenerate) == 2
    degenerate_points = result.points[degenerate]
    theta = np.sort(
        np.mod(
            np.arctan2(degenerate_points[:, 1], degenerate_points[:, 0]),
            2.0 * np.pi,
        )
    )
    np.testing.assert_allclose(theta, [np.pi / 2.0, 3.0 * np.pi / 2.0], atol=1.0e-8)
    np.testing.assert_allclose(
        np.sum(degenerate_points[:, :2] ** 2, axis=1), 0.5, atol=1.0e-9
    )
    np.testing.assert_allclose(degenerate_points[:, 2], 0.0, atol=1.0e-10)
    s = np.sum(result.points[:, :2] ** 2, axis=1)
    theta = np.arctan2(result.points[:, 1], result.points[:, 0])
    B = np.asarray(field.B(s, theta, result.points[:, 2]))
    g = B * np.asarray(field.D_B(s, theta, result.points[:, 2]))
    np.testing.assert_allclose(B, 2.0, atol=1.0e-10)
    np.testing.assert_allclose(g, 0.0, atol=1.0e-10)
    assert {polyline.kind for polyline in result.polylines} == {
        CriticalKind.GAMMA_MIN,
        CriticalKind.GAMMA_MAX,
    }
    assert result.report.refined_segment_count >= 1
    assert result.report.unresolved_segment_count == 0
    assert (
        result.report.ambiguous_segment_length_history[-1]
        < result.report.ambiguous_segment_length_history[0]
    )


def test_unphysical_curve_gap_is_never_reported_regular():
    field = _radial_field()
    points = _circle_points(0.25, 0.0, 4)[:2]
    curve = SurfaceCurveMesh(
        period=2.0 * np.pi,
        points=points,
        segments=np.array([[0, 1]], dtype=np.int64),
        B=np.full(2, 2.0),
        g=np.zeros(2),
        boundary_tags=np.full(2, SurfaceMesh.G_ZERO, dtype=np.int64),
    )

    result = extract_critical_curves(curve, field, b=2.0)

    assert result.status is CriticalCurveStatus.UNRESOLVED
    assert result.report.unresolved_endpoint_count == 2


def test_failed_degenerate_solve_returns_explicit_unresolved_segments():
    field = _sign_changing_field()
    input_theta = np.array([np.pi / 6, 3 * np.pi / 4, 7 * np.pi / 6, 7 * np.pi / 4])
    points = np.vstack([_sign_changing_point(value) for value in input_theta])
    theta = np.arctan2(points[:, 1], points[:, 0])
    s = np.sum(points[:, :2] ** 2, axis=1)
    curve = SurfaceCurveMesh(
        period=2.0 * np.pi,
        points=points,
        segments=np.column_stack((np.arange(4), np.roll(np.arange(4), -1))),
        B=np.asarray(field.B(s, theta, points[:, 2])),
        g=np.asarray(
            field.B(s, theta, points[:, 2]) * field.D_B(s, theta, points[:, 2])
        ),
        boundary_tags=np.full(4, SurfaceMesh.G_ZERO, dtype=np.int64),
    )

    # Both bounded searches are disabled: the chord-local solve by its
    # locality ratio, and the curve walk by its arc-length budget.
    result = extract_critical_curves(
        curve,
        field,
        b=2.0,
        config=CriticalCurveConfig(
            max_midpoint_displacement_ratio=1.0e-3,
            max_walk_arclength_ratio=1.0e-6,
        ),
    )

    assert result.status is CriticalCurveStatus.UNRESOLVED
    assert result.report.unresolved_segment_count == 2
    assert result.report.degenerate_solve_failure_count == 2
    assert result.report.curve_walk_junction_count == 0
    assert result.report.boundary_exit_split_count == 0


def test_curve_walk_resolves_junctions_the_chord_gate_rejects():
    # Same configuration as the test above, with the curve walk left enabled:
    # a junction the chord-local gate rejects must still be found, and at the
    # analytic degenerate points theta = pi/2 and 3 pi/2 of _sign_changing_field.
    field = _sign_changing_field()
    input_theta = np.array([np.pi / 6, 3 * np.pi / 4, 7 * np.pi / 6, 7 * np.pi / 4])
    points = np.vstack([_sign_changing_point(value) for value in input_theta])
    theta = np.arctan2(points[:, 1], points[:, 0])
    s = np.sum(points[:, :2] ** 2, axis=1)
    curve = SurfaceCurveMesh(
        period=2.0 * np.pi,
        points=points,
        segments=np.column_stack((np.arange(4), np.roll(np.arange(4), -1))),
        B=np.asarray(field.B(s, theta, points[:, 2])),
        g=np.asarray(
            field.B(s, theta, points[:, 2]) * field.D_B(s, theta, points[:, 2])
        ),
        boundary_tags=np.full(4, SurfaceMesh.G_ZERO, dtype=np.int64),
    )

    result = extract_critical_curves(
        curve,
        field,
        b=2.0,
        config=CriticalCurveConfig(max_midpoint_displacement_ratio=1.0e-3),
    )

    assert result.report.degenerate_solve_failure_count == 0
    assert result.report.curve_walk_junction_count == 2
    assert result.report.unresolved_segment_count == 0
    degenerate = result.points[result.point_kind == CriticalKind.DEGENERATE]
    assert len(degenerate) == 2
    walked_theta = np.sort(
        np.mod(np.arctan2(degenerate[:, 1], degenerate[:, 0]), 2.0 * np.pi)
    )
    np.testing.assert_allclose(
        walked_theta, [0.5 * np.pi, 1.5 * np.pi], rtol=0.0, atol=1.0e-10
    )
    np.testing.assert_allclose(
        np.sum(degenerate[:, :2] ** 2, axis=1), 0.5, rtol=0.0, atol=1.0e-10
    )


def _fold_junction_field() -> SyntheticFourierField:
    # Along a field line with iota=0.4, alpha=5 theta-2 zeta is constant and
    # B=2+0.1s-cos(u)+(0.3+0.1s^3 cos(alpha))cos(2u)
    #   +0.02 cos(3u+0.9), u=zeta-0.23.
    # The unlocked third-harmonic phase makes its extrema births true folds:
    # B=b, D_B=D2_B=0 at isolated points with nonsingular transverse data.
    phase = 0.23
    third_phase = 0.9 - 3.0 * phase
    cosine = np.zeros((6, 4))
    sine = np.zeros((6, 4))
    cosine[0, :2] = [2.0, 0.1]
    cosine[1, 0] = -np.cos(phase)
    sine[1, 0] = np.sin(phase)
    cosine[2, 0] = 0.3 * np.cos(2.0 * phase)
    sine[2, 0] = -0.3 * np.sin(2.0 * phase)
    cosine[3, 3] = 0.05 * np.cos(2.0 * phase)
    sine[3, 3] = -0.05 * np.sin(2.0 * phase)
    cosine[4, 3] = 0.05 * np.cos(2.0 * phase)
    sine[4, 3] = 0.05 * np.sin(2.0 * phase)
    cosine[5, 0] = 0.02 * np.cos(third_phase)
    sine[5, 0] = 0.02 * np.sin(third_phase)
    return SyntheticFourierField(
        nfp=1,
        m=np.array([0, 0, 0, 5, 5, 0]),
        n=np.array([0, 1, 2, 4, 0, 3]),
        cosine_coefficients=cosine,
        sine_coefficients=sine,
        iota_coefficients=np.array([0.4]),
        G_coefficients=np.array([1.0]),
        I_coefficients=np.array([0.0]),
    )


def _analytic_fold_junctions(b: float) -> tuple[float, np.ndarray, float]:
    """Return analytic ``(s, theta, zeta)`` junction data for the fold field."""
    amplitude = 0.02
    delta = 0.9

    def second_harmonic_amplitude(u: float) -> float:
        return (np.cos(u) - 9.0 * amplitude * np.cos(3.0 * u + delta)) / (
            4.0 * np.cos(2.0 * u)
        )

    def first_parallel_derivative(u: float) -> float:
        coefficient = second_harmonic_amplitude(u)
        return (
            np.sin(u)
            - 2.0 * coefficient * np.sin(2.0 * u)
            - 3.0 * amplitude * np.sin(3.0 * u + delta)
        )

    u = brentq(first_parallel_derivative, -0.6, -0.2, xtol=1.0e-14)
    coefficient = second_harmonic_amplitude(u)
    s = 10.0 * (
        b
        - 2.0
        + np.cos(u)
        - coefficient * np.cos(2.0 * u)
        - amplitude * np.cos(3.0 * u + delta)
    )
    cos_alpha = (coefficient - 0.3) / (0.1 * s**3)
    alpha = np.array([np.arccos(cos_alpha), -np.arccos(cos_alpha)])
    zeta = (u + 0.23) % (2.0 * np.pi)
    theta = np.sort(
        np.mod(
            np.concatenate(
                [
                    (value + 2.0 * zeta + 2.0 * np.pi * np.arange(5)) / 5.0
                    for value in alpha
                ]
            ),
            2.0 * np.pi,
        )
    )
    return float(s), theta, float(zeta)


def test_production_fold_junctions_match_analytic_locations_and_residuals():
    field = _fold_junction_field()
    b = 1.36
    background = StructuredPrismMeshBackend(
        BackgroundMeshConfig(n_radial=8, n_poloidal=24, n_zeta=25)
    ).build(field)
    extraction = MarchingTetrahedraExtractor().extract(background, field, b)

    result = extract_critical_curves(extraction, field, b)

    assert result.report.refined_segment_count == 10
    assert result.report.degenerate_solve_failure_count == 0
    assert result.report.unresolved_segment_count == 0
    degenerate_points = result.points[result.point_kind == CriticalKind.DEGENERATE]
    s = np.sum(degenerate_points[:, :2] ** 2, axis=1)
    theta = np.arctan2(degenerate_points[:, 1], degenerate_points[:, 0])
    expected_s, expected_theta, expected_zeta = _analytic_fold_junctions(b)
    np.testing.assert_allclose(s, expected_s, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(
        np.sort(np.mod(theta, 2.0 * np.pi)),
        expected_theta,
        rtol=0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        degenerate_points[:, 2], expected_zeta, rtol=0.0, atol=1.0e-10
    )
    B = np.asarray(field.B(s, theta, degenerate_points[:, 2]))
    g = B * np.asarray(field.D_B(s, theta, degenerate_points[:, 2]))
    D2_B = np.asarray(field.D2_B(s, theta, degenerate_points[:, 2]))
    np.testing.assert_allclose(B, b, atol=1.0e-9)
    np.testing.assert_allclose(g, 0.0, atol=1.0e-9)
    np.testing.assert_allclose(D2_B, 0.0, atol=1.0e-8)


def test_near_zero_midpoint_stays_unresolved_when_refinement_is_disabled():
    # On zeta=0, D2_B=-0.1(1+cos(theta)) touches zero at theta=pi without
    # changing sign.  Same-kind endpoints must not classify through that event.
    field = SyntheticFourierField(
        nfp=1,
        m=np.array([0, 0, 1, 1]),
        n=np.array([0, 1, 1, -1]),
        cosine_coefficients=np.array(
            [[1.5, 1.0], [0.1, 0.0], [0.05, 0.0], [0.05, 0.0]]
        ),
        sine_coefficients=np.zeros((4, 2)),
        iota_coefficients=np.array([0.0]),
        G_coefficients=np.array([1.0]),
        I_coefficients=np.array([0.0]),
    )
    theta = np.array([np.pi - 0.4, np.pi + 0.4])
    amplitude = 0.1 * (1.0 + np.cos(theta))
    s = 0.5 - amplitude
    points = np.column_stack(
        (np.sqrt(s) * np.cos(theta), np.sqrt(s) * np.sin(theta), np.zeros(2))
    )
    curve = SurfaceCurveMesh(
        period=2.0 * np.pi,
        points=points,
        segments=np.array([[0, 1]], dtype=np.int64),
        B=np.full(2, 2.0),
        g=np.zeros(2),
        boundary_tags=np.full(2, SurfaceMesh.G_ZERO, dtype=np.int64),
    )

    result = extract_critical_curves(
        curve,
        field,
        b=2.0,
        config=CriticalCurveConfig(max_refinement_levels=0),
    )

    assert result.report.unresolved_segment_count == 1
    assert result.segment_kind[0] == CriticalKind.DEGENERATE
    assert result.status is CriticalCurveStatus.UNRESOLVED


def test_midpoint_sampling_finds_two_type_changes_inside_each_coarse_segment():
    # B=1.5+s+0.25 cos(3 theta)cos(zeta) has D2_B=-0.25 cos(3 theta)
    # on zeta=0. Each 2pi/3 coarse segment has maximum-class endpoints but a
    # minimum-class midpoint, with two analytic degenerate points inside.
    field = SyntheticFourierField(
        nfp=1,
        m=np.array([0, 3, 3]),
        n=np.array([0, 1, -1]),
        cosine_coefficients=np.array([[1.5, 1.0], [0.125, 0.0], [0.125, 0.0]]),
        sine_coefficients=np.zeros((3, 2)),
        iota_coefficients=np.array([0.0]),
        G_coefficients=np.array([1.0]),
        I_coefficients=np.array([0.0]),
    )
    theta = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    s = 0.5 - 0.25 * np.cos(3.0 * theta)
    points = np.column_stack(
        (np.sqrt(s) * np.cos(theta), np.sqrt(s) * np.sin(theta), np.zeros(3))
    )
    curve = SurfaceCurveMesh(
        period=2.0 * np.pi,
        points=points,
        segments=np.column_stack((np.arange(3), np.roll(np.arange(3), -1))),
        B=np.full(3, 2.0),
        g=np.zeros(3),
        boundary_tags=np.full(3, SurfaceMesh.G_ZERO, dtype=np.int64),
    )

    result = extract_critical_curves(curve, field, b=2.0)

    assert result.report.unresolved_segment_count == 0
    assert result.report.degenerate_solve_failure_count == 0
    assert np.count_nonzero(result.point_kind == CriticalKind.DEGENERATE) == 6
    assert {polyline.kind for polyline in result.polylines} == {
        CriticalKind.GAMMA_MIN,
        CriticalKind.GAMMA_MAX,
    }


def _chord_curvature_field() -> SyntheticFourierField:
    # B = 1.5 + s + 0.45 cos(theta) cos(zeta) + 0.2 s cos(2 zeta), iota = 0.
    # On zeta=0 the marginal curve at b=2 is s(theta) = (0.5 - 0.45 cos theta)/1.2
    # and D_parallel^2 B = -0.45 cos(theta) - 0.8 s, which on that curve is
    # -(0.45 cos theta + 1)/3: negative for every theta, so the whole curve is
    # GAMMA_MAX.  Interpolating (s, theta) between two curve points does not
    # stay on the curve, and at the interpolated midpoint the same expression
    # can be positive.
    return SyntheticFourierField(
        nfp=1,
        m=np.array([0, 1, 1, 0]),
        n=np.array([0, 1, -1, 2]),
        cosine_coefficients=np.array(
            [[1.5, 1.0], [0.225, 0.0], [0.225, 0.0], [0.0, 0.2]]
        ),
        sine_coefficients=np.zeros((4, 2)),
        iota_coefficients=np.array([0.0]),
        G_coefficients=np.array([1.0]),
        I_coefficients=np.array([0.0]),
    )


def test_midpoint_is_sampled_on_the_curve_not_on_the_chord():
    field = _chord_curvature_field()
    b = 2.0
    theta = np.array([0.55 * np.pi, 1.45 * np.pi])
    s = (b - 1.5 - 0.45 * np.cos(theta)) / 1.2
    points = np.column_stack(
        (np.sqrt(s) * np.cos(theta), np.sqrt(s) * np.sin(theta), np.zeros(2))
    )
    np.testing.assert_allclose(
        np.asarray(field.B(s, theta, np.zeros(2))), b, rtol=0.0, atol=1.0e-12
    )
    # The analytic trap: both endpoints are GAMMA_MAX, so is every point of the
    # curve between them, but the interpolated midpoint reports GAMMA_MIN.
    endpoint_D2 = np.asarray(field.D2_B(s, theta, np.zeros(2)))
    assert np.all(endpoint_D2 < 0.0)
    interpolated_s = float(np.mean(s))
    interpolated_D2 = float(
        np.asarray(field.D2_B(np.array([interpolated_s]), np.array([np.pi]), 0.0))[0]
    )
    assert interpolated_D2 > 0.0
    curve_D2 = float(
        np.asarray(
            field.D2_B(np.array([(b - 1.5 + 0.45) / 1.2]), np.array([np.pi]), 0.0)
        )[0]
    )
    assert curve_D2 < 0.0

    curve = SurfaceCurveMesh(
        period=2.0 * np.pi,
        points=points,
        segments=np.array([[0, 1]], dtype=np.int64),
        B=np.full(2, b),
        g=np.zeros(2),
        boundary_tags=np.full(2, SurfaceMesh.G_ZERO, dtype=np.int64),
    )

    result = extract_critical_curves(curve, field, b=b)

    assert result.segment_kind[0] == CriticalKind.GAMMA_MAX
    assert result.report.unresolved_segment_count == 0
    assert result.report.degenerate_solve_failure_count == 0
    assert result.report.refined_segment_count == 0


def _boozmn_field(name: str) -> BoozerField:
    return BoozerField.from_boozmn(os.path.join(DATA_DIR, name))


def test_boozmn_fold_neck_junction_is_found_beyond_the_chord_gate():
    # Regression for a segment of the 038_Ax_PCA level b=9.8731865848854240
    # whose two endpoints sit on the two arms of one fold of B=b, g=0.  The
    # arms meet at a junction about three chord lengths away, so the
    # chord-local solve rejects it; the two-sided curve walk must find it.
    field = _boozmn_field(
        "boozmn_20260402-01-038_Ax_PCA_20dofs_allNfp_aspect6_eval000290_low_resolution.nc"
    )
    b = 9.873186584885424
    period = 2.0 * np.pi / field.nfp
    first = np.array([0.57398405, 0.53911234, 0.70652848])
    second = np.array([0.57730641, 0.53089672, 0.72708667])
    config = CriticalCurveConfig()
    # The endpoints straddle a type change, so a junction must exist.
    _, _, endpoint_D2 = _field_values(field, np.vstack((first, second)))
    assert endpoint_D2[0] * endpoint_D2[1] < 0.0
    with pytest.raises(CriticalCurveError):
        _degenerate_point(first, second, field, b, period, config)

    reason, resolved = _resolve_by_curve_walk(first, second, field, b, period, config)

    assert reason == "junction"
    junction = resolved[0]
    B, g, D2_B = _field_values(field, junction[np.newaxis, :])
    np.testing.assert_allclose(B, b, rtol=0.0, atol=config.B_tolerance)
    np.testing.assert_allclose(g, 0.0, rtol=0.0, atol=config.g_tolerance)
    np.testing.assert_allclose(D2_B, 0.0, rtol=0.0, atol=config.D2_tolerance)
    assert float(np.sum(junction[:2] ** 2)) <= 1.0
    # It is genuinely outside the old gate: that is why this test exists.
    chord = float(np.linalg.norm(second - first))
    assert np.linalg.norm(junction - 0.5 * (first + second)) > 2.0 * chord


def test_boozmn_fold_neck_pitch_surface_resolves_end_to_end():
    field = _boozmn_field(
        "boozmn_20260402-01-038_Ax_PCA_20dofs_allNfp_aspect6_eval000290_low_resolution.nc"
    )
    b = 9.873186584885424
    background = StructuredPrismMeshBackend(BackgroundMeshConfig(6, 24, 12)).build(
        field
    )
    extraction = MarchingTetrahedraExtractor().extract(background, field, b)
    assert extraction.status is SurfaceStatus.REGULAR

    result = extract_critical_curves(extraction, field, b)

    assert result.report.degenerate_solve_failure_count == 0
    assert result.report.unresolved_segment_count == 0
    assert result.status is CriticalCurveStatus.DEGENERATE
    assert not np.any(result.segment_kind == CriticalKind.DEGENERATE)


def test_boozmn_fold_outside_the_plasma_splits_at_the_edge():
    # Regression for the 178_TURBO level b=7.8114185899999997: the fold whose
    # arms this segment bridges turns around at s slightly above one, so the
    # junction is outside the plasma.  The bridge must be replaced by two arms
    # that each end on s=1 instead of being solved outside the domain.
    field = _boozmn_field(
        "boozmn_20260402-01-178_TURBO_Garabedian_mpol1_xmin0p1_allNfp_aspect6_eval000155.nc"
    )
    bounds = find_global_B_bounds(field)
    b = bounds.refined_min + 0.1 * (bounds.refined_max - bounds.refined_min)
    background = StructuredPrismMeshBackend(BackgroundMeshConfig(6, 24, 12)).build(
        field
    )
    extraction = MarchingTetrahedraExtractor().extract(background, field, b)

    config = CriticalCurveConfig()
    result = extract_critical_curves(extraction, field, b, config)

    assert result.report.boundary_exit_split_count == 2
    assert result.report.degenerate_solve_failure_count == 0
    assert result.status is CriticalCurveStatus.REGULAR
    s = np.sum(result.points[:, :2] ** 2, axis=1)
    inserted = ((result.boundary_tags & SurfaceMesh.EDGE) != 0) & (
        np.abs(s - 1.0) <= config.merge_tolerance
    )
    assert np.count_nonzero(inserted) == 2 * result.report.boundary_exit_split_count
    exits = result.points[inserted]
    B, g, _ = _field_values(field, exits)
    np.testing.assert_allclose(B, b, rtol=0.0, atol=config.B_tolerance)
    np.testing.assert_allclose(g, 0.0, rtol=0.0, atol=config.g_tolerance)
    # No arm is left dangling inside the domain by the split.
    assert result.report.unresolved_endpoint_count == 0


def test_boundary_exit_points_appear_in_the_critical_curve_diagnostic(tmp_path):
    field = _boozmn_field(
        "boozmn_20260402-01-178_TURBO_Garabedian_mpol1_xmin0p1_allNfp_aspect6_eval000155.nc"
    )
    bounds = find_global_B_bounds(field)
    b = bounds.refined_min + 0.1 * (bounds.refined_max - bounds.refined_min)
    background = StructuredPrismMeshBackend(BackgroundMeshConfig(6, 24, 12)).build(
        field
    )
    extraction = MarchingTetrahedraExtractor().extract(background, field, b)
    result = extract_critical_curves(extraction, field, b)

    output = tmp_path / "critical-curves-edge-exit.png"
    figure, axis = plot_critical_curves(result, output_path=output)

    assert output.exists()
    labels = [collection.get_label() for collection in axis.collections]
    assert "edge exit (s=1)" in labels
    plt.close(figure)
