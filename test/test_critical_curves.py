"""Critical-curve extraction tests (DESIGN.md §§5.1–5.4, 10.1, and 23)."""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

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
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField
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

    result = extract_critical_curves(
        curve,
        field,
        b=2.0,
        config=CriticalCurveConfig(max_midpoint_displacement_ratio=1.0e-3),
    )

    assert result.status is CriticalCurveStatus.UNRESOLVED
    assert result.report.unresolved_segment_count == 2
    assert result.report.degenerate_solve_failure_count == 2


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
