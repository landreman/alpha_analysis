"""Surface-wide trace and refinement tests (DESIGN.md §§8.4, 9.5, 23)."""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from alpha_analysis.j_connectivity import (
    SurfaceMesh,
    SurfaceRefinementConfig,
    TraceStatus,
    evaluate_surface_data,
    itineraries_are_continuous,
    refine_surface_data,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField
from alpha_analysis.j_connectivity.visualization import plot_surface_data
from alpha_analysis.j_connectivity.well_trace import (
    WellTraceConfig,
    trace_regular_well,
)

_TRACE_CONFIG = WellTraceConfig(
    samples_per_field_period=32,
    quadrature_rtol=1.0e-7,
    quadrature_atol=1.0e-9,
)


def _field(*, split: bool) -> SyntheticFourierField:
    if split:
        # B = 2 - cos(zeta) + (0.3 + 0.2 s) cos(2 zeta). At b=1.4 the
        # central maximum is tangent at s=0.5; below it one well crosses the
        # maximum, while above it the first return closes before the maximum.
        cosine = np.array([[2.0, 0.0], [-1.0, 0.0], [0.3, 0.2]])
        modes = np.array([0, 1, 2])
    else:
        # B = 2 + 0.2 s - cos(zeta), whose action is smooth in s.
        cosine = np.array([[2.0, 0.2], [-1.0, 0.0]])
        modes = np.array([0, 1])
    return SyntheticFourierField(
        nfp=1,
        m=np.zeros(len(modes), dtype=np.int64),
        n=modes,
        cosine_coefficients=cosine,
        sine_coefficients=np.zeros_like(cosine),
        iota_coefficients=np.array([0.0]),
        G_coefficients=np.array([3.0]),
        I_coefficients=np.array([0.0]),
    )


def _incoming_root(field, b: float, s: float) -> float:
    grid = np.linspace(-np.pi, 0.0, 513)
    values = np.asarray(field.B(s, 0.0, grid), dtype=float) - b
    roots = [
        brentq(lambda zeta: float(field.B(s, 0.0, zeta)) - b, left, right)
        for left, right, f_left, f_right in zip(
            grid[:-1], grid[1:], values[:-1], values[1:]
        )
        if f_left * f_right < 0.0
    ]
    return next(root for root in roots if float(field.D_B(s, 0.0, root)) < 0.0)


def _coordinates(points):
    s = np.sum(points[:, :2] ** 2, axis=1)
    theta = np.arctan2(points[:, 1], points[:, 0])
    theta[s == 0.0] = 0.0
    return s, theta, points[:, 2]


def _surface(field, b: float, s_values=(0.2, 0.8)) -> SurfaceMesh:
    points = []
    for s in s_values:
        root = _incoming_root(field, b, s)
        for theta in (-0.12, 0.12):
            points.append(
                [np.sqrt(s) * np.cos(theta), np.sqrt(s) * np.sin(theta), root]
            )
    points = np.asarray(points)
    triangles = np.asarray(
        [
            triangle
            for index in range(len(s_values) - 1)
            for triangle in (
                (2 * index, 2 * index + 2, 2 * index + 1),
                (2 * index + 1, 2 * index + 2, 2 * index + 3),
            )
        ],
        dtype=np.int64,
    )
    s, theta, zeta = _coordinates(points)
    B = np.asarray(field.B(s, theta, zeta), dtype=float)
    C = np.asarray(field.G(s) + field.iota(s) * field.I(s), dtype=float)
    g = B * np.asarray(field.D_B(s, theta, zeta), dtype=float) / C
    return SurfaceMesh(
        level=b,
        period=2.0 * np.pi,
        points=points,
        triangles=triangles,
        B=B,
        g=g,
        boundary_tags=np.zeros(len(points), dtype=np.int64),
        point_parent_edges=np.full((len(points), 2), -1, dtype=np.int64),
        triangle_parent_tetrahedra=np.full(len(triangles), -1, dtype=np.int64),
        component_ids=np.zeros(len(triangles), dtype=np.int64),
    )


def test_smooth_manufactured_action_converges_under_projected_refinement(tmp_path):
    field = _field(split=False)
    b = 2.5
    original = _surface(field, b)
    assert np.all(
        (original.points[:, 2] >= 0.0) & (original.points[:, 2] < original.period)
    )

    initial_data = evaluate_surface_data(original, field, _TRACE_CONFIG)
    assert all(trace.status is TraceStatus.REGULAR for trace in initial_data.traces)
    trace = initial_data.traces[0]
    shifted = trace_regular_well(
        field,
        b,
        trace.q_in + np.array([0.0, 0.0, original.period]),
        _TRACE_CONFIG,
    )
    assert itineraries_are_continuous(trace, shifted, original.period, tolerance=1.0e-8)
    for trace in initial_data.traces:
        s = trace.q_in[0]
        root = np.arccos(2.0 + 0.2 * s - b)
        expected_A = quad(
            lambda zeta: 3.0
            / (2.0 + 0.2 * s - np.cos(zeta))
            * np.sqrt(1.0 - (2.0 + 0.2 * s - np.cos(zeta)) / b),
            -root,
            root,
            epsabs=1.0e-10,
            epsrel=1.0e-10,
        )[0]
        np.testing.assert_allclose(trace.action_length, expected_A, rtol=2.0e-7)

    result = refine_surface_data(
        original,
        field,
        SurfaceRefinementConfig(
            max_levels=2,
            action_interpolation_rtol=0.0,
            action_interpolation_atol=1.0e-12,
            bounce_time_interpolation_rtol=0.0,
            bounce_time_interpolation_atol=1.0e-12,
        ),
        _TRACE_CONFIG,
    )

    action_errors = np.array(
        [level.max_action_interpolation_error for level in result.report.levels]
    )
    bounce_time_errors = np.array(
        [level.max_bounce_time_interpolation_error for level in result.report.levels]
    )
    assert len(result.surface.triangles) > len(original.triangles)
    assert np.all(np.diff(action_errors) < 0.0)
    assert action_errors[-1] < 0.35 * action_errors[0]
    assert np.all(np.diff(bounce_time_errors) < 0.0)
    assert bounce_time_errors[-1] < 0.35 * bounce_time_errors[0]
    assert np.all(np.isfinite(result.edge_indicators.action_interpolation_error))
    assert np.all(np.isfinite(result.edge_indicators.bounce_time_interpolation_error))
    assert all(level.candidate_edge_count == 0 for level in result.report.levels)
    assert not np.any(result.edge_indicators.itinerary_candidate)
    s, theta, zeta = _coordinates(result.surface.points)
    np.testing.assert_allclose(field.B(s, theta, zeta), b, atol=1.0e-10)
    physical_g = (
        field.B(s, theta, zeta)
        * field.D_B(s, theta, zeta)
        / (field.G(s) + field.iota(s) * field.I(s))
    )
    assert np.all(physical_g < 0.0)

    output = tmp_path / "surface-data.png"
    figure, axes = plot_surface_data(result.data, output_path=output)
    assert output.exists()
    assert len(axes) == 6
    plt.close(figure)


def test_return_map_discontinuity_candidates_sharpen_with_refinement():
    field = _field(split=True)
    off_critical = evaluate_surface_data(
        _surface(field, b=1.4, s_values=(0.45, 0.55)), field, _TRACE_CONFIG
    )
    left, right = off_critical.traces[0], off_critical.traces[2]
    assert left.status is right.status is TraceStatus.REGULAR
    assert (left.n_internal_maxima, right.n_internal_maxima) == (1, 0)
    assert not itineraries_are_continuous(
        left, right, off_critical.surface.period, tolerance=0.2
    )
    changed_kinds = replace(
        left,
        extrema_kind=np.ones_like(left.extrema_kind),
        n_internal_maxima=0,
    )
    assert not itineraries_are_continuous(
        left, changed_kinds, off_critical.surface.period, tolerance=0.2
    )
    original = _surface(field, b=1.4, s_values=(0.2, 0.5, 0.8))

    result = refine_surface_data(
        original,
        field,
        SurfaceRefinementConfig(
            max_levels=3,
            action_interpolation_rtol=10.0,
            action_interpolation_atol=10.0,
            bounce_time_interpolation_rtol=10.0,
            bounce_time_interpolation_atol=10.0,
            itinerary_tolerance=0.2,
        ),
        _TRACE_CONFIG,
    )

    candidate_levels = [
        level for level in result.report.levels if level.candidate_edge_count
    ]
    widths = np.array([level.max_candidate_edge_length for level in candidate_levels])
    assert len(candidate_levels) == 4
    assert np.all(np.diff(widths) < 0.0)
    assert widths[-1] <= 0.3 * widths[0]
    assert candidate_levels[-1].candidate_edge_count >= 1
    assert np.count_nonzero(result.edge_indicators.itinerary_candidate) == (
        candidate_levels[-1].candidate_edge_count
    )
    candidate_edges = result.edge_indicators.edges[
        result.edge_indicators.itinerary_candidate
    ]
    s = np.sum(result.surface.points[:, :2] ** 2, axis=1)
    nonregular = ~result.data.regular
    assert all(
        (s[first] - 0.5) * (s[second] - 0.5) <= 1.0e-12
        or nonregular[first]
        or nonregular[second]
        for first, second in candidate_edges
    )
    assert {
        trace.n_internal_maxima
        for trace in result.data.traces
        if trace.status is TraceStatus.REGULAR
    } == {0, 1}
    assert any(
        trace.status is TraceStatus.TANGENT_OR_TRANSITION
        for trace in result.data.traces
    )
    s, theta, zeta = _coordinates(result.surface.points)
    np.testing.assert_allclose(field.B(s, theta, zeta), 1.4, atol=1.0e-10)


def test_shared_boundary_edge_is_reported_without_false_midpoint_provenance():
    field = _field(split=False)
    surface = _surface(field, b=2.5, s_values=(0.8, 1.0))
    tags = np.zeros(len(surface.points), dtype=np.int64)
    tags[2:] = SurfaceMesh.EDGE
    surface = replace(surface, boundary_tags=tags)

    result = refine_surface_data(
        surface,
        field,
        SurfaceRefinementConfig(
            max_levels=1,
            action_interpolation_rtol=10.0,
            action_interpolation_atol=10.0,
            bounce_time_interpolation_rtol=10.0,
            bounce_time_interpolation_atol=10.0,
        ),
        _TRACE_CONFIG,
    )

    np.testing.assert_array_equal(result.surface.triangles, surface.triangles)
    blocked_edges = result.edge_indicators.edges[
        result.edge_indicators.refinement_blocked
    ]
    np.testing.assert_array_equal(blocked_edges, [[2, 3]])
    assert np.isnan(
        result.edge_indicators.action_interpolation_error[
            result.edge_indicators.refinement_blocked
        ]
    ).all()
    assert result.report.levels[0].boundary_protected_edge_count == 1
    assert result.report.levels[0].unresolved_edge_count == 1
