"""Surface-wide trace and refinement tests (DESIGN.md §§8.4, 9.5, 23)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

import alpha_analysis.j_connectivity.surface_data as surface_data_module
from alpha_analysis import BoozerField
from alpha_analysis.j_connectivity import (
    BoundsConfig,
    GmshBackgroundMeshBackend,
    GmshBackgroundMeshConfig,
    MarchingTetrahedraExtractor,
    SurfaceDownsamplingConfig,
    SurfaceMesh,
    SurfaceRefinementConfig,
    TraceStatus,
    downsample_surface,
    evaluate_surface_data,
    find_global_B_bounds,
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

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


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


def test_out_of_domain_midpoint_projection_remains_explicitly_unresolved():
    # B = 2.97 - s - 0.04 cos(zeta), b=2.  The two endpoints of edge
    # (0, 1) lie at s=0.99, while the level surface bows to s=1.01 at the
    # geometric midpoint.  A local gradient projection therefore leaves the
    # plasma.  Selecting another root from a different in-domain branch would
    # be a silent topology change; the edge must stay blocked and unresolved.
    field = SyntheticFourierField(
        nfp=1,
        m=np.array([0, 0]),
        n=np.array([0, 1]),
        cosine_coefficients=np.array([[2.97, -1.0], [-0.04, 0.0]]),
        sine_coefficients=np.zeros((2, 2)),
        iota_coefficients=np.array([0.0]),
        G_coefficients=np.array([1.0]),
        I_coefficients=np.array([0.0]),
    )
    radius = np.sqrt(0.99)
    points = np.array(
        [
            [radius, 0.0, 2.0 * np.pi / 3.0],
            [radius, 0.0, 4.0 * np.pi / 3.0],
            [radius * np.cos(0.02), radius * np.sin(0.02), 2.0 * np.pi / 3.0],
        ]
    )
    s, theta, zeta = _coordinates(points)
    B = np.asarray(field.B(s, theta, zeta), dtype=float)
    np.testing.assert_allclose(B, 2.0, atol=1.0e-14)
    surface = SurfaceMesh(
        level=2.0,
        period=2.0 * np.pi,
        points=points,
        triangles=np.array([[0, 1, 2]]),
        B=B,
        g=-np.ones(3),
        boundary_tags=np.zeros(3, dtype=np.int64),
        point_parent_edges=np.full((3, 2), -1, dtype=np.int64),
        triangle_parent_tetrahedra=np.array([-1]),
        component_ids=np.array([0]),
    )

    result = refine_surface_data(
        surface,
        field,
        SurfaceRefinementConfig(max_levels=0),
        _TRACE_CONFIG,
    )

    edge_index = np.flatnonzero(
        np.all(result.edge_indicators.edges == np.array([0, 1]), axis=1)
    ).item()
    assert result.edge_indicators.refinement_blocked[edge_index]
    assert result.edge_indicators.projection_failed[edge_index]
    assert result.edge_indicators.itinerary_candidate[edge_index]
    assert np.isnan(result.edge_indicators.action_interpolation_error[edge_index])
    assert result.report.levels[0].projection_failed_edge_count >= 1
    assert result.report.levels[0].boundary_protected_edge_count == 0


def test_real_surface_inverting_projected_bisection_stays_unresolved():
    field = BoozerField.from_boozmn(
        _DATA_DIR / "boozmn_20260402-01-178_TURBO_Garabedian_mpol1_xmin0p1_"
        "allNfp_aspect6_eval000155.nc"
    )
    bounds = find_global_B_bounds(field, BoundsConfig(17, 32, 32))
    b = 0.5 * (bounds.refined_min + bounds.refined_max)
    background = GmshBackgroundMeshBackend(
        GmshBackgroundMeshConfig(target_size=0.3)
    ).build(field)
    incoming = MarchingTetrahedraExtractor().extract(background, field, b).incoming
    reduced = downsample_surface(
        incoming,
        field,
        SurfaceDownsamplingConfig(target_reduction=0.8),
    ).surface

    refinement_config = SurfaceRefinementConfig(
        max_levels=0,
        action_interpolation_rtol=5.0e-2,
        bounce_time_interpolation_rtol=5.0e-2,
    )
    trace_config = WellTraceConfig(quadrature_rtol=1.0e-8, quadrature_atol=1.0e-9)
    result = refine_surface_data(reduced, field, refinement_config, trace_config)

    assert result.report.levels[0].face_validity_failed_edge_count >= 1
    assert np.all(result.data.regular[result.surface.g < -1.0e-10])
    np.testing.assert_allclose(result.surface.B, b, atol=1.0e-10)
    for triangle in result.surface.triangles:
        vertices = result.surface.points[triangle]
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        assert np.linalg.norm(normal) > np.finfo(float).eps


def test_sliver_projected_child_is_blocked_on_the_public_refinement_path(monkeypatch):
    field = SyntheticFourierField(
        nfp=1,
        m=np.array([0, 0]),
        n=np.array([0, 1]),
        cosine_coefficients=np.array([[2.0], [-1.0]]),
        sine_coefficients=np.zeros((2, 1)),
        iota_coefficients=np.array([0.0]),
        G_coefficients=np.array([1.0, 5.0]),
        I_coefficients=np.array([0.0]),
    )
    b = 2.5
    zeta = 4.0 * np.pi / 3.0
    points = np.array(
        [
            [np.sqrt(0.2), 0.0, zeta],
            [np.sqrt(0.8), 0.0, zeta],
            [0.7, 0.2, zeta],
        ]
    )
    s, theta, zeta_values = _coordinates(points)
    B = np.asarray(field.B(s, theta, zeta_values), dtype=float)
    C = np.asarray(field.G(s) + field.iota(s) * field.I(s), dtype=float)
    g = B * np.asarray(field.D_B(s, theta, zeta_values), dtype=float) / C
    surface = SurfaceMesh(
        level=b,
        period=2.0 * np.pi,
        points=points,
        triangles=np.array([[0, 1, 2]]),
        B=B,
        g=g,
        boundary_tags=np.zeros(3, dtype=np.int64),
        point_parent_edges=np.full((3, 2), -1, dtype=np.int64),
        triangle_parent_tetrahedra=np.array([-1]),
        component_ids=np.array([0]),
    )
    original_projection = surface_data_module._projected_midpoint
    unsafe = points[2].copy()
    unsafe[1] = np.nextafter(unsafe[1], -np.inf)

    def sliver_projection(candidate_surface, candidate_field, edge, config):
        if edge != (0, 1):
            return original_projection(candidate_surface, candidate_field, edge, config)
        s_mid, theta_mid, zeta_mid = _coordinates(unsafe[np.newaxis, :])
        C_mid = float(field.G(s_mid[0]) + field.iota(s_mid[0]) * field.I(s_mid[0]))
        g_mid = b * float(field.D_B(s_mid[0], theta_mid[0], zeta_mid[0])) / C_mid
        return unsafe, b, g_mid

    monkeypatch.setattr(surface_data_module, "_projected_midpoint", sliver_projection)
    result = refine_surface_data(
        surface,
        field,
        SurfaceRefinementConfig(max_levels=1),
        _TRACE_CONFIG,
    )

    assert result.report.levels[0].face_validity_failed_edge_count >= 1
    assert result.report.levels[0].refined_edge_count >= 1
