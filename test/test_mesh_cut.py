"""Constrained-cut topology tests (DESIGN.md §§10.3--10.5 and 23)."""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
import pytest
from dataclasses import replace

from alpha_analysis import DATA_DIR, BoozerField
from alpha_analysis.j_connectivity import (
    BackgroundMeshConfig,
    ConstrainedCutConfig,
    CriticalCurveStatus,
    SurfaceMesh,
    StructuredPrismMeshBackend,
    TransitionCurve,
    TransitionMappingConfig,
    TransitionPort,
    TransitionStatus,
    WellTraceConfig,
    cut_surface_at_transitions,
    evaluate_surface_data,
    extract_critical_curves,
    load_cut_surface,
    map_transitions,
    MarchingTetrahedraExtractor,
    save_cut_surface,
)
from alpha_analysis.j_connectivity.mesh_cut import _interpolate_port_action
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField
from alpha_analysis.j_connectivity.visualization import plot_cut_surface


def _grid_component(xs, ys, *, point_offset, component_id):
    points = np.array([(x, y, 0.0) for y in ys for x in xs], dtype=float)
    triangles = []
    nx = len(xs)
    for j in range(len(ys) - 1):
        for i in range(nx - 1):
            lower_left = point_offset + j * nx + i
            lower_right = lower_left + 1
            upper_left = lower_left + nx
            upper_right = upper_left + 1
            triangles.extend(
                [
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                ]
            )
    return (
        points,
        np.asarray(triangles, dtype=np.int64),
        np.full(len(triangles), component_id, dtype=np.int64),
    )


def _surface_and_action():
    # The first component contains a discontinuity at x=0.1 which crosses
    # triangle interiors.  The second component is the independent child-3
    # return sheet.  Keeping it separate catches forbidden proximity merges.
    first_points, first_triangles, first_components = _grid_component(
        [-0.8, 0.0, 0.8], [-0.2, 0.0, 0.2], point_offset=0, component_id=0
    )
    offset = len(first_points)
    third_points, third_triangles, third_components = _grid_component(
        [-0.3, 0.0, 0.3], [0.5, 0.6, 0.7], point_offset=offset, component_id=1
    )
    points = np.vstack((first_points, third_points))
    triangles = np.vstack((first_triangles, third_triangles))
    components = np.concatenate((first_components, third_components))
    tags = np.zeros(len(points), dtype=np.int64)
    tags[
        np.isclose(points[:, 1], -0.2) | np.isclose(points[:, 1], 0.2)
    ] |= SurfaceMesh.EDGE
    tags[np.isclose(points[:, 1], 0.5)] |= SurfaceMesh.G_ZERO
    surface = SurfaceMesh(
        level=1.0,
        period=2.0 * np.pi,
        points=points,
        triangles=triangles,
        B=np.ones(len(points)),
        g=-np.ones(len(points)),
        boundary_tags=tags,
        point_parent_edges=np.full((len(points), 2), -1, dtype=np.int64),
        triangle_parent_tetrahedra=np.full(len(triangles), -1, dtype=np.int64),
        component_ids=components,
    )
    action = np.where(points[:, 0] < 0.1, 3.0, 1.0)
    # Gamma-max vertices are tangent traces, so their ordinary pre-cut action
    # is undefined. The cut must populate the finite child-3 limiting action
    # supplied by transition mapping, never zero or a propagated NaN.
    action[offset:] = np.nan
    return surface, action


def _port(role, points, action):
    n = len(points)
    return TransitionPort(
        role=role,
        points=np.asarray(points, dtype=float),
        zeta_unwrapped=np.zeros(n),
        action_values=np.broadcast_to(np.asarray(action, dtype=float), (n,)).copy(),
        quadrature_error=np.zeros(n),
        source_vertex_ids=np.full(n, -1, dtype=np.int64),
    )


def _transition():
    u = np.array([0.0, 0.2, 0.4])
    companion = np.column_stack((np.full(3, 0.1), [-0.2, 0.0, 0.2], np.zeros(3)))
    marginal = np.column_stack(([-0.2, 0.0, 0.2], np.full(3, 0.5), np.zeros(3)))
    return TransitionCurve(
        transition_id=7,
        b=1.0,
        u=u,
        total_u_length=0.4,
        ports=(
            _port("parent", companion, [3.0, 3.1, 3.2]),
            _port("child_1", companion, [1.0, 1.05, 1.1]),
            _port("child_3", marginal, [2.0, 2.05, 2.1]),
        ),
        marginal_points=marginal,
        field_line_identity=np.column_stack((np.full(3, 0.5), u)),
        event_zeta_unwrapped=np.zeros((3, 3)),
        additivity_residual=np.zeros(3),
        status=TransitionStatus.REGULAR,
        source_critical_status=CriticalCurveStatus.REGULAR,
        controls=TransitionMappingConfig(),
    )


def _triangle_components(triangles):
    edge_owner = {}
    adjacency = [set() for _ in triangles]
    for triangle_id, triangle in enumerate(triangles):
        for i in range(3):
            edge = tuple(sorted((int(triangle[i]), int(triangle[(i + 1) % 3]))))
            if edge in edge_owner:
                other = edge_owner[edge]
                adjacency[triangle_id].add(other)
                adjacency[other].add(triangle_id)
            else:
                edge_owner[edge] = triangle_id
    labels = np.full(len(triangles), -1, dtype=np.int64)
    for seed in range(len(triangles)):
        if labels[seed] >= 0:
            continue
        labels[seed] = labels.max() + 1
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if labels[neighbor] < 0:
                    labels[neighbor] = labels[seed]
                    stack.append(neighbor)
    return labels


def test_constrained_cut_duplicates_branch_actions_and_assigns_three_sheets():
    surface, action = _surface_and_action()
    transition = _transition()

    cut = cut_surface_at_transitions(surface, action, [transition])

    assert cut.unresolved_transition_ids.size == 0
    assert set(cut.sheet_ids) == {0, 1, 2}
    np.testing.assert_array_equal(
        cut.sheet_ids, _triangle_components(cut.surface.triangles)
    )
    by_role = {port.role: port for port in cut.ports}
    expected_by_role = {port.role: port for port in transition.ports}
    assert set(by_role) == {"parent", "child_1", "child_3"}
    assert len({port.sheet_id for port in cut.ports}) == 3
    for role, expected_port in expected_by_role.items():
        port = by_role[role]
        assert port.sheet_id >= 0
        assert np.all(port.polyline_vertex_ids >= 0)
        np.testing.assert_allclose(
            cut.action_values[port.polyline_vertex_ids], expected_port.action_values
        )
    np.testing.assert_allclose(
        by_role["parent"].action_values,
        by_role["child_1"].action_values + by_role["child_3"].action_values,
    )

    # Pin the common-u correspondence at every vertex inserted between mapped
    # samples, not only at the three authoritative port samples.
    points = cut.surface.points
    on_companion = (
        np.isclose(points[:, 0], 0.1)
        & (points[:, 1] >= -0.2)
        & (points[:, 1] <= 0.2)
        & np.isclose(points[:, 2], 0.0)
    )
    for role, intercept, slope in (
        ("parent", 3.0, 0.5),
        ("child_1", 1.0, 0.25),
    ):
        port = by_role[role]
        sheet_vertices = np.unique(
            cut.surface.triangles[cut.sheet_ids == port.sheet_id]
        )
        inserted = sheet_vertices[on_companion[sheet_vertices]]
        assert len(inserted) > len(transition.u)
        inserted_u = points[inserted, 1] + 0.2
        np.testing.assert_allclose(
            cut.action_values[inserted],
            intercept + slope * inserted_u,
            rtol=0.0,
            atol=1.0e-12,
        )

    # Parent and child-1 occupy the same geometric T but have distinct mesh
    # IDs, and no triangle is allowed to interpolate across those two values.
    parent = by_role["parent"]
    child = by_role["child_1"]
    np.testing.assert_allclose(
        cut.surface.points[parent.polyline_vertex_ids],
        cut.surface.points[child.polyline_vertex_ids],
        atol=1.0e-14,
    )
    assert not np.any(parent.polyline_vertex_ids == child.polyline_vertex_ids)
    branch_gap = np.min(parent.action_values - child.action_values)
    triangle_actions = cut.action_values[cut.surface.triangles]
    finite_ranges = np.array(
        [
            np.ptp(finite)
            for values in triangle_actions
            if len(finite := values[np.isfinite(values)])
        ]
    )
    assert np.all(finite_ranges < 0.5 * branch_gap)


def test_closed_transition_interpolates_the_wraparound_action_segment():
    transition = replace(_transition(), total_u_length=0.6)
    parameter = 0.5

    for port in transition.ports:
        expected = 0.5 * (port.action_values[-1] + port.action_values[0])
        np.testing.assert_allclose(
            _interpolate_port_action(transition, port, parameter),
            expected,
            rtol=0.0,
            atol=1.0e-14,
        )


def test_cut_topology_survives_pickle_free_npz_round_trip(tmp_path):
    surface, action = _surface_and_action()
    cut = cut_surface_at_transitions(surface, action, [_transition()])
    path = tmp_path / "cut-surface.npz"

    save_cut_surface(path, cut)
    restored = load_cut_surface(path)

    for name in ("points", "triangles", "B", "g", "boundary_tags", "component_ids"):
        np.testing.assert_array_equal(
            getattr(restored.surface, name), getattr(cut.surface, name)
        )
    for name in (
        "action_values",
        "sheet_ids",
        "cut_edges",
        "unresolved_transition_ids",
    ):
        np.testing.assert_array_equal(getattr(restored, name), getattr(cut, name))
    assert [
        (port.transition_id, port.role, port.sheet_id) for port in restored.ports
    ] == [(port.transition_id, port.role, port.sheet_id) for port in cut.ports]
    for restored_port, port in zip(restored.ports, cut.ports):
        np.testing.assert_array_equal(
            restored_port.polyline_vertex_ids, port.polyline_vertex_ids
        )
        np.testing.assert_array_equal(restored_port.action_values, port.action_values)


def test_nongeneric_contact_remains_explicitly_unresolved(tmp_path):
    surface, action = _surface_and_action()
    transition = replace(
        _transition(),
        status=TransitionStatus.MULTIWAY,
        contact_sample_pairs=np.array([[0, 1]], dtype=np.int64),
    )

    cut = cut_surface_at_transitions(surface, action, [transition])

    np.testing.assert_array_equal(cut.unresolved_transition_ids, [7])
    assert cut.unresolved_transition_reasons == (
        "transition has failed samples or a bracketed nongeneric event",
    )
    assert len(cut.ports) == 3
    assert all(port.sheet_id == -1 for port in cut.ports)
    assert all(np.all(port.polyline_vertex_ids == -1) for port in cut.ports)
    # No invented cut means the two original connected components remain two,
    # while the unresolved hyperedge is still present in the port table.
    assert len(np.unique(cut.sheet_ids)) == 2

    path = tmp_path / "unresolved-cut-surface.npz"
    save_cut_surface(path, cut)
    restored = load_cut_surface(path)
    np.testing.assert_array_equal(
        restored.unresolved_transition_ids, cut.unresolved_transition_ids
    )
    assert restored.unresolved_transition_reasons == cut.unresolved_transition_reasons
    for restored_port, port in zip(restored.ports, cut.ports):
        np.testing.assert_array_equal(
            restored_port.polyline_vertex_ids, port.polyline_vertex_ids
        )
        np.testing.assert_array_equal(restored_port.action_values, port.action_values)


@pytest.mark.parametrize(
    "companion_y",
    [
        # Endpoints on EDGE as an open T must terminate, interior sample
        # leaving a strip thinner than the local resolution requirement --
        # this variant also pins the endpoint screen's ordering against the
        # strip screen.
        pytest.param([-0.2, 0.19, 0.2], id="endpoints-on-edge"),
        # The whole curve parallel to EDGE at sub-resolution distance; the
        # endpoints are within the ADR 0004 snap allowance, so only the
        # interior strip screen may refuse it.
        pytest.param([0.19, 0.19, 0.19], id="parallel-to-edge"),
    ],
)
def test_underresolved_T_to_edge_strip_is_not_cut(companion_y):
    surface, action = _surface_and_action()
    transition = _transition()
    companion = np.column_stack(([-0.4, 0.0, 0.4], companion_y, np.zeros(3)))
    ports = tuple(
        replace(port, points=companion) if port.role in {"parent", "child_1"} else port
        for port in transition.ports
    )
    transition = replace(transition, ports=ports)

    cut = cut_surface_at_transitions(surface, action, [transition])

    np.testing.assert_array_equal(cut.unresolved_transition_ids, [7])
    assert "T-to-EDGE strip width" in cut.unresolved_transition_reasons[0]
    assert len(cut.cut_edges) == 0
    assert all(port.sheet_id == -1 for port in cut.ports)


def test_open_T_endpoint_beyond_snap_tolerance_is_not_cut():
    # An open companion curve terminates on the domain boundary, but a coarse
    # PL surface can leave its endpoint strictly interior. Beyond the local
    # snap allowance (ADR 0004) the terminal segment is genuinely unresolved:
    # cutting along the dangling polyline would not separate the parent and
    # child sides, so the transition must become an explicit
    # geometry-unresolved hyperedge rather than a crash or a wrong sheet graph.
    # The synthetic grid's edges are coarse enough that the default allowance
    # covers the whole interior, so tighten the ratio to reach this branch.
    surface, action = _surface_and_action()
    transition = _transition()
    companion = np.column_stack((np.full(3, 0.1), [-0.2, 0.0, 0.15], np.zeros(3)))
    ports = tuple(
        replace(port, points=companion) if port.role in {"parent", "child_1"} else port
        for port in transition.ports
    )
    transition = replace(transition, ports=ports)

    cut = cut_surface_at_transitions(
        surface,
        action,
        [transition],
        config=ConstrainedCutConfig(max_surface_distance_ratio=0.05),
    )

    np.testing.assert_array_equal(cut.unresolved_transition_ids, [7])
    assert "open companion T endpoint" in cut.unresolved_transition_reasons[0]
    assert "snap tolerance" in cut.unresolved_transition_reasons[0]
    assert len(cut.cut_edges) == 0
    assert all(port.sheet_id == -1 for port in cut.ports)
    # No invented cut: the two original components remain the only sheets.
    assert len(np.unique(cut.sheet_ids)) == 2


def test_open_T_endpoint_within_tolerance_is_snapped_to_edge():
    # Within the snap allowance the cut is extended to the nearest EDGE
    # boundary edge, which is split so the cut terminates on the surface edge
    # (ADR 0004). The authoritative samples keep their positions and actions;
    # every vertex on the extension takes the endpoint sample's clamped action.
    surface, action = _surface_and_action()
    transition = _transition()
    companion = np.column_stack((np.full(3, 0.1), [-0.2, 0.0, 0.15], np.zeros(3)))
    ports = tuple(
        replace(port, points=companion) if port.role in {"parent", "child_1"} else port
        for port in transition.ports
    )
    transition = replace(transition, ports=ports)

    cut = cut_surface_at_transitions(surface, action, [transition])

    assert cut.unresolved_transition_ids.size == 0
    assert set(cut.sheet_ids) == {0, 1, 2}
    np.testing.assert_array_equal(
        cut.sheet_ids, _triangle_components(cut.surface.triangles)
    )
    by_role = {port.role: port for port in cut.ports}
    assert len({port.sheet_id for port in cut.ports}) == 3
    for role in ("parent", "child_1"):
        port = by_role[role]
        np.testing.assert_allclose(
            cut.surface.points[port.polyline_vertex_ids], companion, atol=1.0e-14
        )
        np.testing.assert_allclose(
            cut.action_values[port.polyline_vertex_ids], port.action_values
        )
    # The extension reaches the EDGE boundary: the split terminus (0.1, 0.2, 0)
    # exists once per side, tagged EDGE, and carries the clamped endpoint
    # actions (parent 3.2, child-1 1.1) rather than an interpolated blend.
    points = cut.surface.points
    terminus = np.flatnonzero(
        np.isclose(points[:, 0], 0.1)
        & np.isclose(points[:, 1], 0.2)
        & np.isclose(points[:, 2], 0.0)
    )
    assert len(terminus) == 2
    assert np.all((cut.surface.boundary_tags[terminus] & SurfaceMesh.EDGE) != 0)
    assert sorted(cut.action_values[terminus]) == [1.1, 3.2]
    on_extension = (
        np.isclose(points[:, 0], 0.1)
        & (points[:, 1] > 0.15 + 1.0e-12)
        & (points[:, 1] <= 0.2)
        & np.isclose(points[:, 2], 0.0)
    )
    assert on_extension.sum() >= 2
    assert set(np.round(cut.action_values[on_extension], 12)) == {1.1, 3.2}


def test_sub_resolution_double_back_is_not_cut():
    # A companion polyline can inherit a sub-triangle-scale zigzag from the
    # GAMMA_MAX mesh-edge chain: two nearly coincident anti-parallel strands.
    # Inserting it would overlap its own constrained chain and branch the cut
    # graph, so the transition must become an explicit geometry-unresolved
    # hyperedge rather than a crash or a wrong sheet graph (ADR 0005).
    surface, action = _surface_and_action()
    transition = _transition()
    companion = np.column_stack((np.full(3, 0.1), [-0.2, 0.1, 0.05], np.zeros(3)))
    ports = tuple(
        replace(port, points=companion) if port.role in {"parent", "child_1"} else port
        for port in transition.ports
    )
    transition = replace(transition, ports=ports)

    cut = cut_surface_at_transitions(surface, action, [transition])

    np.testing.assert_array_equal(cut.unresolved_transition_ids, [7])
    assert "doubles back" in cut.unresolved_transition_reasons[0]
    assert len(cut.cut_edges) == 0
    assert all(port.sheet_id == -1 for port in cut.ports)
    assert len(np.unique(cut.sheet_ids)) == 2


def test_inserted_cut_chains_survive_later_constraints():
    # A later constrained segment must neither flip nor split away an edge an
    # earlier chain claimed: the recorded cut path would silently reference a
    # destroyed edge and the cut would dangle. The later segment routes
    # around the existing chain instead.
    from alpha_analysis.j_connectivity.mesh_cut import (
        ConstrainedCutConfig,
        _MutableMesh,
    )

    surface, action = _surface_and_action()
    mesh = _MutableMesh(surface, action, None, ConstrainedCutConfig())
    first = mesh.insert_point(np.array([0.1, -0.15, 0.0]))
    second = mesh.insert_point(np.array([0.1, 0.15, 0.0]))
    chain = mesh.constrain_edge(first, second)
    chain_edges = {
        tuple(sorted((int(a), int(b)))) for a, b in zip(chain[:-1], chain[1:])
    }
    mesh.constrained_edges.update(chain_edges)

    # y = 0.06 crosses the first chain strictly inside one of its edges (the
    # chain's vertices sit near y = -0.15, 0, 0.025, 0.15), so the crossing
    # cannot sneak through a shared chain vertex.
    third = mesh.insert_point(np.array([-0.2, 0.06, 0.0]))
    fourth = mesh.insert_point(np.array([0.4, 0.06, 0.0]))
    crossing = mesh.constrain_edge(third, fourth)

    surviving = mesh.edges()
    assert all(edge in surviving for edge in chain_edges)
    assert all(
        tuple(sorted((int(a), int(b)))) in surviving
        for a, b in zip(crossing[:-1], crossing[1:])
    )


def test_refined_gamma_samples_stay_on_the_provenance_boundary():
    surface, action = _surface_and_action()
    transition = _transition()
    displaced = transition.marginal_points.copy()
    displaced[:, 1] += 0.005
    ports = tuple(
        replace(port, points=displaced) if port.role == "child_3" else port
        for port in transition.ports
    )
    transition = replace(transition, ports=ports, marginal_points=displaced)

    cut = cut_surface_at_transitions(surface, action, [transition])

    child_3 = next(port for port in cut.ports if port.role == "child_3")
    np.testing.assert_allclose(
        cut.surface.points[child_3.polyline_vertex_ids], displaced, atol=1.0e-14
    )
    assert np.all(
        (cut.surface.boundary_tags[child_3.polyline_vertex_ids] & SurfaceMesh.G_ZERO)
        != 0
    )


def test_before_after_cut_diagnostic_writes_png(tmp_path):
    surface, action = _surface_and_action()
    cut = cut_surface_at_transitions(surface, action, [_transition()])
    before = replace(
        cut,
        surface=surface,
        action_values=action,
        sheet_ids=_triangle_components(surface.triangles),
        cut_edges=np.empty((0, 2), dtype=np.int64),
        ports=(),
    )
    path = tmp_path / "cut.png"

    figure, axes = plot_cut_surface(before, cut, output_path=path)
    try:
        assert path.stat().st_size > 0
        assert "Before constrained cut" in axes[0].get_title()
        assert "After cut" in axes[1].get_title()
    finally:
        plt.close(figure)


@pytest.fixture(scope="module")
def production_synthetic_pipeline():
    """Analytic generic split taken through the full production pipeline.

    Shared by the finite-action acceptance test and the all-NaN
    side-assignment tests so the background mesh, extraction, transition
    mapping, and surface-wide traces are built once.
    """
    modes = np.array([0, 1, 2])
    cosine = np.array([[2.0, 0.0], [-1.0, 0.0], [0.3, 0.2]])
    field = SyntheticFourierField(
        nfp=1,
        m=np.zeros(3, dtype=np.int64),
        n=modes,
        cosine_coefficients=cosine,
        sine_coefficients=np.zeros_like(cosine),
        iota_coefficients=np.array([0.4]),
        G_coefficients=np.array([3.0]),
        I_coefficients=np.array([0.0]),
    )
    background = StructuredPrismMeshBackend(BackgroundMeshConfig(4, 16, 12)).build(
        field
    )
    extraction = MarchingTetrahedraExtractor().extract(background, field, 1.4)
    critical = extract_critical_curves(extraction, field, 1.4)
    transitions = map_transitions(
        field,
        critical,
        TransitionMappingConfig(
            max_curve_samples=8,
            action_quadrature_order=24,
            max_action_quadrature_order=48,
        ),
    )
    trace_config = WellTraceConfig(samples_per_field_period=96)
    data = evaluate_surface_data(extraction.incoming, field, trace_config)
    return field, extraction, transitions, data, trace_config


def test_production_synthetic_surface_has_no_uncut_action_jump(
    production_synthetic_pipeline,
):
    # Analytic generic split from test_transitions, now taken through the
    # production background mesh, B=b extraction, well traces, critical
    # curves, transition map, local projection, closed cut, and sheet graph.
    field, extraction, transitions, data, _ = production_synthetic_pipeline

    cut = cut_surface_at_transitions(
        extraction.incoming, data.action_length, transitions, field=field
    )

    assert len(transitions) == 1
    assert transitions[0].status is TransitionStatus.REGULAR
    assert cut.unresolved_transition_ids.size == 0
    assert len(np.unique(cut.sheet_ids)) == 3
    np.testing.assert_array_equal(
        cut.sheet_ids, _triangle_components(cut.surface.triangles)
    )
    assert len({port.sheet_id for port in cut.ports}) == 3
    by_role = {port.role: port for port in cut.ports}
    branch_gap = np.min(
        np.abs(by_role["parent"].action_values - by_role["child_1"].action_values)
    )
    triangle_actions = cut.action_values[cut.surface.triangles]
    assert np.all(np.ptp(triangle_actions, axis=1) < 0.5 * branch_gap)
    for port in cut.ports:
        np.testing.assert_allclose(
            cut.action_values[port.polyline_vertex_ids],
            port.action_values,
            rtol=0.0,
            atol=1.0e-12,
        )
        for vertex_id in np.unique(port.polyline_vertex_ids):
            incident = np.any(cut.surface.triangles == vertex_id, axis=1)
            assert np.any(cut.sheet_ids[incident] == port.sheet_id)
    points = cut.surface.points
    s = np.sum(points[:, :2] ** 2, axis=1)
    theta = np.arctan2(points[:, 1], points[:, 0])
    np.testing.assert_allclose(
        field.B(s, theta, points[:, 2]), cut.surface.level, atol=1.0e-9
    )


def test_nan_action_side_assignment_traces_probes_and_matches_finite_run(
    production_synthetic_pipeline,
):
    # With no finite pre-cut action anywhere (--no-actions), side assignment
    # must trace one probe well per side and reach the same parent/child
    # sheets as the finite-action run -- the §10.3 step-6 decision is made by
    # data either way, never by a coin flip (docs/STATUS.md hardening item).
    field, extraction, transitions, data, trace_config = production_synthetic_pipeline

    finite_cut = cut_surface_at_transitions(
        extraction.incoming, data.action_length, transitions, field=field
    )
    nan_cut = cut_surface_at_transitions(
        extraction.incoming,
        np.full(len(extraction.incoming.points), np.nan),
        transitions,
        field=field,
        trace_config=trace_config,
    )

    assert nan_cut.unresolved_transition_ids.size == 0
    finite_roles = {port.role: port for port in finite_cut.ports}
    nan_roles = {port.role: port for port in nan_cut.ports}
    for role in ("parent", "child_1", "child_3"):
        # Sheet labels may renumber between runs; the triangles they cover
        # may not. The identical deterministic mesh makes masks comparable.
        np.testing.assert_array_equal(
            finite_cut.sheet_ids == finite_roles[role].sheet_id,
            nan_cut.sheet_ids == nan_roles[role].sheet_id,
        )
        np.testing.assert_allclose(
            nan_cut.action_values[nan_roles[role].polyline_vertex_ids],
            nan_roles[role].action_values,
            rtol=0.0,
            atol=1.0e-12,
        )

    # Anchor the assignment absolutely, not merely consistently between the
    # two runs: next to T, the parent sheet's own surface action is the
    # parent well's A_W and the child sheet's is the child's A_1, so a
    # swapped assignment in BOTH runs cannot pass.
    cut_vertex_ids = set(map(int, np.unique(finite_cut.cut_edges)))
    parent_port = finite_roles["parent"]
    child_port = finite_roles["child_1"]
    for port, other in ((parent_port, child_port), (child_port, parent_port)):
        port_ids = set(map(int, port.polyline_vertex_ids))
        neighbor_values = []
        for triangle_id in np.flatnonzero(finite_cut.sheet_ids == port.sheet_id):
            triangle = finite_cut.surface.triangles[triangle_id]
            if not port_ids.intersection(map(int, triangle)):
                continue
            neighbor_values.extend(
                finite_cut.action_values[vertex]
                for vertex in map(int, triangle)
                if vertex not in cut_vertex_ids
                and np.isfinite(finite_cut.action_values[vertex])
            )
        assert neighbor_values
        residual_own = abs(np.mean(neighbor_values) - np.mean(port.action_values))
        residual_other = abs(np.mean(neighbor_values) - np.mean(other.action_values))
        assert residual_own < residual_other


def test_indecisive_side_assignment_is_demoted_not_guessed(
    production_synthetic_pipeline,
):
    # When the two candidate assignments' costs are separated by less than
    # the configured fraction of the parent/child action jump, the transition
    # must become an explicit unresolved hyperedge -- a coin-flip comparison
    # a single bad probe could decide is not a side assignment. The probe
    # costs here differ by about twice the jump, so a ratio of five demotes.
    field, extraction, transitions, _, trace_config = production_synthetic_pipeline

    cut = cut_surface_at_transitions(
        extraction.incoming,
        np.full(len(extraction.incoming.points), np.nan),
        transitions,
        field=field,
        config=ConstrainedCutConfig(side_assignment_margin_ratio=5.0),
        trace_config=trace_config,
    )

    np.testing.assert_array_equal(
        cut.unresolved_transition_ids, [transitions[0].transition_id]
    )
    assert "not decisive" in cut.unresolved_transition_reasons[0]
    assert all(port.sheet_id == -1 for port in cut.ports)
    # The withdrawn cut splits nothing: the incoming surface's own component
    # count is the entire sheet census.
    assert len(np.unique(cut.sheet_ids)) == len(
        np.unique(extraction.incoming.component_ids)
    )


def test_dmerc_reference_zigzag_curve_cuts_at_default_sampling():
    # ADR 0005 acceptance on the reference equilibrium: the GAMMA_MAX
    # mesh-edge chain doubles back on itself at sub-triangle scale, the
    # certified upstream reordering repairs it, and the constrained cut then
    # resolves end to end at the default sampling -- no tuned
    # max_curve_samples. Also pins that a sample vertex crossed by another
    # samples' tagged path keeps its authoritative u (port actions exact).
    field = BoozerField.from_boozmn(
        os.path.join(
            DATA_DIR,
            "boozmn_20260406-01-262-Ax_nfp4_Garabedian_mpol2_ntor2_minx0_allNfp_"
            "aspect10_DMercFail_m0p3_eval000323_low_resolution.nc",
        )
    )
    # Radially global refined bounds from docs/validation/milestone9-real-
    # equilibria.md; lambda_n = 0.8.
    b = 5.040465893072380 + 0.8 * (12.05034354020445 - 5.040465893072380)
    background = StructuredPrismMeshBackend(BackgroundMeshConfig(6, 24, 12)).build(
        field
    )
    extraction = MarchingTetrahedraExtractor().extract(background, field, b)
    critical = extract_critical_curves(extraction, field, b)

    assert critical.status is CriticalCurveStatus.REGULAR
    assert critical.report.reversal_repaired_count >= 2
    assert critical.report.reversal_unrepaired_count == 0

    transitions = map_transitions(
        field,
        critical,
        TransitionMappingConfig(
            max_curve_samples=10,
            action_quadrature_order=32,
            max_action_quadrature_order=512,
        ),
    )
    assert len(transitions) == 1
    assert transitions[0].status is TransitionStatus.REGULAR

    cut = cut_surface_at_transitions(
        extraction.incoming,
        np.full(len(extraction.incoming.points), np.nan),
        transitions,
        field=field,
    )

    assert cut.unresolved_transition_ids.size == 0
    assert len(np.unique(cut.sheet_ids)) == 2
    for port in cut.ports:
        assert port.sheet_id >= 0
        assert np.all(port.polyline_vertex_ids >= 0)
        np.testing.assert_allclose(
            cut.action_values[port.polyline_vertex_ids],
            port.action_values,
            rtol=0.0,
            atol=1.0e-12,
        )
    parent = next(port for port in cut.ports if port.role == "parent")
    child_1 = next(port for port in cut.ports if port.role == "child_1")
    assert parent.sheet_id != child_1.sheet_id
    triangle_actions = cut.action_values[cut.surface.triangles]
    finite_rows = np.all(np.isfinite(triangle_actions), axis=1)
    jump = np.min(np.abs(parent.action_values - child_1.action_values))
    assert np.all(np.ptp(triangle_actions[finite_rows], axis=1) < 0.5 * jump)
