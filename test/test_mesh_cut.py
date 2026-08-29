"""Constrained-cut topology tests (DESIGN.md §§10.3--10.5 and 23)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace

from alpha_analysis.j_connectivity import (
    BackgroundMeshConfig,
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
        action_values=np.full(n, action),
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
            _port("parent", companion, 3.0),
            _port("child_1", companion, 1.0),
            _port("child_3", marginal, 2.0),
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
    assert set(by_role) == {"parent", "child_1", "child_3"}
    assert len({port.sheet_id for port in cut.ports}) == 3
    for role, expected_action in (("parent", 3.0), ("child_1", 1.0), ("child_3", 2.0)):
        port = by_role[role]
        assert port.sheet_id >= 0
        assert np.all(port.polyline_vertex_ids >= 0)
        np.testing.assert_allclose(
            cut.action_values[port.polyline_vertex_ids], expected_action
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
    for triangle in cut.surface.triangles:
        values = cut.action_values[triangle]
        assert not (np.any(np.isclose(values, 1.0)) and np.any(np.isclose(values, 3.0)))


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


def test_underresolved_T_to_edge_strip_is_not_cut():
    surface, action = _surface_and_action()
    transition = _transition()
    companion = np.column_stack(([-0.4, 0.0, 0.4], np.full(3, 0.19), np.zeros(3)))
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


def test_production_synthetic_surface_has_no_uncut_action_jump():
    # Analytic generic split from test_transitions, now taken through the
    # production background mesh, B=b extraction, well traces, critical
    # curves, transition map, local projection, closed cut, and sheet graph.
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
    data = evaluate_surface_data(
        extraction.incoming,
        field,
        WellTraceConfig(samples_per_field_period=96),
    )

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
