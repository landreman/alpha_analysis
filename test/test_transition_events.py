"""Localized nongeneric contacts (DESIGN.md §§5.4 and 23, milestone 10.2)."""

import numpy as np
import pytest
from scipy.optimize import root

from alpha_analysis.j_connectivity import map_transitions, TransitionMappingConfig
from alpha_analysis.j_connectivity.transition_events import (
    localize_transition_contacts,
    build_transition_arcs,
)
from test_transitions import _critical_circles, _stepped_over_contact_field


def test_localized_contacts_preserve_the_two_equal_height_maxima():
    # Independent equations for the second maximum of the analytic field in
    # test_transitions: B(0.5, theta, zeta)=1.6 and dB/dzeta=0.
    # Neither critical-curve interpolation nor the transition mapper is used
    # to obtain the expected contact angle.
    def equations(values):
        zeta, delta = values
        return [
            2
            - np.cos(zeta)
            + 0.6 * np.cos(2 * zeta)
            + delta * (np.sin(6 * zeta) - 6 * np.sin(zeta))
            - 1.6,
            np.sin(zeta)
            - 1.2 * np.sin(2 * zeta)
            + 6 * delta * (np.cos(6 * zeta) - np.cos(zeta)),
        ]

    exact = root(equations, [-0.6, 0.0543], tol=1.0e-12)
    assert np.max(np.abs(equations(exact.x))) < 1.0e-12
    assert 0.0530 - 0.0045 < exact.x[1] < 0.0530 + 0.0045
    theta = np.arccos((exact.x[1] - 0.0530) / 0.0045)
    field = _stepped_over_contact_field()
    critical = _critical_circles(field, s=0.5, zeta_values=(0.0,), count=8)
    source = map_transitions(field, critical, TransitionMappingConfig())
    result = localize_transition_contacts(field, critical, source)
    assert len(result.events) == 2
    angles = sorted(
        np.mod(
            np.arctan2(event.marginal_points[0, 1], event.marginal_points[0, 0]),
            2 * np.pi,
        )
        for event in result.events
    )
    np.testing.assert_allclose(angles, [theta, 2 * np.pi - theta], atol=1.0e-8)
    for event in result.events:
        assert len(event.marginal_points) == 2
        s = np.sum(event.marginal_points[:, :2] ** 2, axis=1)
        theta_values = np.arctan2(
            event.marginal_points[:, 1], event.marginal_points[:, 0]
        )
        zeta_values = event.marginal_points[:, 2]
        np.testing.assert_allclose(s, 0.5, atol=1.0e-10)
        np.testing.assert_allclose(
            field.B(s, theta_values, zeta_values), 1.6, atol=1.0e-10
        )
        np.testing.assert_allclose(
            field.D_B(s, theta_values, zeta_values), 0, atol=1.0e-9
        )
        assert np.all(field.D2_B(s, theta_values, zeta_values) < 0)
        assert (
            event.unresolved
        ), "localizing a contact does not resolve its multiway connectivity"


@pytest.fixture(scope="module")
def two_barrier_pipeline():
    from alpha_analysis.j_connectivity import (
        BackgroundMeshConfig,
        StructuredPrismMeshBackend,
        MarchingTetrahedraExtractor,
        extract_critical_curves,
    )
    from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField

    # The lower maxima of cos(3*zeta)+0.2*cos(zeta) lie at
    # cos(zeta)=-sqrt(7/30). The sin(theta)*sin(zeta) perturbation makes
    # their heights cross at theta=0,pi, s=0.5. The higher maximum near
    # zeta=0 bounds a parent well containing both lower maxima.
    field = SyntheticFourierField(
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
    zeta = np.arccos(-np.sqrt(7 / 30))
    b = float(field.B(0.5, 0, zeta))
    background = StructuredPrismMeshBackend(BackgroundMeshConfig(4, 16, 36)).build(
        field
    )
    extraction = MarchingTetrahedraExtractor().extract(background, field, b)
    critical = extract_critical_curves(extraction, field, b)
    source = map_transitions(field, critical)
    localized = localize_transition_contacts(field, critical, source)
    return field, extraction, critical, localized


def test_contact_arcs_share_physical_events_and_have_one_sided_actions(
    two_barrier_pipeline,
):
    field, extraction, critical, localized = two_barrier_pipeline
    result = build_transition_arcs(field, critical, localized)
    assert len(result.events) == 2
    assert len(result.arcs) == 4
    assert all(len(event.occurrences) == 2 for event in result.events)
    for arc in result.arcs:
        assert all(index >= 0 for index in arc.endpoint_event_ids)
        curve = arc.curve
        parent, first, third = curve.ports
        np.testing.assert_allclose(
            parent.action_values,
            first.action_values + third.action_values,
            atol=1.0e-9,
            rtol=0,
        )
        # All endpoint limits are at the analytic simultaneous maxima, and
        # every port remains on that same field line (iota=0 here).
        for end in (0, -1):
            point = curve.marginal_points[end]
            np.testing.assert_allclose(np.sum(point[:2] ** 2), 0.5, atol=1.0e-9)
            assert abs(point[1]) < 1.0e-8
            for port in curve.ports:
                np.testing.assert_allclose(port.points[end, :2], point[:2], atol=1.0e-9)


def test_regular_arcs_cut_into_six_wells_without_dangling_event_ends(
    two_barrier_pipeline,
):
    from alpha_analysis.j_connectivity.mesh_cut import cut_surface_at_transition_arcs

    field, extraction, critical, localized = two_barrier_pipeline
    arrangement = build_transition_arcs(field, critical, localized)
    cut = cut_surface_at_transition_arcs(
        extraction.incoming,
        np.full(len(extraction.incoming.points), np.nan),
        arrangement,
        field=field,
    )
    # Two marginal barriers inside an ordinary [a,d] well have exactly six
    # limiting trapped branches: [a,m1], [m1,m2], [m2,d], [a,m2], [m1,d],
    # and [a,d]. Both contacts join those six incident sheets; they must not
    # be four independent three-port events or disconnected dangling slits.
    assert len(np.unique(cut.sheet_ids)) == 6
    assert len(cut.unresolved_transition_ids) == 0
    assert len(cut.events) == 2
    assert all(port.sheet_id >= 0 for port in cut.ports)
    for event in cut.events:
        assert event.unresolved
        assert len({cut.ports[index].sheet_id for index in event.port_indices}) == 6
    for port in cut.ports:
        np.testing.assert_allclose(
            cut.action_values[port.polyline_vertex_ids], port.action_values, atol=1.0e-9
        )
    for arc in arrangement.arcs:
        ports = {
            port.role: port
            for port in cut.ports
            if port.transition_id == arc.curve.transition_id
        }
        parent = set(ports["parent"].polyline_vertex_ids)
        child = set(ports["child_1"].polyline_vertex_ids)
        assert parent.isdisjoint(child)
        assert not any(
            parent.intersection(triangle) and child.intersection(triangle)
            for triangle in cut.surface.triangles
        )
