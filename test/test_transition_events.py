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


def test_sampled_equal_height_contact_keeps_an_explicit_event():
    from dataclasses import replace
    from alpha_analysis.j_connectivity import TransitionStatus

    field = _stepped_over_contact_field()

    def equations(values):
        theta, zeta = values
        return [
            float(field.B(0.5, theta, zeta)) - 1.6,
            float(field.D_B(0.5, theta, zeta)),
        ]

    solution = root(equations, [1.3, -0.6], tol=1.0e-12)
    assert np.max(np.abs(equations(solution.x))) < 1.0e-12
    critical = _critical_circles(
        field, s=0.5, zeta_values=(0.0,), count=8, theta_offset=solution.x[0]
    )
    source = map_transitions(field, critical)
    assert source[0].sample_status[0] is TransitionStatus.MULTIWAY
    localized = localize_transition_contacts(field, critical, source)
    sampled = [
        event
        for event in localized.events
        if any(
            occurrence.source_sample_pair == (0, 0) for occurrence in event.occurrences
        )
    ]
    assert len(sampled) == 1
    assert sampled[0].unresolved
    assert len(sampled[0].marginal_points) == 2
    point = sampled[0].marginal_points[1]
    np.testing.assert_allclose(
        field.B(np.sum(point[:2] ** 2), np.arctan2(point[1], point[0]), point[2]),
        1.6,
        atol=1.0e-10,
    )
    assert all(np.isnan(port.action_values[0]) for port in source[0].ports)
    # The same physical sample can terminate an open source curve. Its u
    # remains the upper endpoint; it must not wrap to zero or create a
    # zero-length incident arc.
    polyline = critical.polylines[0]
    open_curve = replace(
        polyline,
        vertex_ids=polyline.vertex_ids[::-1],
        u=polyline.u.copy(),
        total_length=float(polyline.u[-1]),
        closed=False,
    )
    open_critical = replace(critical, polylines=(open_curve,))
    open_source = map_transitions(field, open_critical)
    open_localized = localize_transition_contacts(field, open_critical, open_source)
    endpoint_event = next(
        event
        for event in open_localized.events
        if any(item.source_sample_pair == (7, 7) for item in event.occurrences)
    )
    arcs = build_transition_arcs(field, open_critical, open_localized).arcs
    incident = [
        arc for arc in arcs if endpoint_event.event_id in arc.endpoint_event_ids
    ]
    assert len(incident) == 1
    assert incident[0].endpoint_event_ids[1] == endpoint_event.event_id
    assert incident[0].source_interval[1] == open_curve.total_length


def test_scan_alias_brackets_dissolve_without_inventing_physical_events():
    from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField
    from alpha_analysis.j_connectivity import TransitionStatus
    from scipy.integrate import quad

    # At s=1/2, b-B = W*(0.8-W*cos(18*zeta+theta)), where
    # W=(1-cos(zeta))*(cos(zeta)-1/4). Since 0<W<=9/64<0.8,
    # the two ordinary crossings stay at +/-acos(1/4) for every theta;
    # no internal maximum can cross b. Coarse scans miss the small extrema.
    polynomial = np.polynomial.polynomial.polymul([1, -2, 1], [1 / 16, -1 / 2, 1])
    coefficients = np.polynomial.chebyshev.poly2cheb(polynomial)
    modes_m = [0, 0, 0]
    modes_n = [0, 1, 2]
    cosine = [[2, 0], [-1, 0], [0.3, 0.2]]
    for j, value in enumerate(coefficients):
        for n, weight in (
            [(18, value)] if j == 0 else [(18 - j, value / 2), (18 + j, value / 2)]
        ):
            modes_m.append(1)
            modes_n.append(-n)
            cosine.append([weight, 0])
    field = SyntheticFourierField(
        nfp=1,
        m=np.array(modes_m),
        n=np.array(modes_n),
        cosine_coefficients=np.array(cosine),
        sine_coefficients=np.zeros_like(cosine),
        iota_coefficients=np.array([0.0]),
        G_coefficients=np.array([3.0]),
        I_coefficients=np.array([0.0]),
    )
    critical = _critical_circles(field, s=0.5, zeta_values=(0.0,), count=8)
    controls = TransitionMappingConfig(
        samples_per_field_period=8, samples_per_wavelength=2
    )
    source = map_transitions(field, critical, controls)
    assert len(source[0].contact_sample_pairs) == 4
    localized = localize_transition_contacts(field, critical, source)
    assert len(localized.events) == 0
    assert len(localized.scan_artifacts) == 4
    arrangement = build_transition_arcs(field, critical, localized)
    assert len(arrangement.arcs) == 1
    arc = arrangement.arcs[0].curve
    assert arc.status is TransitionStatus.REGULAR and arc.sampling_certified
    np.testing.assert_array_equal(arc.interior_maximum_count, 2)
    a = np.arccos(0.25)
    for index in (0, len(arc.u) // 2):
        theta = np.arctan2(arc.marginal_points[index, 1], arc.marginal_points[index, 0])

        def integrand(zeta):
            B = float(field.B(0.5, theta, zeta))
            return 3 / B * np.sqrt(max(0, 1 - B / 1.4))

        expected = (
            quad(integrand, -a, 0, epsabs=1e-11, epsrel=1e-11)[0]
            + quad(integrand, 0, a, epsabs=1e-11, epsrel=1e-11)[0]
        )
        np.testing.assert_allclose(
            arc.ports[0].action_values[index], expected, rtol=1e-9, atol=1e-10
        )


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
    np.testing.assert_allclose(
        sum(arc.curve.total_u_length for arc in result.arcs),
        sum(source.total_u_length for source in localized.transitions),
        atol=1e-12,
        rtol=0,
    )
    for arc in result.arcs:
        assert all(index >= 0 for index in arc.endpoint_event_ids)
        curve = arc.curve
        np.testing.assert_allclose(curve.u, arc.source_u - arc.source_u[0], atol=1e-12)
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

    # A source certification cannot authorize a newly discovered itinerary
    # change inside an event arc. Inject an extra interior maximum at a source
    # sample away from all contact probes, without inventing another event.
    from dataclasses import replace
    from alpha_analysis.j_connectivity import TransitionStatus

    source = localized.transitions[0]
    assert source.sampling_certified
    probed_u = {
        float(sample.u[0]) % source.total_u_length
        for event in localized.events
        for occurrence in event.occurrences
        if occurrence.source_transition_id == source.transition_id
        for sample in occurrence.samples
    }
    index = next(i for i, u in enumerate(source.u) if u not in probed_u)
    counts = source.interior_maximum_count.copy()
    counts[index] += 1
    changed = replace(
        localized,
        transitions=(
            replace(source, interior_maximum_count=counts),
            *localized.transitions[1:],
        ),
    )
    guarded = build_transition_arcs(field, critical, changed)
    rejected = [
        arc
        for arc in guarded.arcs
        if arc.unresolved_reason
        == "unexplained interior-maximum count change within arc"
    ]
    assert rejected
    assert all(
        arc.curve.status is TransitionStatus.UNRESOLVED
        and not arc.curve.sampling_certified
        for arc in rejected
    )


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
    assert cut.max_corridor_faces_used <= 64
    assert cut.unresolved_action_flux(field) > 0
    _assert_cut_endpoints_have_boundary_or_event_provenance(cut)
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


def test_regular_arc_certification_survives_a_source_contact_stop(two_barrier_pipeline):
    field, extraction, critical, _ = two_barrier_pipeline
    sources = map_transitions(
        field, critical, TransitionMappingConfig(max_curve_samples=64)
    )
    assert any(not curve.sampling_certified for curve in sources)
    localized = localize_transition_contacts(field, critical, sources)
    arrangement = build_transition_arcs(field, critical, localized)
    interrupted = {
        curve.transition_id for curve in sources if not curve.sampling_certified
    }
    # An event interrupts certification of the whole curve, not every
    # continuous arc. At least one independently certifiable arc must survive.
    assert any(
        arc.curve.sampling_certified and arc.source_transition_id in interrupted
        for arc in arrangement.arcs
    )
    for source in sources:
        assert (
            source.sampling_samples_used
            + sum(
                arc.additional_source_samples
                for arc in arrangement.arcs
                if arc.source_transition_id == source.transition_id
            )
            <= 64
        )


def test_uncut_incident_arc_preserves_distinct_event_limits(two_barrier_pipeline):
    from dataclasses import replace
    from alpha_analysis.j_connectivity import TransitionStatus
    from alpha_analysis.j_connectivity.mesh_cut import cut_surface_at_transition_arcs

    field, extraction, critical, localized = two_barrier_pipeline
    arrangement = build_transition_arcs(field, critical, localized)
    # Keep the missing connection explicit. With one incident arc unresolved,
    # distinct limiting wells may still share a mesh vertex at the event.
    first, *others = arrangement.arcs
    first = replace(
        first,
        curve=replace(first.curve, status=TransitionStatus.UNRESOLVED),
        unresolved_reason="incident arc not resolved",
    )
    arrangement = replace(arrangement, arcs=(first, *others))
    cut = cut_surface_at_transition_arcs(
        extraction.incoming,
        np.full(len(extraction.incoming.points), np.nan),
        arrangement,
        field=field,
    )
    conflicts = _assert_port_limits_are_preserved_or_explicitly_unknown(cut)
    assert conflicts
    # Isolate each unknown event vertex from the other missing surface data.
    # Here iota=0, so |ds wedge d alpha|=2|dx wedge dy|: each triangle's
    # measure is the absolute 2D determinant, independently of surface_flux.
    # A single NaN must retain every incident triangle, including triangles
    # whose other two vertices have known action.
    for vertex in conflicts:
        action = np.ones(len(cut.action_values))
        action[vertex] = np.nan
        isolated = replace(
            cut,
            action_values=action,
            unresolved_event_action_vertex_ids=np.array([vertex]),
        )
        incident = [
            triangle for triangle in cut.surface.triangles if vertex in triangle
        ]
        xy = cut.surface.points[np.asarray(incident), :2]
        first = xy[:, 1] - xy[:, 0]
        second = xy[:, 2] - xy[:, 0]
        expected = np.sum(
            np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
        )
        assert expected > 0
        np.testing.assert_allclose(
            isolated.unresolved_action_flux(field), expected, rtol=1e-12, atol=1e-14
        )
        known = replace(
            isolated,
            action_values=np.ones(len(action)),
            unresolved_event_action_vertex_ids=np.empty(0, dtype=np.int64),
        )
        assert known.unresolved_action_flux(field) == 0


def _assert_port_limits_are_preserved_or_explicitly_unknown(cut):
    by_vertex = {}
    for port in cut.ports:
        if port.sheet_id < 0:
            continue
        for vertex, value in zip(port.polyline_vertex_ids, port.action_values):
            by_vertex.setdefault(int(vertex), []).append(value)
    conflicts = set()
    for vertex, values in by_vertex.items():
        if np.ptp(values) > 1e-8:
            conflicts.add(vertex)
            assert np.isnan(cut.action_values[vertex]), values
        else:
            np.testing.assert_allclose(cut.action_values[vertex], values, atol=1e-8)
    assert conflicts == set(cut.unresolved_event_action_vertex_ids)
    return conflicts


@pytest.fixture(scope="module")
def w7x_event_pipeline():
    from pathlib import Path
    from alpha_analysis import BoozerField
    from alpha_analysis.j_connectivity import (
        BackgroundMeshConfig,
        StructuredPrismMeshBackend,
        MarchingTetrahedraExtractor,
        extract_critical_curves,
    )

    field = BoozerField.from_boozmn(
        str(
            Path(__file__).parents[1]
            / "data"
            / "boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
        )
    )
    b = 2.7781394
    background = StructuredPrismMeshBackend(BackgroundMeshConfig(6, 24, 12)).build(
        field
    )
    extraction = MarchingTetrahedraExtractor().extract(background, field, b)
    critical = extract_critical_curves(extraction, field, b)
    source = map_transitions(field, critical)
    localized = localize_transition_contacts(field, critical, source)
    arrangement = build_transition_arcs(field, critical, localized)
    return field, extraction, source, arrangement


def _assert_cut_endpoints_have_boundary_or_event_provenance(cut):
    from alpha_analysis.j_connectivity.surface_extract import SurfaceMesh

    authorized = set(np.flatnonzero(cut.surface.boundary_tags & SurfaceMesh.EDGE))
    for event in cut.events:
        authorized.update(
            int(cut.ports[p].polyline_vertex_ids[s])
            for p, s in zip(event.port_indices, event.sample_indices)
        )
    edges = set(tuple(sorted(map(int, edge))) for edge in cut.cut_edges)
    vertices, degree = np.unique(np.asarray(list(edges)).ravel(), return_counts=True)
    assert all(
        int(vertex) in authorized
        for vertex, count in zip(vertices, degree)
        if count != 2
    )


@pytest.mark.slow
def test_w7x_reference_four_contacts_cut_regular_arcs_with_explicit_events(
    w7x_event_pipeline, tmp_path
):
    from alpha_analysis.j_connectivity.mesh_cut import (
        cut_surface_at_transition_arcs,
        save_cut_surface,
        load_cut_surface,
    )

    field, extraction, source, arrangement = w7x_event_pipeline
    assert sum(len(curve.contact_sample_pairs) for curve in source) == 4
    assert len(arrangement.events) == 2
    assert all(len(event.occurrences) == 2 for event in arrangement.events)
    # Unlike the synthetic iota=0 chart, this checks actual helical lifts
    # across the periodic seam against every port's stored physical point.
    for arc in arrangement.arcs:
        curve = arc.curve
        s, alpha = curve.field_line_identity.T
        for port in curve.ports:
            theta = alpha + field.iota(s) * port.zeta_unwrapped
            np.testing.assert_allclose(
                port.points[:, 0], np.sqrt(s) * np.cos(theta), atol=1e-8
            )
            np.testing.assert_allclose(
                port.points[:, 1], np.sqrt(s) * np.sin(theta), atol=1e-8
            )
    cut = cut_surface_at_transition_arcs(
        extraction.incoming,
        np.full(len(extraction.incoming.points), np.nan),
        arrangement,
        field=field,
    )
    resolved = {port.transition_id for port in cut.ports if port.sheet_id >= 0}
    assert len(resolved) >= 2
    assert len(cut.events) == 2
    _assert_cut_endpoints_have_boundary_or_event_provenance(cut)
    assert _assert_port_limits_are_preserved_or_explicitly_unknown(cut)
    for event in cut.events:
        assert event.unresolved
        assert len(event.port_indices) > 3
    for transition_id in cut.unresolved_transition_ids:
        ports = [p for p in cut.ports if p.transition_id == transition_id]
        assert len(ports) == 3 and all(p.sheet_id == -1 for p in ports)
    path = tmp_path / "events.npz"
    save_cut_surface(path, cut)
    restored = load_cut_surface(path)
    np.testing.assert_array_equal(restored.sheet_ids, cut.sheet_ids)
    np.testing.assert_array_equal(restored.action_values, cut.action_values)
    np.testing.assert_array_equal(
        restored.unresolved_event_action_vertex_ids,
        cut.unresolved_event_action_vertex_ids,
    )
    for original, event in zip(cut.events, restored.events):
        assert original.event_id == event.event_id
        assert original.unresolved == event.unresolved
        np.testing.assert_array_equal(original.marginal_points, event.marginal_points)
        np.testing.assert_array_equal(original.port_indices, event.port_indices)
        np.testing.assert_array_equal(original.sample_indices, event.sample_indices)
    assert len(restored.events) == len(cut.events)
