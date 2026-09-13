"""R2 atlas acceptance checks against analytic field-line geometry (DESIGN §23)."""

import numpy as np
from dataclasses import replace
from scipy.integrate import quad
from scipy.optimize import brentq, root

from alpha_analysis.j_connectivity.branch_atlas import (
    AtlasConfig,
    build_atlas,
    classify_cell,
    height_at,
    transition_at,
    seam_image,
    identify_lifted_overlap,
    refine_atlas_cell,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField


def field(cosine, n, *, m=None, iota=0.0, C=3.0):
    cosine = np.asarray(cosine, dtype=float)
    return SyntheticFourierField(
        nfp=1,
        m=np.zeros(len(n), dtype=int) if m is None else np.asarray(m),
        n=np.asarray(n),
        cosine_coefficients=cosine,
        sine_coefficients=np.zeros_like(cosine),
        iota_coefficients=np.array([iota]),
        G_coefficients=np.array([C]),
        I_coefficients=np.array([0.0]),
    )


def test_atlas_counts_each_incoming_bounce_once():
    # B=2-cos(2z) has exactly two physical incoming roots per 2pi period.
    atlas = build_atlas(field([[2], [-1]], [0, 2]), 2.0, AtlasConfig(2, 4, 2))
    # The axis has one physical line despite several chart alpha labels.
    assert len(atlas.owned_wells) == 2 * (1 + 4)
    assert all(len(sample.wells) == 2 for sample in atlas.samples)
    assert all(
        cell.multiplicity_lower == cell.multiplicity_upper == 2 for cell in atlas.cells
    )
    assert all(cell.matched_local_branches == (0, 1) for cell in atlas.cells)
    assert np.isclose(sum(cell.area for cell in atlas.cells), 2 * np.pi)
    assert np.isclose(
        sum(cell.area * cell.multiplicity_lower for cell in atlas.cells), 4 * np.pi
    )


def test_multicover_chart_and_periodic_lifts_preserve_distinct_wells():
    atlas = build_atlas(
        field([[2], [-1]], [0, 2], iota=0.25), 2.0, AtlasConfig(2, 4, 2)
    )
    first = atlas.samples[0].wells
    assert len(first) == 2
    assert first[0].zeta_in != first[1].zeta_in
    assert first[0].branch_id != first[1].branch_id
    alpha = 0.3
    assert np.isclose(seam_image(alpha, 0.5, atlas.field, 1), alpha + np.pi / 2)
    assert np.isclose(seam_image(alpha, 0.5, atlas.field, -1), alpha - np.pi / 2)
    base = first[0]
    L = 2 * np.pi / atlas.field.nfp
    lifted = replace(
        base,
        alpha=seam_image(base.alpha, base.s, atlas.field, 1),
        zeta_in=base.zeta_in - L,
        zeta_out=base.zeta_out - L,
    )
    assert identify_lifted_overlap(atlas.field, base, lifted, 1)
    assert not identify_lifted_overlap(atlas.field, first[1], lifted, 1)
    reverse = build_atlas(field([[2], [-1]], [0, 2], C=-3), 2.0, AtlasConfig(2, 4, 2))
    assert all(w.zeta_out < w.zeta_in for w in reverse.owned_wells)
    assert all(c.multiplicity_upper == 2 for c in reverse.cells)


def test_height_gradient_keeps_radial_shear_term():
    s, alpha, z = 0.5, 0.3, 0.5
    iota = 0.5 + 0.2 * s
    a = 0.2
    c = -a * iota * np.sin(alpha + iota * z) / np.sin(z)
    f = SyntheticFourierField(
        1,
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([[2.0], [a], [c]]),
        np.zeros((3, 1)),
        np.array([0.5, 0.2]),
        np.array([3.0]),
        np.array([0.0]),
    )
    h = height_at(f, s, alpha, z)
    expected = 0.2 * z * (-a * np.sin(alpha + iota * z))
    np.testing.assert_allclose(h.gradient_s, expected, atol=1e-12)
    delta = 1e-5
    finite_difference = (
        height_at(f, s + delta, alpha, z).value
        - height_at(f, s - delta, alpha, z).value
    ) / (2 * delta)
    np.testing.assert_allclose(h.gradient_s, finite_difference, rtol=1e-6)


def test_below_b_fold_preserves_return_branch():
    from alpha_analysis.j_connectivity.forward_catalogue import (
        ForwardLineCatalogue,
        ForwardScanConfig,
    )

    # The cos(alpha)cos(12z) term changes the number of extrema inside a
    # trapped interval from three to five near alpha=1.62, but every new
    # maximum remains below b=1.4. A first-return branch must continue.
    f = field(
        [[2], [-1], [0.3], [0.015], [0.015]], [0, 1, 2, 12, -12], m=[0, 0, 0, 1, 1]
    )
    snapshots = []
    for alpha in (1.61, 1.63):
        scan = ForwardLineCatalogue(
            f, 0.5, alpha, -np.pi, ForwardScanConfig(max_periods=1)
        )
        scan.extend_to(1)
        query = scan.query(1.4)
        assert len(query.wells) == 1 and query.root_complete
        snapshots.append((query.wells[0], len(query.wells[0].extrema_u)))
    assert [count for _, count in snapshots] == [3, 5]
    assert all(well.zeta_in < 0 < well.zeta_out for well, _ in snapshots)
    assert abs(snapshots[0][0].zeta_out - snapshots[1][0].zeta_out) < 0.01
    cell = classify_cell(f, 1.4, (0.49, 0.51), (1.61, 1.63), periods=2, subdivisions=12)
    assert cell.multiplicity_lower == cell.multiplicity_upper == 1
    assert cell.matched_local_branches == (0,)


def test_barrier_crossing_creates_matched_additive_ports():
    # B=2-cos z+(0.3+0.2s)cos 2z has H(0)=1.3+0.2s.
    f = field([[2, 0], [-1, 0], [0.3, 0.2]], [0, 1, 2])
    h = height_at(f, 0.5, 0.2, 0.0)
    np.testing.assert_allclose(
        [h.value, h.gradient_s, h.gradient_alpha], [1.4, 0.2, 0.0], atol=1e-10
    )
    event = transition_at(f, 1.4, 0.5, 0.2, periods=2)
    assert event.status == "generic"
    assert len(event.ports) == 3
    assert {port.role for port in event.ports} == {"parent", "child_1", "child_3"}
    actions = {port.role: port.action_length for port in event.ports}
    ports = {port.role: port for port in event.ports}
    np.testing.assert_allclose(
        actions["parent"], actions["child_1"] + actions["child_3"], rtol=0, atol=1e-8
    )
    np.testing.assert_allclose(
        [ports["child_1"].zeta_out, ports["child_3"].zeta_in],
        [event.marginal_zeta[0]] * 2,
        atol=1e-10,
        rtol=0,
    )
    np.testing.assert_allclose(
        [ports["child_1"].zeta_in, ports["child_3"].zeta_out],
        [ports["parent"].zeta_in, ports["parent"].zeta_out],
        atol=1e-10,
        rtol=0,
    )
    a = np.arccos(0.25)
    expected = quad(
        lambda z: 3 / f.B(0.5, 0.2, z) * np.sqrt(max(0, 1 - f.B(0.5, 0.2, z) / 1.4)),
        -a,
        a,
        epsabs=1e-11,
    )[0]
    np.testing.assert_allclose(actions["parent"], expected, rtol=1e-8)
    # Independently solve ordinary roots on both sides of the height crossing.
    # The reported limiting actions must agree with these one-sided well actions.
    eps = 1e-5
    lower, upper = 1.4 - eps, 1.4 + eps

    def B(z):
        return float(f.B(0.5, 0.2, z))

    def action(pitch, left, right):
        return quad(
            lambda z: 3 / B(z) * np.sqrt(max(0, 1 - B(z) / pitch)),
            left,
            right,
            epsabs=1e-11,
        )[0]

    outer = brentq(lambda z: B(z) - lower, 1.0, 1.6)
    inner = brentq(lambda z: B(z) - lower, 0.0, 0.2)
    parent_outer = brentq(lambda z: B(z) - upper, 1.0, 1.6)
    np.testing.assert_allclose(
        [actions["child_1"], actions["child_3"], actions["parent"]],
        [
            action(lower, -outer, -inner),
            action(lower, inner, outer),
            action(upper, -parent_outer, parent_outer),
        ],
        atol=3e-4,
        rtol=0,
    )
    np.testing.assert_allclose(event.parameter, [0.5, 0.2], atol=0, rtol=0)
    assert all(port.event_parameter == event.parameter for port in event.ports)


def test_moving_root_certificate_retains_hidden_barriers():
    # At z=0, H(s,alpha)=1+s-s^2-.25 cos(2alpha). H=1.4 encloses
    # (s=.5,alpha=pi/2) wholly inside (0,1)x(0,pi), although all four
    # corners are below b. Along the s=.5 edge it crosses twice.
    f = field(
        [[2, 0, 0], [-1, 0, 0], [0, 1, -1], [-0.125, 0, 0], [-0.125, 0, 0]],
        [0, 1, 2, 2, -2],
        m=[0, 0, 0, 2, 2],
    )
    b = 1.4
    for s, alpha in ((0, 0), (1, 0), (0, np.pi), (1, np.pi)):
        assert f.B(s, alpha, 0) < b
    assert f.B(0.5, np.pi / 2, 0) > b
    root1 = 0.5 * np.arccos(-0.6)
    root2 = np.pi - root1
    np.testing.assert_allclose([f.B(0.5, a, 0) for a in (root1, root2)], b, atol=1e-12)
    loop_cell = classify_cell(f, b, (0, 1), (0, np.pi), periods=2)
    crossing_cell = classify_cell(f, b, (0.5, 0.8), (0, np.pi), periods=2)
    assert loop_cell.multiplicity_upper is None
    assert crossing_cell.multiplicity_upper is None
    assert loop_cell.unknown_reason and crossing_cell.unknown_reason
    # A shallow local maximum is shifted halfway between the 64 nodes per
    # period. At the center alpha, both adjacent samples lie below b while
    # the interior barrier lies above it. Across this small alpha cell the
    # barrier crosses b, changing one first-return well into two. Coalesced
    # scan brackets must retain the D-sign proof; a guessed sign would
    # incorrectly certify the entire cell as one well.
    shift = np.pi / 64
    cosine = np.array([[2.0], [np.cos(shift)], [0.3 * np.cos(2 * shift)], [0.5]])
    sine = np.array([[0.0], [-np.sin(shift)], [-0.3 * np.sin(2 * shift)], [0.0]])
    narrow = SyntheticFourierField(
        1,
        np.array([0, 0, 0, 1]),
        np.array([0, 1, 2, 0]),
        cosine,
        sine,
        np.array([0.0]),
        np.array([3.0]),
        np.array([0.0]),
    )
    narrow_b = 1.2999
    alpha_mid = np.pi / 2
    barrier_z = np.pi + shift
    assert all(
        narrow.B(0.5, alpha_mid, z) < narrow_b for z in (np.pi, np.pi + np.pi / 32)
    )
    assert narrow.B(0.5, alpha_mid, barrier_z) > narrow_b
    assert narrow.B(0.5, alpha_mid - 5e-4, barrier_z) > narrow_b
    assert narrow.B(0.5, alpha_mid + 5e-4, barrier_z) < narrow_b
    narrow_cell = classify_cell(
        narrow,
        narrow_b,
        (0.4, 0.6),
        (alpha_mid - 5e-4, alpha_mid + 5e-4),
        periods=2,
    )
    assert narrow_cell.multiplicity_upper is None
    assert "hidden barrier" in narrow_cell.unknown_reason
    # An extrema fold wholly below b does not change the first-return well.
    harmless = field(
        [[2], [-1], [0.3], [0.015], [0.015]],
        [0, 1, 2, 12, -12],
        m=[0, 0, 0, 1, 1],
    )
    regular = classify_cell(
        harmless, 1.4, (0.49, 0.51), (1.61, 1.63), periods=2, subdivisions=12
    )
    assert regular.multiplicity_lower == regular.multiplicity_upper == 1


def test_unknown_multiplicity_remains_in_coverage_bound():
    f = field(
        [[2, 0, 0], [-1, 0, 0], [0, 1, -1], [-0.125, 0, 0], [-0.125, 0, 0]],
        [0, 1, 2, 2, -2],
        m=[0, 0, 0, 2, 2],
    )
    atlas = build_atlas(f, 1.4, AtlasConfig(2, 4, 2))
    assert not np.isclose(f.B(0, 0, 0), f.B(0, np.pi / 2, 0))
    assert len(atlas.owned_wells) == sum(len(sample.wells) for sample in atlas.samples)
    assert atlas.unknown_area > 0
    assert atlas.multiplicity_area_upper is None
    assert np.isclose(sum(cell.area for cell in atlas.cells), 2 * np.pi)
    assert np.isclose(
        atlas.unknown_area,
        sum(cell.area for cell in atlas.cells if cell.multiplicity_upper is None),
    )


def test_refinement_preserves_unknown_complement_and_owned_area():
    f = field([[2, 0], [-1, 0], [0.3, 0.2]], [0, 1, 2])
    atlas = build_atlas(f, 1.4, AtlasConfig(2, 2, 2))
    assert atlas.unknown_area == 2 * np.pi
    refined = refine_atlas_cell(atlas, 0, (0.2, 0.21), (0.2, 0.21))
    assert np.isclose(sum(c.area for c in refined.cells), 2 * np.pi)
    target = next(
        c
        for c in refined.cells
        if c.s_interval == (0.2, 0.21) and c.alpha_interval == (0.2, 0.21)
    )
    assert target.multiplicity_upper == 1
    assert target.matched_local_branches == (0,)
    assert refined.known_owned_area >= target.area
    assert np.isclose(
        refined.unknown_area,
        sum(c.area for c in refined.cells if c.multiplicity_upper is None),
    )
    assert refined.unknown_area <= 2 * np.pi - target.area
    assert refined.multiplicity_area_upper is None


def test_six_well_equal_height_event_is_enclosed_as_multiway():
    # Equal-height lower barriers in cos(3z)+0.2cos z give three elementary
    # segments and six distinct contiguous limiting wells.
    f = field([[2], [1], [0.2]], [0, 3, 1])
    z = np.arccos(-np.sqrt(7 / 30))
    b = float(f.B(0.5, 0.0, z))
    event = transition_at(f, b, 0.5, 0.0, periods=2)
    assert event.status == "multiway_unknown"
    assert len(event.ports) == 6
    assert len({(p.zeta_in, p.zeta_out) for p in event.ports}) == 6


def test_dmerc_reference_has_physical_generic_ports():
    from pathlib import Path
    from alpha_analysis import BoozerField

    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "boozmn_20260406-01-262-Ax_nfp4_Garabedian_mpol2_ntor2_minx0_allNfp_aspect10_DMercFail_m0p3_eval000323_low_resolution.nc"
    )
    f = BoozerField.from_boozmn(path)
    b, s = 10.648368010778035, 0.8  # R0 radially global 0.8 reference pitch

    def equations(q):
        return [float(f.B(s, q[0], q[1])) - b, float(f.D_B(s, q[0], q[1]))]

    solution = root(equations, [-0.145, 0.0133], tol=1e-11)
    assert np.max(np.abs(equations(solution.x))) < 1e-9
    theta, z = solution.x
    assert float(f.D2_B(s, theta, z)) < 0
    event = transition_at(f, b, s, theta - float(f.iota(s)) * z, periods=4)
    assert event.status == "generic"
    assert len(event.ports) == 3
    np.testing.assert_allclose(event.marginal_zeta, [z], atol=1e-8)
    actions = {p.role: p.action_length for p in event.ports}
    ports = {p.role: p for p in event.ports}
    np.testing.assert_allclose(
        [ports["child_1"].zeta_out, ports["child_3"].zeta_in],
        [z, z],
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(
        [ports["child_1"].zeta_in, ports["child_3"].zeta_out],
        [ports["parent"].zeta_in, ports["parent"].zeta_out],
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(
        actions["parent"], actions["child_1"] + actions["child_3"], atol=1e-7, rtol=0
    )
    alpha = theta - float(f.iota(s)) * z
    height = height_at(f, s, alpha, z)
    delta = 1e-5
    numerical_gradient = (
        height_at(f, s + delta, alpha, z).value
        - height_at(f, s - delta, alpha, z).value
    ) / (2 * delta)
    np.testing.assert_allclose(height.gradient_s, numerical_gradient, rtol=2e-5)


def test_real_spline_cell_has_one_certified_owned_branch():
    from pathlib import Path
    from alpha_analysis import BoozerField

    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "boozmn_20260406-01-262-Ax_nfp4_Garabedian_mpol2_ntor2_minx0_allNfp_aspect10_DMercFail_m0p3_eval000323_low_resolution.nc"
    )
    f = BoozerField.from_boozmn(path)
    b = 5.040465893072380 + 0.5 * (12.050343540204448 - 5.040465893072380)
    coarse = classify_cell(f, b, (0.4, 0.401), (0, 0.01), periods=2, subdivisions=12)
    fine = classify_cell(f, b, (0.4, 0.4001), (0, 0.001), periods=2, subdivisions=12)
    assert coarse.multiplicity_upper == fine.multiplicity_upper == 1
    assert coarse.matched_local_branches == fine.matched_local_branches == (0,)


def test_root_certificate_survives_scan_node_crossing():
    """A translated well keeps one owned entry across arbitrary scan nodes (§23 R3.5)."""
    f = field([[2], [1]], [0, 1], m=[0, 1])
    for alpha_center in (0.0, 0.03):
        for width in (1e-3, 1e-7):
            cell = classify_cell(
                f,
                2.0,
                (0.4, 0.6),
                (alpha_center - width / 2, alpha_center + width / 2),
                periods=2,
                subdivisions=8,
            )
            assert cell.multiplicity_lower == cell.multiplicity_upper == 1, (
                alpha_center,
                width,
                cell.unknown_reason,
            )
            assert cell.matched_local_branches == (0,)
    reverse = field([[2], [1]], [0, 1], m=[0, 1], C=-3)
    reverse_cell = classify_cell(reverse, 2.0, (0.4, 0.6), (-5e-4, 5e-4), periods=2)
    assert reverse_cell.multiplicity_lower == reverse_cell.multiplicity_upper == 1
    from alpha_analysis.j_connectivity.branch_atlas import _line, _owned

    forward_well = _owned(_line(f, 0.5, 0.0, 2), 2.0, False)[0][0]
    reverse_well = _owned(_line(reverse, 0.5, 0.0, 2), 2.0, False)[0][0]
    np.testing.assert_allclose(
        [forward_well.zeta_in, forward_well.zeta_out],
        [np.pi / 2, 3 * np.pi / 2],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        [reverse_well.zeta_in, reverse_well.zeta_out],
        [3 * np.pi / 2, np.pi / 2],
        atol=1e-9,
    )
