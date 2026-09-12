"""R3 direct-contour acceptance tests against analytic field geometry (§23)."""

import numpy as np
from pathlib import Path

from alpha_analysis import BoozerField, DATA_DIR

from alpha_analysis.j_connectivity.contour_trace import (
    ContourConfig,
    DirectContourOracle,
    ContourStatus,
)
from alpha_analysis.j_connectivity.branch_atlas import transition_at
from alpha_analysis.j_connectivity.forward_catalogue import (
    ForwardLineCatalogue,
    ForwardScanConfig,
    batched_bounce_integrals,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField


def field(radial=(0.0,), alpha_cos=0.0, *, iota=(0.0,), G=(3.0,), I=(0.0,)):
    """B=2-cos(z)+radial(s)+alpha_cos*cos(theta), in field units."""
    degree = max(len(radial), 1)
    coeff = np.zeros((3, degree))
    coeff[0, 0] = 2
    coeff[0, : len(radial)] += radial
    coeff[1, 0] = -1
    coeff[2, 0] = alpha_cos
    return SyntheticFourierField(
        1,
        np.array([0, 0, 1]),
        np.array([0, 1, 0]),
        coeff,
        np.zeros_like(coeff),
        np.asarray(iota),
        np.asarray(G),
        np.asarray(I),
    )


def test_direct_contour_closes_without_edge():
    # B=2-cos(z)+0.8(s-.5)^2+0.08(1-cos alpha).
    f = field((0.28, -0.8, 0.8), -0.08)
    oracle = DirectContourOracle(f, 2.0, ContourConfig(step=0.025, max_steps=500))
    seed = oracle.seed(0.6, 0.0)
    result = oracle.query(seed)
    assert result.status is ContourStatus.INACCESSIBLE, result.reason
    assert len(result.paths) >= 1
    assert result.paths[0].closed
    assert max(p.s for path in result.paths for p in path.points) < 0.8
    assert (
        max(
            abs(p.action_length - seed.action_length)
            for path in result.paths
            for p in path.points
        )
        < 2e-5
    )


def test_direct_contour_reaches_edge_with_constant_action():
    f = field((0.0, 0.15), 0.1)
    oracle = DirectContourOracle(f, 2.0, ContourConfig(step=0.025, max_steps=300))
    seed = oracle.seed(0.5, 0.0)
    result = oracle.query(seed)
    assert result.status is ContourStatus.ACCESSIBLE, result.reason
    witness = result.witness
    assert witness is not None
    np.testing.assert_allclose(witness.points[-1].s, 1, atol=1e-8)
    assert max(abs(p.action_length - seed.action_length) for p in witness.points) < 2e-5


def test_direct_contour_branches_at_same_event_parameter():
    # The Γmax curve is transverse to child A-contours: a child can continue
    # at its new action only if the marginal root is approached from its side.
    coeff = np.array([[2, 0], [-1, 0], [0.3, 0.25], [0.1, 0]])
    f = SyntheticFourierField(
        1,
        np.array([0, 0, 0, 1]),
        np.array([0, 1, 2, 0]),
        coeff,
        np.zeros_like(coeff),
        np.array([0.0]),
        np.array([3.0]),
        np.array([0.0]),
    )
    alpha = 1.2
    b = 1.3 + 0.25 * 0.5 + 0.1 * np.cos(alpha)
    event = transition_at(f, b, 0.5, alpha, periods=2)
    oracle = DirectContourOracle(f, b)
    result = oracle.query_event(event, "parent")
    assert event.status == "generic"
    assert {p.role for p in result.event_ports} == {"parent", "child_1", "child_3"}
    assert all(p.event_parameter == event.parameter for p in result.event_ports)
    actions = {port.role: port.action_length for port in event.ports}
    np.testing.assert_allclose(
        actions["parent"], actions["child_1"] + actions["child_3"], atol=1e-8
    )
    assert {role for role, _, _ in result.port_outcomes} == {
        "parent",
        "child_1",
        "child_3",
    }
    assert result.status is ContourStatus.UNKNOWN  # event curve is not certified
    for port in event.ports:
        continued = [
            path
            for path in result.paths
            if len(path.points) >= 3
            and abs(path.points[0].zeta_in - port.zeta_in) < 0.2
            and abs(path.points[0].zeta_out - port.zeta_out) < 0.2
        ]
        assert continued, port.role
        assert (
            max(
                abs(point.action_length - port.action_length)
                for path in continued
                for point in path.points
            )
            <= 2e-5
        )
    child = next(port for port in event.ports if port.role == "child_1")
    seed = oracle._regular_seed_near_port(event, child)
    assert seed is not None
    canonical = oracle.query(seed, events=(event,))
    lifted = type(seed)(
        seed.s,
        seed.alpha,
        seed.zeta_in - oracle.period,
        seed.zeta_out - oracle.period,
        seed.action_length,
        seed.error_A,
    )
    same_event_on_lift = oracle.query(lifted, events=(event,))
    assert same_event_on_lift.status is canonical.status
    assert same_event_on_lift.reason == canonical.reason


def test_transition_curve_prevents_uncertified_edge_witness():
    # A real Γmax separates root patterns. An edge intersection through that
    # pattern cannot become an accessibility witness without a certificate.
    f = SyntheticFourierField(
        1,
        np.array([0, 0, 0, 1]),
        np.array([0, 1, 2, 0]),
        np.array([[2, 0], [-1, 0], [0.3, 0.25], [0.06, 0]]),
        np.zeros((4, 2)),
        np.array([0.21]),
        np.array([3.0]),
        np.array([0.0]),
    )
    oracle = DirectContourOracle(f, 1.4, ContourConfig(max_steps=100))
    result = oracle.query(oracle.seed(0.5, 1.2))
    assert result.status is ContourStatus.UNKNOWN
    assert any(path.edge_reached for path in result.paths)
    assert "uncertified root topology" in result.reason
    assert result.witness is None


def test_contour_budget_exhaustion_remains_unknown():
    f = field((0.28, -0.8, 0.8), -0.08)
    oracle = DirectContourOracle(f, 2.0, ContourConfig(max_steps=2))
    result = oracle.query(oracle.seed(0.6, 0.0))
    assert result.status is ContourStatus.UNKNOWN
    assert "budget" in result.reason.lower()


def test_small_steps_cannot_fake_contour_closure():
    f = field((0.28, -0.8, 0.8), -0.08)
    oracle = DirectContourOracle(
        f, 2.0, ContourConfig(step=0.001, min_step=0.001, max_steps=9)
    )
    result = oracle.query(oracle.seed(0.6, 0.0))
    assert result.status is ContourStatus.UNKNOWN
    assert not any(path.closed for path in result.paths)


def test_action_gradient_keeps_radial_c_and_shear():
    f = field((0.0, 0.08), 0.12, iota=(0.15, 0.3), G=(3.0, 0.4), I=(0.2,))
    oracle = DirectContourOracle(f, 2.0)
    q = oracle.seed(0.45, 0.25)
    gradient = oracle.action_gradient(q)
    eps = 2e-4
    radial = (
        oracle.sample(q.s + eps, q.alpha, q).action_length
        - oracle.sample(q.s - eps, q.alpha, q).action_length
    ) / (2 * eps)
    angular = (
        oracle.sample(q.s, q.alpha + eps, q).action_length
        - oracle.sample(q.s, q.alpha - eps, q).action_length
    ) / (2 * eps)
    np.testing.assert_allclose(gradient, [radial, angular], rtol=3e-3, atol=2e-4)
    assert abs(gradient[0]) > 0.01


def test_saddle_and_disconnected_equal_action_are_not_merged():
    # The origin is a critical contour; equal actions elsewhere do not imply
    # connected contours. The oracle must leave a saddle query unresolved.
    f = field((0.25, -1.0, 1.0), 0.25)
    oracle = DirectContourOracle(f, 2.0)
    q = oracle.seed(0.5, 0.0)
    result = oracle.query(q)
    assert result.status is ContourStatus.UNKNOWN
    assert "saddle/flat action gradient" in result.reason


def test_periodic_lifts_and_multiple_wells_keep_root_identity():
    f = SyntheticFourierField(
        1,
        np.array([0, 0]),
        np.array([0, 2]),
        np.array([[2.0], [-1.0]]),
        np.zeros((2, 1)),
        np.array([0.25]),
        np.array([3.0]),
        np.array([0.0]),
    )
    oracle = DirectContourOracle(f, 2.0)
    one = oracle.seed(0.5, 0.0, zeta_in_hint=-np.pi / 4)
    two = oracle.seed(0.5, 0.0, zeta_in_hint=3 * np.pi / 4)
    assert not np.isclose(one.zeta_in, two.zeta_in)
    lifted = type(one)(
        one.s,
        one.alpha + 0.25 * 2 * np.pi,
        one.zeta_in - 2 * np.pi,
        one.zeta_out - 2 * np.pi,
        one.action_length,
    )
    assert oracle._equivalent(one, lifted)
    assert not oracle._equivalent(two, lifted)
    np.testing.assert_allclose(one.action_length, two.action_length, atol=1e-10)
    assert not oracle._equivalent(one, two)


def test_dmerc_reference_closed_edge_and_transition_probes():
    name = "boozmn_20260406-01-262-Ax_nfp4_Garabedian_mpol2_ntor2_minx0_allNfp_aspect10_DMercFail_m0p3_eval000323_low_resolution.nc"
    f = BoozerField.from_boozmn(Path(DATA_DIR) / name)
    b = 10.648368010778036  # R0 radially global sampled extrema, lambda_n=0.8.
    oracle = DirectContourOracle(
        f, b, ContourConfig(step=0.035, max_steps=80, max_certificate_boxes=64)
    )
    closed = oracle.query(oracle.seed(0.3, 0.0))
    assert all(path.closed for path in closed.paths)
    assert closed.status is ContourStatus.UNKNOWN
    edge_seed = oracle.seed(0.95, -0.1577992744671513)
    # R1's independent composite quadrature includes interior extrema.
    sigma = np.sign(float(f.C(edge_seed.s)))
    z0 = -sigma * oracle.period
    scan = ForwardLineCatalogue(
        f,
        edge_seed.s,
        edge_seed.alpha + float(f.iota(edge_seed.s)) * z0,
        z0,
        ForwardScanConfig(max_periods=oracle.config.scan_periods),
    )
    scan.extend_to(oracle.config.scan_periods)
    well = min(scan.query(b).wells, key=lambda w: abs(w.zeta_in))
    independent = batched_bounce_integrals(scan, (well,))[0]
    np.testing.assert_allclose(
        edge_seed.action_length, independent.A, atol=2e-6, rtol=0
    )
    assert edge_seed.error_A <= oracle.config.action_atol / 2
    edge = oracle.query(edge_seed)
    assert any(path.edge_reached for path in edge.paths)
    assert edge.status is ContourStatus.ACCESSIBLE
    assert edge.witness is not None
    assert abs(edge.witness.points[-1].s - 1) < 1e-12
    assert (
        max(
            abs(point.action_length - edge_seed.action_length)
            for point in edge.witness.points
        )
        <= oracle.config.action_atol
    )
    assert max(point.error_A for point in edge.witness.points) <= (
        oracle.config.action_atol / 2
    )
    event = transition_at(f, b, 0.8, -0.1577992744671513, periods=2)
    branched = oracle.query_event(event, "parent")
    assert event.status == "generic"
    assert {role for role, _, _ in branched.port_outcomes} == {
        "parent",
        "child_1",
        "child_3",
    }
    assert all(port.event_parameter == event.parameter for port in branched.event_ports)
