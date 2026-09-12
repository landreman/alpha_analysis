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
    # H_max(0)=1.3+0.2s, so s=.5 is a generic parent/children event.
    coeff = np.array([[2, 0], [-1, 0], [0.3, 0.2]])
    f = SyntheticFourierField(
        1,
        np.zeros(3, dtype=int),
        np.array([0, 1, 2]),
        coeff,
        np.zeros_like(coeff),
        np.array([0.0]),
        np.array([3.0]),
        np.array([0.0]),
    )
    event = transition_at(f, 1.4, 0.5, 0.2, periods=2)
    oracle = DirectContourOracle(f, 1.4)
    result = oracle.query_event(event, "parent")
    assert event.status == "generic"
    assert {p.role for p in result.event_ports} == {"parent", "child_1", "child_3"}
    assert all(p.event_parameter == event.parameter for p in result.event_ports)
    assert {role for role, _, _ in result.port_outcomes} == {
        "parent",
        "child_1",
        "child_3",
    }
    assert (
        result.status is ContourStatus.UNKNOWN
        or result.status is ContourStatus.ACCESSIBLE
    )


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
    assert oracle.query(q).status is ContourStatus.UNKNOWN


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


def test_dmerc_reference_closed_edge_and_transition_probes():
    name = "boozmn_20260406-01-262-Ax_nfp4_Garabedian_mpol2_ntor2_minx0_allNfp_aspect10_DMercFail_m0p3_eval000323_low_resolution.nc"
    f = BoozerField.from_boozmn(Path(DATA_DIR) / name)
    b = 10.648368010778036  # R0 radially global sampled extrema, lambda_n=0.8.
    oracle = DirectContourOracle(
        f, b, ContourConfig(step=0.035, max_steps=80, max_certificate_boxes=64)
    )
    closed = oracle.query(oracle.seed(0.3, 0.0))
    assert all(path.closed for path in closed.paths)
    assert closed.status in (ContourStatus.INACCESSIBLE, ContourStatus.UNKNOWN)
    edge = oracle.query(oracle.seed(0.95, -0.1577992744671513))
    assert any(path.edge_reached for path in edge.paths)
    assert edge.status is ContourStatus.ACCESSIBLE
    assert edge.witness is not None
    assert abs(edge.witness.points[-1].s - 1) < 1e-12
    event = transition_at(f, b, 0.8, -0.1577992744671513, periods=2)
    branched = oracle.query_event(event, "parent")
    assert event.status == "generic"
    assert {role for role, _, _ in branched.port_outcomes} == {
        "parent",
        "child_1",
        "child_3",
    }
    assert all(port.event_parameter == event.parameter for port in branched.event_ports)
