"""R1 production-path checks for DESIGN.md §§4.2 and 9."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from alpha_analysis.j_connectivity import TraceStatus, trace_regular_well
from alpha_analysis.j_connectivity.forward_catalogue import (
    CatalogueLinewisePredicate,
    ForwardLineCatalogue,
    ForwardScanConfig,
    batched_bounce_integrals,
    adaptive_bounce_integral,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField


def _field(cosine, n, *, sine=None, iota=0.0, C=3.0, nfp=1):
    cosine = np.asarray(cosine, dtype=float)
    return SyntheticFourierField(
        nfp=nfp,
        m=np.zeros(len(n), dtype=int),
        n=np.asarray(n, dtype=int),
        cosine_coefficients=cosine[:, None],
        sine_coefficients=(
            np.zeros_like(cosine) if sine is None else np.asarray(sine, dtype=float)
        )[:, None],
        iota_coefficients=np.array([iota]),
        G_coefficients=np.array([C]),
        I_coefficients=np.array([0.0]),
    )


def test_shared_catalogue_matches_independent_well_traces():
    field = _field([2.0, -1.0, 0.4], [0, 1, 2])
    catalogue = ForwardLineCatalogue(field, 0.4, 0.7, -np.pi)
    catalogue.extend_to(2)
    assert catalogue.scanned_periods == 2
    for b in (1.5, 2.0, 2.5):
        result = catalogue.query(b)
        assert result.root_complete
        assert len(result.wells) >= 1
        for well in result.wells:
            expected = trace_regular_well(field, b, well.q_in)
            assert expected.status is TraceStatus.REGULAR
            np.testing.assert_allclose(
                well.zeta_out, expected.zeta_out_unwrapped, atol=2e-10
            )
            assert well.zeta_out > well.zeta_in
        integrals = batched_bounce_integrals(catalogue, result.wells)
        assert all(item.status is TraceStatus.REGULAR for item in integrals)
        np.testing.assert_allclose(
            [item.A for item in integrals],
            [
                trace_regular_well(field, b, well.q_in).action_length
                for well in result.wells
            ],
            rtol=2e-7,
        )
    assert catalogue.scanned_periods == 2  # pitch queries reused the same samples


def _long_well_field():
    """B=2-cos(0.2 zeta) has a well spanning several 2*pi windows."""
    return SyntheticFourierField(
        nfp=1,
        m=np.array([0, 1]),
        n=np.array([0, 0]),
        cosine_coefficients=np.array([[2.0], [-1.0]]),
        sine_coefficients=np.zeros((2, 1)),
        iota_coefficients=np.array([0.2]),
        G_coefficients=np.array([1.0]),
        I_coefficients=np.array([0.0]),
    )


def test_window_end_is_not_passing():
    field = _long_well_field()
    catalogue = ForwardLineCatalogue(field, 0.5, -np.pi / 2, 0.0)
    catalogue.extend_to(2)
    censored = catalogue.query(2.0)
    assert not censored.passing_certified
    assert censored.open_right
    assert not censored.root_complete


def test_catalogue_resumes_without_losing_first_crossing():
    field = _long_well_field()
    catalogue = ForwardLineCatalogue(field, 0.5, -np.pi / 2, 0.0)
    catalogue.extend_to(2)
    first_samples = catalogue.sample_count
    catalogue.extend_to(4)
    resolved = catalogue.query(2.0)
    assert catalogue.sample_count > first_samples
    assert resolved.wells
    np.testing.assert_allclose(resolved.wells[0].zeta_in, 0.0, atol=1e-11)
    np.testing.assert_allclose(resolved.wells[0].zeta_out, 5 * np.pi, atol=1e-10)


def test_near_incoming_root_remains_window_censored():
    # A point merely close to the incoming root cannot become an exact root.
    simple = _field([2.0, -1.0], [0, 1])
    close_start = -2 * np.pi / 3 + 5e-11
    close = ForwardLineCatalogue(simple, 0.4, 0.0, close_start)
    close.extend_to(1)
    close_result = close.query(2.5)
    assert close_result.open_left
    assert not close_result.root_complete


def test_scan_detects_or_bounds_hidden_barrier():
    # The 16th harmonic makes a shallow above-b peak within a coarse cell.
    field = _field([2.0, -0.4, 0.08], [0, 1, 16], C=1.0)
    b = 2.083829683510348
    start = brentq(lambda z: float(field.B(0.5, 0.0, z)) - b, 4.43, 4.45)
    dense = np.linspace(start + 1e-9, start + 2 * np.pi, 100001)
    residual = field.B(0.5, 0.0, dense) - b
    first = np.flatnonzero((residual[:-1] < 0) & (residual[1:] >= 0))[0]
    expected = brentq(
        lambda z: float(field.B(0.5, 0.0, z)) - b, dense[first], dense[first + 1]
    )
    catalogue = ForwardLineCatalogue(
        field, 0.5, 0.0, start - 0.1, ForwardScanConfig(samples_per_period=4)
    )
    catalogue.extend_to(1)
    result = catalogue.query(b)
    assert result.root_complete
    np.testing.assert_allclose(result.wells[0].zeta_out, expected, atol=1e-10)

    class NoFourierEnvelope:
        nfp = field.nfp

        def __getattr__(self, name):
            if name in {
                "m",
                "n",
                "xm",
                "xn",
                "cosine_coefficients",
                "sine_coefficients",
            }:
                raise AttributeError(name)
            return getattr(field, name)

    uncovered = ForwardLineCatalogue(
        NoFourierEnvelope(),
        0.5,
        0.0,
        start - 0.1,
        ForwardScanConfig(samples_per_period=4),
    )
    uncovered.extend_to(1)
    unknown = uncovered.query(b)
    assert unknown.status is TraceStatus.ROOT_FAILURE
    assert unknown.unknown_cells
    assert not unknown.passing_certified


def test_batched_integrals_preserve_near_tangent_accuracy():
    field = _field([2.0, -1.0], [0, 1], C=-3.0)
    b = 1.00001
    root = np.arccos(2 - b)
    catalogue = ForwardLineCatalogue(field, 0.4, 0.7, root)
    catalogue.extend_to(1)
    result = catalogue.query(b)
    assert result.wells
    well = result.wells[0]
    value = batched_bounce_integrals(catalogue, [well])[0]
    reference = adaptive_bounce_integral(catalogue, well)
    assert value.status is reference.status is TraceStatus.REGULAR
    assert value.method == "batched_gl"
    np.testing.assert_allclose(
        [value.A, value.K], [reference.A, reference.K], rtol=2e-7
    )

    # Independent product formula for 2-cos(z): b-B=2sin((r+z)/2)sin((r-z)/2).
    def independent(t):
        z = root * np.sin(t)
        jac = root * np.cos(t)
        rad = 2 * np.sin((root + z) / 2) * np.sin((root - z) / 2) / b
        B = 2 - np.cos(z)
        return 3 * jac / B * np.sqrt(rad), 3 * jac / B / np.sqrt(rad)

    exact = [
        quad(lambda t: independent(t)[i], -np.pi / 2, np.pi / 2)[0] for i in (0, 1)
    ]
    np.testing.assert_allclose([value.A, value.K], exact, rtol=2e-7)
    assert well.zeta_out < well.zeta_in  # negative C follows decreasing zeta


def test_linewise_masks_keep_censored_weight_possible_and_detect_closed_passing_line():
    simple = _field([2.0, -1.0], [0, 1])
    predicate = CatalogueLinewisePredicate(simple, ForwardScanConfig(max_periods=1))
    masks = predicate(np.array([0.5]), np.array([0.0]), np.array([0.0]), 2.0)
    assert masks.definitely_trapped[0]
    assert masks.possibly_trapped[0]
    assert not masks.unresolved_reasons

    long_well = SyntheticFourierField(
        nfp=1,
        m=np.array([0, 1]),
        n=np.array([0, 0]),
        cosine_coefficients=np.array([[2.0], [-1.0]]),
        sine_coefficients=np.zeros((2, 1)),
        iota_coefficients=np.array([0.2]),
        G_coefficients=np.array([1.0]),
        I_coefficients=np.array([0.0]),
    )
    censored = CatalogueLinewisePredicate(long_well, ForwardScanConfig(max_periods=1))(
        np.array([0.5]), np.array([0.0]), np.array([0.0]), 2.0
    )
    assert not censored.definitely_trapped[0]
    assert censored.possibly_trapped[0]
    assert censored.unresolved_reasons

    degenerate = _field([2.0, -1.0], [0, 1], C=0.0)
    failed = CatalogueLinewisePredicate(degenerate, ForwardScanConfig(max_periods=1))(
        np.array([0.5]), np.array([0.0]), np.array([0.0]), 2.0
    )
    assert not failed.definitely_trapped[0]
    assert failed.possibly_trapped[0]
    assert "line scan failed" in failed.unresolved_reasons[0]

    # B=2+cos(theta), iota=0: the theta=pi line is below b forever.
    plateau = SyntheticFourierField(
        nfp=1,
        m=np.array([0, 1]),
        n=np.array([0, 0]),
        cosine_coefficients=np.array([[2.0], [1.0]]),
        sine_coefficients=np.zeros((2, 1)),
        iota_coefficients=np.array([0.0]),
        G_coefficients=np.array([1.0]),
        I_coefficients=np.array([0.0]),
    )
    plateau_masks = CatalogueLinewisePredicate(
        plateau, ForwardScanConfig(max_periods=1)
    )(np.array([0.5]), np.array([np.pi]), np.array([0.0]), 2.5)
    assert not plateau_masks.definitely_trapped[0]
    assert not plateau_masks.possibly_trapped[0]


def test_exact_separatrix_does_not_get_a_finite_critical_K():
    field = _field([2.0, -1.0], [0, 1])
    catalogue = ForwardLineCatalogue(field, 0.4, 0.0, 0.0)
    catalogue.extend_to(1)
    result = catalogue.query(3.0)
    assert result.status is TraceStatus.TANGENT_OR_TRANSITION
    assert result.tangent_candidates
    assert not result.passing_certified
    assert not result.wells
    assert "tangent" in result.reason
    assert "window boundary" in result.reason
