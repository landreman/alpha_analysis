"""Physics tests for the regular first-return well tracer (DESIGN.md §§9, 20.2)."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from alpha_analysis import (
    DATA_DIR,
    BoozerField,
    BoozerSurface,
    compute_J_invariant,
)
from alpha_analysis.j_connectivity import (
    TraceStatus,
    WellTraceConfig,
    sample_well_profile,
    trace_regular_well,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField
from alpha_analysis.j_connectivity.visualization import plot_well_profile


def _fourier_field(
    *,
    nfp: int,
    m: list[int],
    n: list[int],
    cosine: list[float],
    sine: list[float] | None = None,
    iota: float = 0.0,
    C: float = 3.0,
) -> SyntheticFourierField:
    """Return a radially constant analytic field with signed ``G+iota I=C``."""
    if sine is None:
        sine = np.zeros(len(m))
    return SyntheticFourierField(
        nfp=nfp,
        m=np.asarray(m),
        n=np.asarray(n),
        cosine_coefficients=np.asarray(cosine)[:, np.newaxis],
        sine_coefficients=np.asarray(sine)[:, np.newaxis],
        iota_coefficients=np.array([iota]),
        G_coefficients=np.array([C]),
        I_coefficients=np.array([0.0]),
    )


def _simple_well(C: float = 3.0) -> SyntheticFourierField:
    # B(zeta) = 2 - cos(zeta), with b=2.5 roots at +/- 2*pi/3.
    return _fourier_field(nfp=1, m=[0, 0], n=[0, 1], cosine=[2.0, -1.0], C=C)


def _split_well() -> SyntheticFourierField:
    # B = 2 - cos(zeta) + 0.4 cos(2 zeta).
    return _fourier_field(
        nfp=1,
        m=[0, 0, 0],
        n=[0, 1, 2],
        cosine=[2.0, -1.0, 0.4],
    )


def test_regular_trace_obeys_first_return_invariants_and_independent_A_K():
    field = _simple_well()
    b = 2.5
    root = 2.0 * np.pi / 3.0
    trace = trace_regular_well(field, b, np.array([0.4, 0.7, -root]))

    assert trace.status is TraceStatus.REGULAR
    np.testing.assert_allclose(trace.zeta_out_unwrapped, root, atol=2e-11)
    np.testing.assert_allclose(trace.q_out_reduced, [0.4, 0.7, root], atol=2e-11)
    assert abs(trace.B_residual_in) < 1e-12
    assert abs(trace.B_residual_out) < 1e-10
    np.testing.assert_array_equal(trace.extrema_kind, [1])
    np.testing.assert_allclose(trace.extrema_zeta_unwrapped, [0.0], atol=2e-11)
    np.testing.assert_allclose(trace.extrema_B, [1.0], atol=2e-11)
    assert trace.n_internal_maxima == 0

    def B(zeta):
        return 2.0 - np.cos(zeta)

    expected_A = quad(
        lambda zeta: 3.0 / B(zeta) * np.sqrt(1.0 - B(zeta) / b),
        -root,
        root,
        epsabs=2e-12,
        epsrel=2e-12,
    )[0]
    expected_K = quad(
        lambda zeta: 3.0 / (B(zeta) * np.sqrt(1.0 - B(zeta) / b)),
        -root,
        root,
        epsabs=2e-11,
        epsrel=2e-11,
    )[0]
    np.testing.assert_allclose(trace.action_length, expected_A, rtol=2e-10)
    np.testing.assert_allclose(trace.bounce_time_length, expected_K, rtol=2e-10)
    assert trace.quadrature_error_A < 1e-9
    assert trace.quadrature_error_K < 1e-8

    profile = sample_well_profile(field, trace, n_samples=513)
    assert np.all(profile.B[1:-1] < b)
    np.testing.assert_allclose(profile.zeta_unwrapped[[0, -1]], [-root, root])
    np.testing.assert_allclose(
        [profile.cumulative_A[-1], profile.cumulative_K[-1]],
        [trace.action_length, trace.bounce_time_length],
        rtol=2e-4,
    )


def test_physical_field_direction_uses_sign_of_G_plus_iota_I():
    b = 2.5
    root = 2.0 * np.pi / 3.0
    positive = trace_regular_well(_simple_well(C=3.0), b, np.array([0.4, 0.7, -root]))
    negative = trace_regular_well(_simple_well(C=-3.0), b, np.array([0.4, 0.7, root]))

    assert negative.status is TraceStatus.REGULAR
    np.testing.assert_allclose(negative.zeta_out_unwrapped, -root, atol=2e-11)
    np.testing.assert_allclose(
        [negative.action_length, negative.bounce_time_length],
        [positive.action_length, positive.bounce_time_length],
        rtol=2e-11,
    )


def test_internal_extrema_and_itinerary_survive_a_periodic_shift():
    # B = 2 - cos(zeta) + 0.4 cos(2 zeta).  The b=2 well has two
    # minima separated by one sub-threshold internal maximum.
    field = _split_well()
    crossing_cosine = (1.0 - np.sqrt(2.28)) / 1.6
    root = np.arccos(crossing_cosine)
    q_in = np.array([0.6, 1.2, -root])
    trace = trace_regular_well(field, 2.0, q_in)
    shifted = trace_regular_well(field, 2.0, q_in + [0.0, 0.0, 4.0 * np.pi])

    critical = np.arccos(0.625)
    assert trace.status is shifted.status is TraceStatus.REGULAR
    np.testing.assert_array_equal(trace.extrema_kind, [1, -1, 1])
    np.testing.assert_allclose(
        trace.extrema_zeta_unwrapped, [-critical, 0.0, critical], atol=2e-11
    )
    np.testing.assert_allclose(trace.extrema_B, [1.2875, 1.4, 1.2875], atol=2e-11)
    assert trace.n_internal_maxima == 1
    assert shifted.itinerary_hash == trace.itinerary_hash
    np.testing.assert_allclose(
        [shifted.action_length, shifted.bounce_time_length],
        [trace.action_length, trace.bounce_time_length],
        rtol=2e-11,
    )
    np.testing.assert_allclose(shifted.q_out_reduced, trace.q_out_reduced, atol=2e-11)
    shifted_profile = sample_well_profile(field, shifted, n_samples=129)
    np.testing.assert_allclose(
        shifted_profile.zeta_unwrapped[[0, -1]],
        [-root + 4.0 * np.pi, root + 4.0 * np.pi],
        atol=2e-11,
    )


def test_long_well_preserves_period_count_and_cap_is_unresolved():
    # Along the line, B = 2 - cos(theta), theta = -pi/2 + 0.2 zeta.
    # The first exit is 5*pi away: ten nfp=4 field periods.
    field = _fourier_field(
        nfp=4, m=[0, 1], n=[0, 0], cosine=[2.0, -1.0], iota=0.2, C=1.0
    )
    q_in = np.array([0.5, -0.5 * np.pi, 0.0])
    trace = trace_regular_well(field, 2.0, q_in)

    assert trace.status is TraceStatus.REGULAR
    assert trace.field_period_count == 10
    np.testing.assert_allclose(trace.zeta_out_unwrapped, 5.0 * np.pi, atol=2e-10)
    np.testing.assert_allclose(trace.q_out_reduced, [0.5, 0.5 * np.pi, 0.0], atol=2e-10)

    capped = trace_regular_well(
        field, b=2.0, q_in=q_in, config=WellTraceConfig(max_field_periods=4)
    )
    assert capped.status is TraceStatus.MAX_PERIODS
    assert capped.field_period_count == 4
    assert np.isnan(capped.action_length)
    assert np.isnan(capped.bounce_time_length)
    assert np.isnan(capped.B_residual_out)


def test_fourier_aware_scan_finds_a_narrow_high_mode_first_return():
    mode = 128
    b = 1.01
    phase_root = np.arccos(0.99)
    root = phase_root / mode
    field = _fourier_field(nfp=1, m=[0, 0], n=[0, mode], cosine=[2.0, -1.0])

    trace = trace_regular_well(field, b, np.array([0.4, 0.7, -root]))
    assert trace.status is TraceStatus.REGULAR
    np.testing.assert_allclose(trace.zeta_out_unwrapped, root, atol=2e-12)
    np.testing.assert_array_equal(trace.extrema_kind, [1])
    np.testing.assert_allclose(trace.extrema_zeta_unwrapped, [0.0], atol=2e-12)
    expected_A = quad(
        lambda zeta: 3.0
        / (2.0 - np.cos(mode * zeta))
        * np.sqrt(1.0 - (2.0 - np.cos(mode * zeta)) / b),
        -root,
        root,
        epsabs=2e-13,
        epsrel=2e-12,
    )[0]
    np.testing.assert_allclose(trace.action_length, expected_A, rtol=2e-10)

    class ModeBlindField:
        nfp = field.nfp

        def __getattr__(self, name):
            if name in {"m", "n", "xm", "xn"}:
                raise AttributeError(name)
            return getattr(field, name)

    under_resolved = trace_regular_well(
        ModeBlindField(), b, np.array([0.4, 0.7, -root])
    )
    assert not (
        under_resolved.status is TraceStatus.REGULAR
        and np.array_equal(under_resolved.extrema_kind, [1])
    )


def test_shallow_wells_narrower_than_one_scan_cell_remain_regular():
    field = _simple_well()
    default_scan_step = 2.0 * np.pi / WellTraceConfig().samples_per_field_period

    for depth in (1.0e-3, 1.0e-4, 1.0e-5):
        b = 1.0 + depth
        root = np.arccos(1.0 - depth)
        assert 2.0 * root < default_scan_step

        trace = trace_regular_well(field, b, np.array([0.4, 0.7, -root]))

        # For B=2-cos(zeta), this product form evaluates b-B without the
        # endpoint cancellation that the production field evaluator faces.
        def independent_pair(t):
            zeta = root * np.sin(t)
            jacobian = root * np.cos(t)
            B_value = 2.0 - np.cos(zeta)
            radicand = (
                2.0 * np.sin(0.5 * (root + zeta)) * np.sin(0.5 * (root - zeta)) / b
            )
            if radicand <= 0.0:
                return 0.0, 3.0 * np.sqrt(2.0 * root / (b * np.sin(root)))
            common = 3.0 * jacobian / B_value
            return common * np.sqrt(radicand), common / np.sqrt(radicand)

        expected_A = quad(
            lambda t: independent_pair(t)[0],
            -0.5 * np.pi,
            0.5 * np.pi,
            epsabs=1.0e-13,
            epsrel=1.0e-12,
        )[0]
        expected_K = quad(
            lambda t: independent_pair(t)[1],
            -0.5 * np.pi,
            0.5 * np.pi,
            epsabs=1.0e-12,
            epsrel=1.0e-12,
        )[0]

        assert trace.status is TraceStatus.REGULAR
        np.testing.assert_allclose(trace.zeta_out_unwrapped, root, atol=2e-11)
        np.testing.assert_allclose(trace.action_length, expected_A, rtol=2e-10)
        np.testing.assert_allclose(trace.bounce_time_length, expected_K, rtol=2e-10)
        assert trace.quadrature_error_A <= max(
            WellTraceConfig().quadrature_atol,
            WellTraceConfig().quadrature_rtol * abs(trace.action_length),
        )
        assert trace.quadrature_error_K <= max(
            WellTraceConfig().quadrature_atol,
            WellTraceConfig().quadrature_rtol * abs(trace.bounce_time_length),
        )


def test_underresolved_scan_cannot_walk_past_a_hidden_first_exit():
    mode = 32
    phase = 0.125 * np.pi
    amplitude = 0.5
    cosine_phase = np.cos(phase)
    sine_phase = np.sin(phase)
    field = _fourier_field(
        nfp=1,
        m=[0, 0, 0, 0, 0],
        n=[0, 1, mode, mode - 1, mode + 1],
        cosine=[
            2.0,
            0.0,
            amplitude * cosine_phase,
            -0.5 * amplitude * cosine_phase,
            -0.5 * amplitude * cosine_phase,
        ],
        sine=[
            0.0,
            0.2,
            amplitude * sine_phase,
            -0.5 * amplitude * sine_phase,
            -0.5 * amplitude * sine_phase,
        ],
    )

    class ModeBlindField:
        nfp = field.nfp

        def __getattr__(self, name):
            if name in {"m", "n", "xm", "xn"}:
                raise AttributeError(name)
            return getattr(field, name)

    q_in = np.array([0.4, 0.7, 0.0])
    resolved = trace_regular_well(field, 2.0, q_in)
    under_resolved = trace_regular_well(ModeBlindField(), 2.0, q_in)

    assert resolved.status is TraceStatus.REGULAR
    assert under_resolved.status is TraceStatus.ROOT_FAILURE
    assert np.isnan(under_resolved.action_length)
    assert np.isnan(under_resolved.bounce_time_length)


def test_entry_and_in_trace_tangent_contacts_are_not_regular_wells():
    entry = trace_regular_well(_simple_well(), b=3.0, q_in=np.array([0.4, 0.7, np.pi]))

    field = _split_well()
    b = 1.4 + 5.0e-10
    critical = np.arccos(0.625)

    def B(zeta):
        return 2.0 - np.cos(zeta) + 0.4 * np.cos(2.0 * zeta)

    incoming_root = brentq(lambda zeta: B(zeta) - b, -np.pi, -critical)
    internal = trace_regular_well(field, b=b, q_in=np.array([0.4, 0.7, incoming_root]))

    # B-b = 0.2 sin(zeta) - 0.1 sin(2 zeta) has a regular incoming
    # root at -pi and a cubic, sign-changing outgoing contact at zero.
    grazing_field = _fourier_field(
        nfp=1,
        m=[0, 0, 0],
        n=[0, 1, 2],
        cosine=[2.0, 0.0, 0.0],
        sine=[0.0, -0.2, 0.1],
    )
    outgoing = trace_regular_well(
        grazing_field, b=2.0, q_in=np.array([0.4, 0.7, -np.pi])
    )

    for trace in (entry, internal, outgoing):
        assert trace.status is TraceStatus.TANGENT_OR_TRANSITION
        assert np.isnan(trace.action_length)
        assert np.isnan(trace.bounce_time_length)
    np.testing.assert_allclose(internal.extrema_B[-1], 1.4, atol=2e-12)
    np.testing.assert_allclose(outgoing.tangent_zeta_unwrapped[-1], 0.0, atol=2e-11)


def test_selected_W7X_well_agrees_with_legacy_compute_J_invariant():
    path = os.path.join(
        DATA_DIR, "boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
    )
    field = BoozerField.from_boozmn(path)
    s = 0.5
    surface = BoozerSurface(field, s)
    b_min, b_max = field.get_min_max()
    b = b_min + 0.3 * (b_max - b_min)
    zeta_center = np.pi / field.nfp
    theta_center = 0.5 * np.pi + surface.iota * zeta_center
    legacy = compute_J_invariant(
        surface,
        b,
        theta_center,
        zeta_center,
        n_phi=1001,
        phi_margin=5.0,
        refine=True,
    )
    trace = trace_regular_well(
        field,
        b,
        np.array([s, legacy["theta_left"], legacy["phi_left"]]),
    )

    assert trace.status is TraceStatus.REGULAR
    legacy_length_scale = field.R00 * 2.0 * np.pi / field.nfp
    np.testing.assert_allclose(
        trace.action_length / legacy_length_scale, legacy["J"], rtol=1e-11
    )
    np.testing.assert_allclose(
        trace.q_out_reduced[1:],
        [legacy["theta_right"] % (2.0 * np.pi), legacy["phi_right"]],
        atol=2e-9,
    )


def test_W7X_near_threshold_internal_maximum_has_honest_quadrature():
    path = os.path.join(
        DATA_DIR, "boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
    )
    field = BoozerField.from_boozmn(path)
    b = 2.743049654230
    q_in = np.array([0.5, 5.457232542581316, 0.3071391502993066])
    trace = trace_regular_well(field, b, q_in)

    assert trace.status is TraceStatus.REGULAR
    assert trace.n_internal_maxima == 1
    assert 0.0 < b - trace.extrema_B[1] < 2.0e-7

    s, theta_in, zeta_in = q_in
    iota = float(field.iota(s))
    C = abs(float(field.G(s)) + iota * float(field.I(s)))
    sigma = np.sign(float(field.G(s)) + iota * float(field.I(s)))
    u_out = sigma * (trace.zeta_out_unwrapped - zeta_in)
    theta_out = theta_in + iota * (trace.zeta_out_unwrapped - zeta_in)
    slope_in = sigma * float(field.D_B(s, theta_in, zeta_in))
    slope_out = sigma * float(field.D_B(s, theta_out, trace.zeta_out_unwrapped))
    curvature_in = float(field.D2_B(s, theta_in, zeta_in))
    curvature_out = float(field.D2_B(s, theta_out, trace.zeta_out_unwrapped))

    def independent_pair(x):
        u = u_out * np.sin(0.5 * np.pi * x) ** 2
        jacobian = 0.5 * np.pi * u_out * np.sin(np.pi * x)
        zeta = zeta_in + sigma * u
        theta = theta_in + iota * (zeta - zeta_in)
        B_value = float(field.B(s, theta, zeta))
        field_difference = b - B_value
        if x < 1.0e-4:
            field_difference = (
                -trace.B_residual_in - slope_in * u - 0.5 * curvature_in * u**2
            )
        elif x > 1.0 - 1.0e-4:
            distance = u_out - u
            field_difference = (
                -trace.B_residual_out
                + slope_out * distance
                - 0.5 * curvature_out * distance**2
            )
        root = np.sqrt(field_difference / b)
        common = C * jacobian / B_value
        return common * root, common / root

    relative_extrema = sigma * (trace.extrema_zeta_unwrapped - zeta_in) / u_out
    extrema_x = 2.0 / np.pi * np.arcsin(np.sqrt(relative_extrema))
    expected_A = quad(
        lambda x: independent_pair(x)[0],
        0.0,
        1.0,
        points=extrema_x,
        epsabs=1.0e-10,
        epsrel=1.0e-10,
        limit=500,
    )[0]
    expected_K = quad(
        lambda x: independent_pair(x)[1],
        0.0,
        1.0,
        points=extrema_x,
        epsabs=1.0e-8,
        epsrel=1.0e-8,
        limit=500,
    )[0]

    np.testing.assert_allclose(trace.action_length, expected_A, rtol=1.0e-12)
    np.testing.assert_allclose(trace.bounce_time_length, expected_K, rtol=2.0e-8)


def test_unattainable_quadrature_tolerance_is_an_explicit_failure():
    root = 2.0 * np.pi / 3.0
    trace = trace_regular_well(
        _simple_well(),
        2.5,
        np.array([0.4, 0.7, -root]),
        config=WellTraceConfig(quadrature_atol=1.0e-18, quadrature_rtol=1.0e-18),
    )

    assert trace.status is TraceStatus.QUADRATURE_FAILURE
    assert np.isnan(trace.action_length)
    assert np.isnan(trace.bounce_time_length)


def test_well_profile_plot_writes_the_required_static_diagnostic(tmp_path):
    root = 2.0 * np.pi / 3.0
    field = _simple_well()
    trace = trace_regular_well(field, 2.5, np.array([0.4, 0.7, -root]))
    output = tmp_path / "well-profile.png"

    figure, axes = plot_well_profile(field, trace, output_path=output, n_samples=257)

    assert output.exists()
    assert output.stat().st_size > 0
    assert np.asarray(axes).size == 6
    assert "REGULAR" in axes[1, 2].texts[0].get_text()
    plt.close(figure)

    tangent_field = _split_well()
    b = 1.4 + 5.0e-10
    critical = np.arccos(0.625)
    B = lambda zeta: 2.0 - np.cos(zeta) + 0.4 * np.cos(2.0 * zeta)
    incoming_root = brentq(lambda zeta: B(zeta) - b, -np.pi, -critical)
    tangent = trace_regular_well(tangent_field, b, np.array([0.4, 0.7, incoming_root]))
    unresolved_output = tmp_path / "well-profile-unresolved.png"

    unresolved_figure, unresolved_axes = plot_well_profile(
        tangent_field, tangent, output_path=unresolved_output, n_samples=257
    )

    assert unresolved_output.exists()
    assert len(unresolved_axes[0, 0].lines) >= 3
    assert any(
        line.get_label() == "tangent candidate" for line in unresolved_axes[0, 0].lines
    )
    assert "TANGENT_OR_TRANSITION" in unresolved_axes[1, 2].texts[0].get_text()
    plt.close(unresolved_figure)
