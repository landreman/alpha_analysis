from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from alpha_analysis.j_connectivity.denominator import (
    BoundsConfig,
    DenominatorConfig,
    UniformSourceProfile,
    compute_denominator,
    denominator_convergence,
    find_global_B_bounds,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField
from alpha_analysis.j_connectivity.visualization import (
    plot_B_extrema_profiles,
    plot_denominator_convergence,
)


def _field(
    *,
    nfp=3,
    m=(0,),
    n=(0,),
    cosine=((2.0,),),
    sine=None,
    G=(1.0,),
    I=(0.0,),
    iota=(0.7,),
):
    cosine_array = np.asarray(cosine, dtype=float)
    sine_array = np.zeros_like(cosine_array) if sine is None else np.asarray(sine)
    return SyntheticFourierField(
        nfp=nfp,
        m=np.asarray(m),
        n=np.asarray(n),
        cosine_coefficients=cosine_array,
        sine_coefficients=sine_array,
        G_coefficients=np.asarray(G, dtype=float),
        I_coefficients=np.asarray(I, dtype=float),
        iota_coefficients=np.asarray(iota, dtype=float),
    )


def test_manufactured_denominator_uses_rho_profile_and_absolute_C():
    field = _field(G=(-2.0, -1.0))
    seen_rho = []

    def source(rho):
        seen_rho.append(np.asarray(rho))
        return 1.0 + np.asarray(rho) ** 2

    result = compute_denominator(
        field, source, DenominatorConfig(n_s=3, n_theta=5, n_zeta=7)
    )

    # Integral_0^1 (1+s)(2+s)/2^2 ds times 4 pi^2/nfp.
    expected = 23.0 * np.pi**2 / (6.0 * field.nfp)
    np.testing.assert_allclose(result.V_h, expected, rtol=2e-14)
    np.testing.assert_allclose(seen_rho[0] ** 2, result.nodes_s, atol=2e-16)


def test_uniform_source_validation_and_source_failures_are_explicit():
    rho = np.array([0.0, 0.3, 1.0])
    np.testing.assert_array_equal(UniformSourceProfile(2.5)(rho), np.full(3, 2.5))
    with pytest.raises(ValueError, match="nonnegative"):
        UniformSourceProfile(-1.0)
    with pytest.raises(ValueError, match="source profile"):
        compute_denominator(
            _field(),
            lambda rho: np.where(np.asarray(rho) < 0.5, 1.0, -0.1),
            DenominatorConfig(n_s=4, n_theta=4, n_zeta=4),
        )


def test_periodic_resolution_converges_to_manufactured_integral():
    epsilon = 0.35
    field = _field(
        nfp=2,
        m=(0, 1),
        n=(0, 2),
        cosine=((2.0,), (2.0 * epsilon,)),
        sine=((0.0,), (0.0,)),
        G=(3.0,),
    )
    configs = tuple(
        DenominatorConfig(n_s=2, n_theta=n, n_zeta=n) for n in (4, 8, 16, 32)
    )
    convergence = denominator_convergence(field, UniformSourceProfile(), configs)

    # The phase average of (1 + epsilon cos chi)^-2 is
    # (1-epsilon^2)^-3/2, and B has the leading factor 2.
    expected = 3.0 * (4.0 * np.pi**2 / field.nfp) / (4.0 * (1.0 - epsilon**2) ** 1.5)
    errors = np.abs(np.array([x.V_h for x in convergence.estimates]) - expected)
    assert np.all(np.diff(errors) < 0.0)
    assert errors[-1] < 1.0e-11
    assert np.all(np.diff(convergence.absolute_changes) < 0.0)


def test_refined_global_extrema_and_radial_profiles_are_analytic():
    field = _field(
        nfp=3,
        m=(0, 0, 1),
        n=(0, 3, 0),
        cosine=((2.0, 0.2), (0.1, 0.0), (0.3, 0.0)),
        sine=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
    )
    result = find_global_B_bounds(
        field,
        BoundsConfig(
            n_s=6,
            n_theta=7,
            n_zeta=5,
            candidate_count=6,
            safety_factor=1.5,
            absolute_margin=1.0e-6,
        ),
    )

    np.testing.assert_allclose(result.refined_min, 1.6, atol=2e-10)
    np.testing.assert_allclose(result.refined_max, 2.6, atol=2e-10)
    np.testing.assert_allclose(
        result.profile_min, 1.6 + 0.2 * result.profile_s, atol=2e-9
    )
    np.testing.assert_allclose(
        result.profile_max, 2.4 + 0.2 * result.profile_s, atol=2e-9
    )
    assert result.lower < result.refined_min
    assert result.upper > result.refined_max
    assert result.safety_margin >= 1.0e-6
    np.testing.assert_allclose(result.minimum_location[0], 0.0, atol=2e-10)
    np.testing.assert_allclose(result.maximum_location[0], 1.0, atol=2e-10)


def test_explicit_user_bounds_override_margin_but_must_bracket_extrema():
    field = _field(cosine=((2.0, 0.2),))
    result = find_global_B_bounds(
        field,
        BoundsConfig(
            n_s=4,
            n_theta=4,
            n_zeta=4,
            user_b_min=1.9,
            user_b_max=2.3,
        ),
    )
    assert result.lower == 1.9
    assert result.upper == 2.3

    with pytest.raises(ValueError, match="bracket"):
        find_global_B_bounds(
            field,
            BoundsConfig(
                n_s=4,
                n_theta=4,
                n_zeta=4,
                user_b_min=2.1,
                user_b_max=2.3,
            ),
        )


def test_denominator_diagnostic_plots_show_convergence_and_extrema():
    field = _field(cosine=((2.0, 0.1),))
    convergence = denominator_convergence(
        field,
        UniformSourceProfile(),
        tuple(DenominatorConfig(3, n, n) for n in (4, 8)),
    )
    bounds = find_global_B_bounds(field, BoundsConfig(4, 5, 5))

    figure_convergence, axes_convergence = plot_denominator_convergence(convergence)
    figure_extrema, axis_extrema = plot_B_extrema_profiles(bounds)
    assert len(axes_convergence) == 2
    assert len(axis_extrema.lines) == 2
    assert "periodic" in axes_convergence[0].get_xlabel().lower()
    assert "s" in axis_extrema.get_xlabel().lower()
    plt.close(figure_convergence)
    plt.close(figure_extrema)
