from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.integrate import quad

from alpha_analysis.j_connectivity.denominator import (
    DenominatorConfig,
    UniformSourceProfile,
    compute_denominator,
)
from alpha_analysis.j_connectivity.population import (
    LinewiseTrappingMasks,
    OwnedWeightBounds,
    PopulationQuadratureConfig,
    WeightBounds,
    build_population_context,
    build_population_ledger,
    compute_pitch_band_estimate,
    compute_population_slice,
    enclose_pitch_band_fraction,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField
from alpha_analysis.j_connectivity.visualization import plot_population_diagnostics


def _field(
    *,
    nfp=1,
    m=(0,),
    n=(0,),
    cosine=((2.0,),),
    iota=(0.7,),
    G=(1.0,),
):
    cosine = np.asarray(cosine, dtype=float)
    return SyntheticFourierField(
        nfp=nfp,
        m=np.asarray(m),
        n=np.asarray(n),
        cosine_coefficients=cosine,
        sine_coefficients=np.zeros_like(cosine),
        iota_coefficients=np.asarray(iota),
        G_coefficients=np.asarray(G),
        I_coefficients=np.array([0.0]),
    )


def test_historical_cut_matrix_is_preserved():
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "validation"
        / "milestone10.3-real-equilibria.json"
    )
    payload = json.loads(path.read_text())
    cases = payload["cases"]

    assert len(cases) == 120
    coordinates = {
        (
            case["file"],
            case["lambda_n"],
            case["backend"],
            case["extractor"],
        )
        for case in cases.values()
    }
    assert len(coordinates) == 120
    assert len({item[0] for item in coordinates}) == 5
    assert {item[1] for item in coordinates} == {0.05, 0.1, 0.5, 0.8, 0.9, 0.95}
    assert {item[2] for item in coordinates} == {"structured", "gmsh"}
    assert {item[3] for item in coordinates} == {
        "marching_tetrahedra",
        "pyvista",
    }

    assert payload["summary"]["classifications"] == {
        "exception": 1,
        "no_transitions": 32,
        "resolved": 10,
        "unresolved_explicit": 77,
    }
    assert payload["summary"]["resolved_fraction"] == 42 / 120
    assert payload["summary"]["case_count"] == 120

    exception = cases["3:0.5:gmsh:marching_tetrahedra"]
    assert exception["outcome"] == "exception"
    assert exception["exception_type"] == "ConstrainedCutError"
    assert "lacks sides" in exception["exception_message"]
    assert exception["elapsed_seconds"] > 6000.0

    unresolved = [
        case
        for case in cases.values()
        if case.get("classification") == "unresolved_explicit"
    ]
    assert len(unresolved) == 77
    assert all(case["terminal_reasons"] for case in unresolved)
    assert all(case["failure_class_counts"] for case in unresolved)
    assert payload["controls"]["budgets"]["max_field_period_caps"][-1] == 1024
    assert len(payload["controls"]["implementation_sha256"]) == 6


def test_population_ledger_matches_analytic_trapped_fraction():
    # B=B0(1-a(s)^2 sin^2(theta)) has exact maximum B0 and
    # sqrt(1-B/Bmax)=a(s) |sin(theta)|. The independently integrated radial
    # variation makes h(rho)=1+rho^2 directly affect the normalized result.
    B0 = 2.0
    a_squared_0 = 0.16
    a_squared_1 = 0.20
    field = _field(
        nfp=2,
        m=(0, 2),
        n=(0, 0),
        cosine=(
            (B0 * (1.0 - 0.5 * a_squared_0), -0.5 * B0 * a_squared_1),
            (0.5 * B0 * a_squared_0, 0.5 * B0 * a_squared_1),
        ),
        G=(2.0, 1.0),
    )

    def source(rho):
        return 1.0 + np.asarray(rho) ** 2

    config = PopulationQuadratureConfig(n_s=16, n_theta=4096, n_zeta=2)
    context = build_population_context(
        field,
        source,
        config,
        source_name="h(rho)=1+rho^2",
        surface_maximum=np.full(config.n_s, B0),
        surface_maximum_scope="analytic",
        surface_maximum_is_certified_exact=True,
    )
    denominator = compute_denominator(
        field,
        source,
        DenominatorConfig(config.n_s, config.n_theta, config.n_zeta),
    )
    band = compute_pitch_band_estimate(
        context,
        B0 * (1.0 - a_squared_0 - a_squared_1),
        B0,
        denominator.V_h,
        dense_line_assumption=True,
    )

    def angular_integrals(s):
        a_squared = a_squared_0 + a_squared_1 * s
        a = np.sqrt(a_squared)
        c = 1.0 - a_squared
        numerator = 2.0 * a / c + 2.0 * np.arctan(a / np.sqrt(c)) / c**1.5
        denominator_theta = np.pi * (2.0 - a_squared) / c**1.5
        return numerator, denominator_theta

    numerator = quad(
        lambda s: (1.0 + s) * (2.0 + s) * angular_integrals(s)[0],
        0.0,
        1.0,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
    )[0]
    denominator_reference = quad(
        lambda s: (1.0 + s) * (2.0 + s) * angular_integrals(s)[1],
        0.0,
        1.0,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
    )[0]
    expected = numerator / denominator_reference
    np.testing.assert_allclose(band.fraction, expected, atol=3.0e-7, rtol=0.0)
    assert band.source_name == "h(rho)=1+rho^2"
    assert band.trapping_scope == "dense_line_surface_maximum"
    assert band.bound_scope == "estimate"


def test_surface_maximum_is_not_linewise_trapping_on_rational_plateau():
    # With iota=0 and B=2+cos(theta), each field line holds theta fixed.  No
    # line has a bounded B<b interval although the surface maximum is 3.
    field = _field(
        m=(0, 1),
        n=(0, 0),
        cosine=((2.0,), (1.0,)),
        iota=(0.0,),
    )
    config = PopulationQuadratureConfig(n_s=3, n_theta=128, n_zeta=4)
    context = build_population_context(
        field,
        UniformSourceProfile(),
        config,
        source_name="h=1",
        surface_maximum=np.full(config.n_s, 3.0),
        surface_maximum_scope="analytic-certified-upper",
        surface_maximum_is_certified_upper=True,
    )

    linewise = compute_population_slice(
        context,
        2.5,
        linewise_trapped=lambda s, theta, zeta, b, B: LinewiseTrappingMasks(
            definitely_trapped=np.zeros_like(B, dtype=bool),
            possibly_trapped=np.zeros_like(B, dtype=bool),
        ),
    )
    surface_upper = compute_population_slice(context, 2.5)

    assert linewise.total_weight == 0.0
    assert linewise.trapping_scope == "linewise_masks"
    assert surface_upper.total_weight_lower == 0.0
    assert surface_upper.total_weight_upper > 0.0
    with pytest.raises(ValueError, match="interval-valued"):
        _ = surface_upper.total_weight
    assert surface_upper.trapping_scope == "surface_maximum_upper_only"
    assert surface_upper.bound_scope == "estimate"

    unresolved = compute_population_slice(
        context,
        2.5,
        linewise_trapped=lambda s, theta, zeta, b, B: LinewiseTrappingMasks(
            definitely_trapped=np.zeros_like(B, dtype=bool),
            possibly_trapped=B < b,
            unresolved_reasons=("root scan reached its period cap",),
        ),
    )
    assert unresolved.total_weight_lower == 0.0
    assert unresolved.total_weight_upper > 0.0
    assert any("period cap" in item for item in unresolved.uncontrolled_errors)

    with pytest.raises(ValueError, match="subset"):
        compute_population_slice(
            context,
            2.5,
            linewise_trapped=lambda s, theta, zeta, b, B: LinewiseTrappingMasks(
                definitely_trapped=np.ones_like(B, dtype=bool),
                possibly_trapped=np.zeros_like(B, dtype=bool),
            ),
        )
    with pytest.raises(ValueError, match="B >= b"):
        compute_population_slice(
            context,
            2.5,
            linewise_trapped=lambda s, theta, zeta, b, B: LinewiseTrappingMasks(
                definitely_trapped=np.zeros_like(B, dtype=bool),
                possibly_trapped=np.ones_like(B, dtype=bool),
                unresolved_reasons=("not traced",),
            ),
        )
    with pytest.raises(ValueError, match="requires a recorded reason"):
        compute_population_slice(
            context,
            2.5,
            linewise_trapped=lambda s, theta, zeta, b, B: LinewiseTrappingMasks(
                definitely_trapped=np.zeros_like(B, dtype=bool),
                possibly_trapped=B < b,
            ),
        )
    with pytest.raises(ValueError, match="below a sampled field value"):
        build_population_context(
            field,
            UniformSourceProfile(),
            config,
            source_name="h=1",
            surface_maximum=np.full(config.n_s, 2.9),
        )


def test_population_slice_matches_independent_analytic_integral():
    B0 = 2.0
    a = 0.6
    b = 1.9
    field = _field(
        m=(0, 2),
        n=(0, 0),
        cosine=((B0 * (1.0 - 0.5 * a**2),), (0.5 * B0 * a**2,)),
        G=(2.0, 1.0),
    )
    config = PopulationQuadratureConfig(n_s=4, n_theta=65536, n_zeta=2)
    context = build_population_context(
        field,
        lambda rho: 1.0 + np.asarray(rho) ** 2,
        config,
        source_name="h(rho)=1+rho^2",
        surface_maximum=np.full(config.n_s, B0),
        surface_maximum_scope="analytic",
        surface_maximum_is_certified_exact=True,
    )
    result = compute_population_slice(context, b, dense_line_assumption=True)

    theta_root = np.arcsin(np.sqrt((1.0 - b / B0) / a**2))

    def angular_integrand(theta):
        B = B0 * (1.0 - a**2 * np.sin(theta) ** 2)
        return 1.0 / (B * np.sqrt(1.0 - B / b))

    one_allowed_interval = quad(
        angular_integrand,
        theta_root,
        np.pi - theta_root,
        epsabs=1.0e-11,
        epsrel=1.0e-11,
    )[0]
    radial_weight = quad(lambda s: (1.0 + s) * (2.0 + s), 0.0, 1.0)[0]
    expected = 2.0 * one_allowed_interval * (2.0 * np.pi / field.nfp) * radial_weight
    np.testing.assert_allclose(result.total_weight, expected, rtol=0.01, atol=0.0)


def test_missing_weight_uses_total_upper_minus_covered_lower():
    total = WeightBounds(9.0, 11.0, bound_scope="field_enclosure", is_certified=True)
    reachable = OwnedWeightBounds(
        owner_ids=np.array([10, 11]),
        lower=np.array([1.0, 1.5]),
        upper=np.array([1.2, 1.8]),
        bound_scope="field_enclosure",
        is_certified=True,
    )
    nonreachable = OwnedWeightBounds(
        owner_ids=np.array([20]),
        lower=np.array([2.0]),
        upper=np.array([2.4]),
        bound_scope="field_enclosure",
        is_certified=True,
    )
    unresolved = OwnedWeightBounds(
        owner_ids=np.array([30]),
        lower=np.array([0.5]),
        upper=np.array([0.9]),
        bound_scope="field_enclosure",
        is_certified=True,
    )

    ledger = build_population_ledger(total, reachable, nonreachable, unresolved)

    assert ledger.covered_lower == 5.0
    assert ledger.missing_weight_upper == 6.0
    assert ledger.Q_lower == 2.5
    assert ledger.Q_upper == 9.0
    assert ledger.bound_scope == "field_enclosure"

    overlapping = OwnedWeightBounds(
        owner_ids=np.array([11]),
        lower=np.array([0.2]),
        upper=np.array([0.3]),
        bound_scope="field_enclosure",
        is_certified=True,
    )
    with pytest.raises(ValueError, match="overlap"):
        build_population_ledger(total, reachable, nonreachable, overlapping)

    excessive = OwnedWeightBounds(
        owner_ids=np.array([31]),
        lower=np.array([20.0]),
        upper=np.array([21.0]),
        bound_scope="field_enclosure",
        is_certified=True,
    )
    with pytest.raises(ValueError, match="exceeds total upper"):
        build_population_ledger(total, reachable, nonreachable, excessive)

    estimate = WeightBounds(
        9.0,
        11.0,
        bound_scope="estimate",
        is_certified=False,
        uncontrolled_errors=("spatial quadrature",),
    )
    with pytest.raises(ValueError, match="certified total upper"):
        build_population_ledger(estimate, reachable, nonreachable, unresolved)
    model_reachable = replace(reachable, bound_scope="model_enclosure")
    assert (
        build_population_ledger(total, model_reachable, nonreachable).bound_scope
        == "model_enclosure"
    )
    uncontrolled_reachable = replace(
        reachable,
        bound_scope="estimate",
        is_certified=False,
        uncontrolled_errors=("clipped-cell quadrature",),
    )
    with pytest.raises(ValueError, match="certified owned weight"):
        build_population_ledger(total, uncontrolled_reachable, nonreachable)
    with pytest.raises(ValueError, match="bound_scope must be one of"):
        WeightBounds(9.0, 11.0, bound_scope="claimed-field-ish", is_certified=True)
    with pytest.raises(ValueError, match="cannot retain uncontrolled errors"):
        WeightBounds(
            9.0,
            11.0,
            bound_scope="field_enclosure",
            is_certified=True,
            uncontrolled_errors=("spatial quadrature",),
        )


def test_whole_pitch_band_upper_weight():
    field = _field(m=(0, 1), n=(0, 0), cosine=((2.0,), (0.5,)), iota=(np.sqrt(2.0),))
    config = PopulationQuadratureConfig(n_s=3, n_theta=2048, n_zeta=2)
    context = build_population_context(
        field,
        UniformSourceProfile(),
        config,
        source_name="h=1",
        surface_maximum=np.full(config.n_s, 2.5),
        surface_maximum_scope="analytic",
        surface_maximum_is_certified_exact=True,
    )
    denominator = compute_denominator(
        field,
        UniformSourceProfile(),
        DenominatorConfig(config.n_s, config.n_theta, config.n_zeta),
    )
    band = compute_pitch_band_estimate(
        context,
        1.5,
        2.5,
        denominator.V_h,
        dense_line_assumption=True,
        dense_line_certification="irrational transform is constant at sqrt(2)",
    )
    denominator_bounds = WeightBounds(
        denominator.V_h - 1.0e-4,
        denominator.V_h + 1.0e-4,
        bound_scope="field_enclosure",
        is_certified=True,
    )
    bounds = enclose_pitch_band_fraction(
        band,
        pitch_weight_absolute_error=1.0e-4,
        denominator=denominator_bounds,
        bound_scope="field_enclosure",
    )

    assert bounds.lower <= band.fraction <= bounds.upper
    assert bounds.upper - bounds.lower > 0.0
    assert bounds.lower == bounds.pitch_weight_lower / (2.0 * bounds.denominator.upper)
    assert bounds.upper == bounds.pitch_weight_upper / (2.0 * bounds.denominator.lower)
    assert bounds.bound_scope == "field_enclosure"
    assert bounds.source_name == "h=1"
    assert bounds.dense_line_certification is not None
    assert bounds.surface_maximum_scope == "analytic"
    assert "spatial quadrature" in bounds.estimate_uncontrolled_errors
    with pytest.raises(ValueError, match="field-enclosed denominator"):
        enclose_pitch_band_fraction(
            band,
            pitch_weight_absolute_error=1.0e-4,
            denominator=replace(denominator_bounds, bound_scope="model_enclosure"),
            bound_scope="field_enclosure",
        )

    unproven = compute_pitch_band_estimate(
        context, 1.5, 2.5, denominator.V_h, dense_line_assumption=True
    )
    unproven_bounds = enclose_pitch_band_fraction(
        unproven,
        pitch_weight_absolute_error=1.0e-4,
        denominator=denominator_bounds,
        bound_scope="field_enclosure",
    )
    assert unproven_bounds.lower == 0.0
    assert unproven_bounds.upper > 0.0
    assert unproven_bounds.dense_line_certification is None

    uncertified = compute_pitch_band_estimate(
        replace(
            context,
            surface_maximum_is_certified_upper=False,
            surface_maximum_is_certified_exact=False,
        ),
        1.5,
        2.5,
        denominator.V_h,
        dense_line_assumption=True,
    )
    with pytest.raises(ValueError, match="certified surface maximum"):
        enclose_pitch_band_fraction(
            uncertified,
            pitch_weight_absolute_error=1.0e-4,
            denominator=denominator_bounds,
            bound_scope="field_enclosure",
        )

    upper_not_exact = compute_pitch_band_estimate(
        replace(
            context,
            surface_maximum=np.full(config.n_s, 2.6),
            surface_maximum_is_certified_exact=False,
        ),
        1.5,
        2.6,
        denominator.V_h,
        dense_line_assumption=True,
        dense_line_certification="irrational transform is constant at sqrt(2)",
    )
    upper_not_exact_bounds = enclose_pitch_band_fraction(
        upper_not_exact,
        pitch_weight_absolute_error=1.0e-4,
        denominator=denominator_bounds,
        bound_scope="field_enclosure",
    )
    assert upper_not_exact_bounds.lower == 0.0
    assert upper_not_exact_bounds.upper > 0.0

    plateau_field = _field(m=(0, 1), n=(0, 0), cosine=((2.0,), (1.0,)), iota=(0.0,))
    plateau_context = build_population_context(
        plateau_field,
        UniformSourceProfile(),
        config,
        source_name="h=1",
        surface_maximum=np.full(config.n_s, 3.0),
        surface_maximum_scope="analytic",
        surface_maximum_is_certified_exact=True,
    )
    plateau_band = compute_pitch_band_estimate(
        plateau_context,
        1.0,
        3.0,
        compute_denominator(
            plateau_field,
            UniformSourceProfile(),
            DenominatorConfig(config.n_s, config.n_theta, config.n_zeta),
        ).V_h,
        dense_line_assumption=True,
    )
    plateau_denominator_bounds = WeightBounds(
        plateau_band.denominator_estimate - 1.0e-4,
        plateau_band.denominator_estimate + 1.0e-4,
        bound_scope="field_enclosure",
        is_certified=True,
    )
    plateau_bounds = enclose_pitch_band_fraction(
        plateau_band,
        pitch_weight_absolute_error=1.0e-4,
        denominator=plateau_denominator_bounds,
        bound_scope="field_enclosure",
    )
    assert plateau_band.pitch_weight > 0.0
    assert plateau_bounds.lower == 0.0

    first = compute_pitch_band_estimate(
        context, 1.7, 2.0, denominator.V_h, dense_line_assumption=True
    )
    second = compute_pitch_band_estimate(
        context, 2.0, 2.3, denominator.V_h, dense_line_assumption=True
    )
    joined = compute_pitch_band_estimate(
        context, 1.7, 2.3, denominator.V_h, dense_line_assumption=True
    )
    np.testing.assert_allclose(
        first.pitch_weight + second.pitch_weight,
        joined.pitch_weight,
        rtol=2.0e-15,
        atol=2.0e-15,
    )

    def band_integrand(theta):
        B = 2.0 + 0.5 * np.cos(theta)
        lower = max(B, 2.0)
        upper = min(2.5, 2.3)
        if upper <= lower:
            return 0.0
        return (np.sqrt(1.0 - B / upper) - np.sqrt(1.0 - B / lower)) / B**2

    support_root = np.arccos(0.6)
    numerator = quad(
        band_integrand,
        0.0,
        2.0 * np.pi,
        points=[
            support_root,
            0.5 * np.pi,
            np.pi,
            1.5 * np.pi,
            2.0 * np.pi - support_root,
        ],
        limit=100,
    )[0]
    denominator_theta = quad(
        lambda theta: (2.0 + 0.5 * np.cos(theta)) ** -2,
        0.0,
        2.0 * np.pi,
    )[0]
    np.testing.assert_allclose(
        second.fraction, numerator / denominator_theta, rtol=5.0e-5, atol=1.0e-8
    )


def test_population_diagnostic_shows_pitch_and_radial_weights():
    field = _field(m=(0, 1), n=(0, 0), cosine=((2.0,), (0.4,)))
    config = PopulationQuadratureConfig(n_s=4, n_theta=64, n_zeta=2)
    context = build_population_context(
        field,
        UniformSourceProfile(),
        config,
        source_name="h=1",
        surface_maximum=np.full(config.n_s, 2.4),
        surface_maximum_scope="analytic",
    )
    slices = tuple(
        compute_population_slice(context, b, dense_line_assumption=True)
        for b in (1.7, 2.0, 2.3)
    )

    figure, axes = plot_population_diagnostics(
        slices,
        field_label="analytic cosine field",
        source_label="h=1",
        comparison_slices=slices,
    )
    assert "pitch" in axes[0].get_xlabel().lower()
    assert "s" in axes[1].get_xlabel().lower()
    assert "analytic cosine field" in figure._suptitle.get_text()
    assert "comparison grid" in {item.get_text() for item in axes[0].get_legend().texts}
    plt.close(figure)


def test_r0_real_field_evidence_covers_matrix_as_estimates():
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "validation"
        / "r0-population-ledger.json"
    )
    payload = json.loads(path.read_text())
    assert payload["milestone"] == "R0"
    assert payload["source"] == "h(rho)=1 (UniformSourceProfile)"
    assert "no accessibility" in payload["result_scope"]
    assert len(payload["fields"]) == 5
    assert len(payload["cases"]) == 30
    assert {case["lambda_n"] for case in payload["cases"]} == {
        0.05,
        0.1,
        0.5,
        0.8,
        0.9,
        0.95,
    }
    assert all(case["bound_scope"] == "estimate" for case in payload["cases"])
    assert all(
        case["classification"] == "population_estimate_only"
        for case in payload["cases"]
    )
    assert all(case["fine_uncontrolled_errors"] for case in payload["cases"])
    assert any("radial support boundaries" in item for item in payload["assumptions"])
    repository = Path(__file__).resolve().parents[1]
    for field in payload["fields"]:
        assert (
            field["sha256"]
            == hashlib.sha256(
                (repository / "data" / field["file"]).read_bytes()
            ).hexdigest()
        )
    module = repository / "alpha_analysis" / "j_connectivity" / "population.py"
    assert (
        payload["provenance"]["population_module_sha256"]
        == hashlib.sha256(module.read_bytes()).hexdigest()
    )
    assert payload["provenance"]["worker_count"] == 1
    assert payload["provenance"]["population_module_dirty"] is False
    assert payload["convergence_diagnostics"]["maximum_slice_relative_change"] > 0.0
