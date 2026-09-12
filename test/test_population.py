from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from alpha_analysis.j_connectivity.denominator import (
    DenominatorConfig,
    UniformSourceProfile,
    compute_denominator,
)
from alpha_analysis.j_connectivity.population import (
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
    # B=B0(1-a^2 sin^2(theta)) has an exact surface maximum B0 and
    # sqrt(1-B/Bmax)=a |sin(theta)|.  The independent angular integrals are
    # elementary, while h(rho)=1+rho^2 makes evaluation at s detectably wrong.
    B0 = 2.0
    a = 0.6
    field = _field(
        nfp=2,
        m=(0, 2),
        n=(0, 0),
        cosine=((B0 * (1.0 - 0.5 * a**2),), (0.5 * B0 * a**2,)),
        G=(2.0, 1.0),
    )

    def source(rho):
        return 1.0 + np.asarray(rho) ** 2

    config = PopulationQuadratureConfig(n_s=4, n_theta=4096, n_zeta=2)
    context = build_population_context(
        field,
        source,
        config,
        source_name="h(rho)=1+rho^2",
        surface_maximum=np.full(config.n_s, B0),
        surface_maximum_scope="analytic",
    )
    denominator = compute_denominator(
        field,
        source,
        DenominatorConfig(config.n_s, config.n_theta, config.n_zeta),
    )
    band = compute_pitch_band_estimate(
        context,
        B0 * (1.0 - a**2),
        B0,
        denominator.V_h,
        dense_line_assumption=True,
    )

    c = 1.0 - a**2
    numerator_theta = 2.0 * a / c + 2.0 * np.arctan(a / np.sqrt(c)) / c**1.5
    denominator_theta = np.pi * (2.0 - a**2) / c**1.5
    expected = numerator_theta / denominator_theta
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
        linewise_trapped=lambda s, theta, zeta, b, B: np.zeros_like(B, dtype=bool),
    )
    surface_upper = compute_population_slice(context, 2.5)

    assert linewise.total_weight == 0.0
    assert linewise.trapping_scope == "linewise_mask"
    assert surface_upper.total_weight > 0.0
    assert surface_upper.trapping_scope == "surface_maximum_upper_only"
    assert surface_upper.bound_scope == "quadrature_estimate_of_upper_model"


def test_missing_weight_uses_total_upper_minus_covered_lower():
    total = WeightBounds(9.0, 11.0)
    reachable = OwnedWeightBounds(
        owner_ids=np.array([10, 11]),
        lower=np.array([1.0, 1.5]),
        upper=np.array([1.2, 1.8]),
    )
    nonreachable = OwnedWeightBounds(
        owner_ids=np.array([20]), lower=np.array([2.0]), upper=np.array([2.4])
    )
    unresolved = OwnedWeightBounds(
        owner_ids=np.array([30]), lower=np.array([0.5]), upper=np.array([0.9])
    )

    ledger = build_population_ledger(total, reachable, nonreachable, unresolved)

    assert ledger.covered_lower == 5.0
    assert ledger.missing_weight_upper == 6.0
    assert ledger.Q_lower == 2.5
    assert ledger.Q_upper == 9.0

    overlapping = OwnedWeightBounds(
        owner_ids=np.array([11]), lower=np.array([0.2]), upper=np.array([0.3])
    )
    with pytest.raises(ValueError, match="overlap"):
        build_population_ledger(total, reachable, nonreachable, overlapping)

    excessive = OwnedWeightBounds(
        owner_ids=np.array([31]), lower=np.array([20.0]), upper=np.array([21.0])
    )
    with pytest.raises(ValueError, match="exceeds total upper"):
        build_population_ledger(total, reachable, nonreachable, excessive)


def test_whole_pitch_band_upper_weight():
    field = _field(m=(0, 1), n=(0, 0), cosine=((2.0,), (0.5,)))
    config = PopulationQuadratureConfig(n_s=3, n_theta=2048, n_zeta=2)
    context = build_population_context(
        field,
        UniformSourceProfile(),
        config,
        source_name="h=1",
        surface_maximum=np.full(config.n_s, 2.5),
        surface_maximum_scope="analytic",
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
    )
    bounds = enclose_pitch_band_fraction(
        band,
        pitch_weight_absolute_error=1.0e-4,
        denominator=WeightBounds(denominator.V_h - 1.0e-4, denominator.V_h + 1.0e-4),
        bound_scope="analytic-field-with-supplied-quadrature-errors",
    )

    assert bounds.lower <= band.fraction <= bounds.upper
    assert bounds.upper - bounds.lower > 0.0
    assert bounds.upper <= bounds.pitch_weight_upper / (2.0 * bounds.denominator.lower)
    assert bounds.bound_scope == "analytic-field-with-supplied-quadrature-errors"


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
        slices, field_label="analytic cosine field", source_label="h=1"
    )
    assert "pitch" in axes[0].get_xlabel().lower()
    assert "s" in axes[1].get_xlabel().lower()
    assert "analytic cosine field" in figure._suptitle.get_text()
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
    assert all(len(field["sha256"]) == 64 for field in payload["fields"])
    assert len(payload["provenance"]["population_module_sha256"]) == 64
    assert payload["provenance"]["worker_count"] == 1
    assert payload["provenance"]["population_module_dirty"] is False
    assert payload["convergence_diagnostics"]["maximum_slice_relative_change"] > 0.0
