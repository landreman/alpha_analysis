import os

import numpy as np
import alpha_analysis.J_invariant as J_invariant_module

from alpha_analysis import (
    DATA_DIR,
    BoozerField,
    BoozerSurface,
    compute_J_invariant,
)

boozmn_file_name = os.path.join(
    DATA_DIR, "boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
)


def test_compute_J_invariant_reference_grid():
    booz = BoozerField.from_boozmn(boozmn_file_name)

    alpha_values = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
    rho_values = np.array([0.5, 1.0])
    s_values = rho_values**2
    lambda_n_values = np.array([0.1, 0.3, 0.5, 0.8])

    b_min, b_max = booz.get_min_max()
    phi_center = np.pi / booz.nfp

    j_reference = np.array(
        [
            [
                [0.00873977512318135, 0.023394976504410036],
                [0.043955587975138856, 0.02417013078841274],
                [0.047411593347227256, 0.021248913705857228],
                [0.023917904151112786, 0.024170130788412694],
            ],
            [
                [0.14376622898855615, 0.091361876815906],
                [0.1488120349311294, 0.11350986852550063],
                [0.15033078696630559, 0.11476743295690289],
                [0.14665286628320484, 0.0873463502918537],
            ],
            [
                [0.5668853990400646, 0.4598254566459877],
                [0.253930388305994, 0.1987444796980752],
                [0.25089923422733934, 0.20052247249992536],
                [0.8453458338883063, 0.4396318158857731],
            ],
            [
                [np.nan, 1.4271735115821336],
                [np.nan, 1.4192534097688618],
                [np.nan, 0.3250939627968473],
                [np.nan, 1.4192534097688614],
            ],
        ]
    )

    j_computed = np.empty_like(j_reference)

    for lambda_idx, lambda_n in enumerate(lambda_n_values):
        b_bounce = b_min + lambda_n * (b_max - b_min)

        for rho_idx, s in enumerate(s_values):
            surf = BoozerSurface(booz, s)
            for alpha_idx, alpha in enumerate(alpha_values):
                theta_center = alpha + surf.iota * phi_center
                data = compute_J_invariant(
                    surf,
                    b_bounce,
                    theta_center,
                    phi_center,
                    n_phi=1001,
                    phi_margin=5.0,
                    refine=True,
                )
                j_computed[lambda_idx, alpha_idx, rho_idx] = data["J"]

    np.testing.assert_allclose(
        j_computed,
        j_reference,
        rtol=1e-13,
        atol=1e-13,
        equal_nan=True,
    )

def test_J_refine_doesnt_change_too_much():
    """The J invariant computed with refine=True should be close to that found with refine=False."""
    booz = BoozerField.from_boozmn(boozmn_file_name)
    s = 0.5
    surf = BoozerSurface(booz, s)

    B_bounces = [0.1, 2.4, 2.7, 3.1, 5.1]
    alphas = np.linspace(0.0, 2.0 * np.pi, 10)
    phi_center = np.pi / surf.nfp
    n_phi = 1001
    phi_margin = 4.0

    for B_bounce in B_bounces:
        for alpha in alphas:
            data_refined = compute_J_invariant(
                surf,
                B_bounce,
                alpha,
                phi_center,
                n_phi=n_phi,
                phi_margin=phi_margin,
                refine=True,
            )

            data_unrefined = compute_J_invariant(
                surf,
                B_bounce,
                alpha,
                phi_center,
                n_phi=n_phi,
                phi_margin=phi_margin,
                refine=False,
            )

            np.testing.assert_allclose(
                data_refined["J"], data_unrefined["J"], atol=1e-14, rtol=0.004
            )


def test_plot_J_invariant_cli_forwards_args(monkeypatch):
    captured = {}

    def _fake_plot_J_invariant(
        boozmn_file,
        n_alpha,
        n_rho,
        contour_levels,
        refine,
        show=True,
    ):
        captured["boozmn_file"] = boozmn_file
        captured["n_alpha"] = n_alpha
        captured["n_rho"] = n_rho
        captured["contour_levels"] = contour_levels
        captured["refine"] = refine
        captured["show"] = show

    monkeypatch.setattr(J_invariant_module, "plot_J_invariant", _fake_plot_J_invariant)
    exit_code = J_invariant_module.plot_J_invariant_cli(
        [
            boozmn_file_name,
            "--n_alpha",
            "12",
            "--n_rho",
            "15",
            "--contour_levels",
            "27",
            "--no-refine",
        ]
    )

    assert exit_code == 0
    assert captured["boozmn_file"] == boozmn_file_name
    assert captured["n_alpha"] == 12
    assert captured["n_rho"] == 15
    assert captured["contour_levels"] == 27
    assert captured["refine"] is False
    assert captured["show"] is True
