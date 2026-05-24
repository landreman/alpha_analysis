import os

import numpy as np
import alpha_analysis.bounce_points as bounce_points_module

from alpha_analysis import DATA_DIR, BoozerField, BoozerSurface, find_bounce_points, plot_bounce_points

boozmn_file_name = os.path.join(DATA_DIR, "boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc")

def test_refine_doesnt_change_too_much():
    """The bounce points found with refine=True should be close to those found with refine=False."""
    booz = BoozerField.from_boozmn(boozmn_file_name)
    s = 0.5
    surf = BoozerSurface(booz, s)

    B_bounce = 2.7
    alphas = np.linspace(0.0, 2.0 * np.pi, 10)
    phi_center = np.pi / surf.nfp

    for alpha in alphas:
        data_refined = find_bounce_points(
            surf,
            B_bounce,
            alpha,
            phi_center,
            refine=True,
        )

        data_unrefined = find_bounce_points(
            surf,
            B_bounce,
            alpha,
            phi_center,
            refine=False,
        )

        for key in ["left_index", "right_index", "B", "allowed", "well_crosses_left_edge", "well_crosses_right_edge"]:
            np.testing.assert_array_equal(data_refined[key], data_unrefined[key])

        for key in ["phi_left", "phi_right", "theta_left", "theta_right"]:
            np.testing.assert_allclose(data_refined[key], data_unrefined[key], atol=0.03, rtol=0.03)

def test_bounce_points_off_edges():
    """Make sure find_bounce_points works as expected when the well crosses the edges of the phi grid."""
    booz = BoozerField.from_boozmn(boozmn_file_name)
    s = 0.5
    surf = BoozerSurface(booz, s)

    B_bounce = 5.0  # Larger than B_max
    alphas = [0, np.pi]
    phi_center = np.pi / surf.nfp
    n_phi = 101

    for alpha in alphas:
        for refine in [True, False]:
            data = find_bounce_points(
                surf,
                B_bounce,
                alpha,
                phi_center,
                n_phi=n_phi,
                phi_margin=1.1,
                refine=refine,
            )

            assert data["well_crosses_left_edge"]
            assert data["well_crosses_right_edge"]
            np.testing.assert_equal(data["left_index"], 0)
            np.testing.assert_equal(data["right_index"], n_phi - 1)


def test_bounce_points_none():
    """Make sure find_bounce_points works as expected when there are no allowed points."""
    booz = BoozerField.from_boozmn(boozmn_file_name)
    s = 0.5
    surf = BoozerSurface(booz, s)

    B_bounce = 0.9  # Smaller than B_min
    alphas = [0, np.pi]
    phi_center = np.pi / surf.nfp
    n_phi = 101

    for alpha in alphas:
        for refine in [True, False]:
            data = find_bounce_points(
                surf,
                B_bounce,
                alpha,
                phi_center,
                n_phi=n_phi,
                phi_margin=1.1,
                refine=refine,
            )
            assert not np.any(data["allowed"])
            assert np.isnan(data["left_index"])
            assert np.isnan(data["right_index"])
            assert np.isnan(data["phi_left"])
            assert np.isnan(data["phi_right"])


def test_plot_bounce_points_doesnt_crash():
    """Make a plot of bounce points for a W7-X boozmn file."""
    for refine in [True, False]:
        for B_bounce in [2.4, 2.7, 3.1]:
            plot_bounce_points(
                boozmn_file_name,
                0.5,
                B_bounce,
                refine=refine,
                show=False,
            )


def test_plot_bounce_points_cli_forwards_args(monkeypatch):
    captured = {}

    def _fake_plot_bounce_points(filename, s, B_bounce, n_alpha, n_phi, phi_margin, refine, show=True):
        captured["filename"] = filename
        captured["s"] = s
        captured["B_bounce"] = B_bounce
        captured["n_alpha"] = n_alpha
        captured["n_phi"] = n_phi
        captured["phi_margin"] = phi_margin
        captured["refine"] = refine
        captured["show"] = show

    monkeypatch.setattr(bounce_points_module, "plot_bounce_points", _fake_plot_bounce_points)
    exit_code = bounce_points_module.plot_bounce_points_cli(
        [
            boozmn_file_name,
            "0.25",
            "2.7",
            "--n_alpha",
            "12",
            "--n_phi",
            "301",
            "--phi_margin",
            "4.5",
            "--no-refine",
        ]
    )

    assert exit_code == 0
    assert captured["filename"] == boozmn_file_name
    assert captured["s"] == 0.25
    assert captured["B_bounce"] == 2.7
    assert captured["n_alpha"] == 12
    assert captured["n_phi"] == 301
    assert captured["phi_margin"] == 4.5
    assert captured["refine"] is False
    assert captured["show"] is True
