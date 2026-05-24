import os

import numpy as np
import matplotlib.pyplot as plt

from alpha_analysis import DATA_DIR, BoozerField, BoozerSurface, find_bounce_points, plot_bounce_points

boozmn_file_name = os.path.join(DATA_DIR, "boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc")
wout_file_name = os.path.join(DATA_DIR, "wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc")

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
