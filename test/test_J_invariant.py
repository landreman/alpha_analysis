import os

import numpy as np

from alpha_analysis import (
    DATA_DIR,
    BoozerField,
    BoozerSurface,
    compute_J_invariant,
)

boozmn_file_name = os.path.join(
    DATA_DIR, "boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
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
