import os

import numpy as np
from scipy.io import netcdf_file
import matplotlib.pyplot as plt

from alpha_analysis import DATA_DIR, BoozerField, BoozerSurface

boozmn_file_name = os.path.join(
    DATA_DIR, "boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
)
wout_file_name = os.path.join(
    DATA_DIR, "wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
)


def test_load_boozmn():
    b = BoozerField.from_boozmn(boozmn_file_name)

    with netcdf_file(wout_file_name, mmap=False) as f:
        np.testing.assert_allclose(
            b.iota_data, f.variables["iotas"][()][1:], atol=0, rtol=1e-15
        )
        np.testing.assert_allclose(
            b.I_data, f.variables["buco"][()][1:], atol=0, rtol=1e-15
        )
        np.testing.assert_allclose(
            b.G_data, f.variables["bvco"][()][1:], atol=0, rtol=1e-15
        )

    # Evaluate G, I, iota, and bmnc at the s_half points to make sure the
    # results perfectly match the values in the boozmn file.
    np.testing.assert_allclose(b.G(b.s_half), b.G_data, atol=0, rtol=1e-15)
    np.testing.assert_allclose(b.I(b.s_half), b.I_data, atol=0, rtol=1e-15)
    np.testing.assert_allclose(b.iota(b.s_half), b.iota_data, atol=0, rtol=1e-15)
    np.testing.assert_allclose(b.bmnc(b.s_half), b.bmnc_data, atol=1e-15, rtol=1e-15)

    # Make sure we can evaluate all the splines at s=0 and s=1:
    s = np.array([0.0, 1.0])
    assert np.isfinite(b.G(s)).all()
    assert np.isfinite(b.I(s)).all()
    assert np.isfinite(b.iota(s)).all()
    assert np.isfinite(b.bmnc(s)).all()


def test_compute_B_1d_shape_and_values():
    booz = BoozerField.from_boozmn(boozmn_file_name)
    s = booz.s_half

    nthetaphi = 9
    theta = np.zeros(nthetaphi)
    phi = np.zeros(nthetaphi)

    B = booz.compute_B(s, theta, phi)
    assert B.shape == (s.size, nthetaphi)

    expected = np.sum(booz.bmnc_data, axis=1, keepdims=True)
    np.testing.assert_allclose(B, np.repeat(expected, nthetaphi, axis=1), rtol=1e-13, atol=1e-13)


def test_compute_B_2d_shape_matches_flattened():
    booz = BoozerField.from_boozmn(boozmn_file_name)
    s = booz.s_half

    n1, n2 = 3, 4
    theta_2d = np.linspace(0.0, 2.0 * np.pi, n1 * n2, endpoint=False).reshape(n1, n2)
    phi_2d = np.linspace(0.0, np.pi, n1 * n2, endpoint=False).reshape(n1, n2)

    B_2d = booz.compute_B(s, theta_2d, phi_2d)
    assert B_2d.shape == (s.size, n1, n2)

    B_flat = booz.compute_B(s, theta_2d.reshape(-1), phi_2d.reshape(-1))
    np.testing.assert_allclose(B_2d.reshape(s.size, -1), B_flat, rtol=1e-13, atol=1e-13)


def test_compute_B_field_surface_agree_scalar_s():
    booz = BoozerField.from_boozmn(boozmn_file_name)
    s = 0.5
    surf = booz.surface(s)

    theta = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi / booz.nfp, 12, endpoint=False)
    B_field = booz.compute_B(s, theta, phi)
    B_surf = surf.compute_B(theta, phi)
    np.testing.assert_allclose(B_field, B_surf, rtol=1e-13, atol=1e-13)

    phi2d, theta2d = np.meshgrid(phi, theta)
    B_field_2d = booz.compute_B(s, theta2d, phi2d)
    B_surf_2d = surf.compute_B(theta2d, phi2d)
    np.testing.assert_allclose(B_field_2d, B_surf_2d, rtol=1e-13, atol=1e-13)


def test_compute_B_field_surface_agree_multiple_s():
    booz = BoozerField.from_boozmn(boozmn_file_name)
    s_vals = np.array([0.2, 0.5, 0.8])
    theta = np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi / booz.nfp, 7, endpoint=False)

    B_field = booz.compute_B(s_vals, theta, phi)
    for j, s in enumerate(s_vals):
        surf = booz.surface(float(s))
        B_surf = surf.compute_B(theta, phi)
        np.testing.assert_allclose(B_field[j], B_surf, rtol=1e-13, atol=1e-13)


def test_B_reference():
    """Compare B to reference values from a W7-X boozmn file."""
    booz = BoozerField.from_boozmn(boozmn_file_name)

    make_plot = False

    if make_plot:
        ntheta = 30
        nphi = 31
    else:
        ntheta = 3
        nphi = 4

    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi / booz.nfp, nphi, endpoint=False)
    phi2d, theta2d = np.meshgrid(phi, theta)
    s = 0.5
    surf = booz.surface(s)
    B = surf.compute_B(theta2d, phi2d)
    B_reference = np.array(
        [
            [
                2.743048654229539,
                2.566500171993845,
                2.421530030838217,
                2.566500171993845,
            ],
            [
                3.067412946539301,
                2.576178573264619,
                2.339680548217751,
                2.852598410500511,
            ],
            [3.0674129465393, 2.852598410500511, 2.339680548217752, 2.576178573264619],
        ]
    )

    if make_plot:
        plt.contourf(phi2d, theta2d, B, levels=25)
        plt.colorbar()
        plt.show()
    else:
        # np.set_printoptions(precision=15)
        # print(B)
        np.testing.assert_allclose(B, B_reference, rtol=1e-13, atol=1e-13)


def test_get_min_max():
    booz = BoozerField.from_boozmn(boozmn_file_name)
    B_min, B_max = booz.get_min_max()
    print("B_min, B_max:", B_min, B_max)

    # Reference values from a W7-X boozmn file:
    B_min_ref = 2.293715871749871
    B_max_ref = 3.370464075302584

    np.testing.assert_allclose(B_min, B_min_ref, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(B_max, B_max_ref, rtol=1e-13, atol=1e-13)

    # Try a different resolution:
    B_min, B_max = booz.get_min_max(n_s=20, n_theta=32, n_phi=33)
    np.testing.assert_allclose(B_min, B_min_ref, rtol=2e-4)
    np.testing.assert_allclose(B_max, B_max_ref, rtol=1e-13, atol=1e-13)
