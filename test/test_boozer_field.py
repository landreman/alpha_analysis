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
    np.testing.assert_allclose(
        B, np.repeat(expected, nthetaphi, axis=1), rtol=1e-13, atol=1e-13
    )


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


def test_compute_B_tensor_alpha_phi_matches_compute_B():
    booz = BoozerField.from_boozmn(boozmn_file_name)
    surf = booz.surface(0.5)

    alpha = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi / booz.nfp, 9, endpoint=False)

    phi2d, alpha2d = np.meshgrid(phi, alpha)
    theta2d = alpha2d + surf.iota * phi2d

    B_reference = surf.compute_B(theta2d, phi2d)
    B_tensor = surf.compute_B_tensor_alpha_phi(alpha, phi)

    assert B_tensor.shape == (alpha.size, phi.size)
    np.testing.assert_allclose(B_tensor, B_reference, rtol=1e-13, atol=1e-13)


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


def test_B_is_axis_regular():
    """The Fourier interpolation must be single-valued and continuous at the
    axis, with the physical rho^|m| harmonic scaling (DESIGN.md section 7.3,
    option 1).  Unconstrained spline extrapolation below the innermost
    coefficient surface leaves m != 0 harmonics finite at s=0 and makes |B|
    multivalued there by about 1e-3 for this file."""
    booz = BoozerField.from_boozmn(boozmn_file_name)
    s0 = float(booz.s_bmnc[0])
    zeta = 0.3

    # Single-valued at the axis: the poloidal spread of B at s -> 0 must
    # vanish like sqrt(s), so at s = 1e-14 it is far below the ~1e-3 spread
    # the unconstrained extrapolation produces.
    theta = np.linspace(0.0, 2.0 * np.pi, 17)
    spread = np.ptp(booz.B(1.0e-14, theta, zeta))
    assert spread < 1.0e-6

    # Continuous across the innermost coefficient surface where the
    # continuation hands over to the spline.
    below = float(booz.B(s0 * (1.0 - 1.0e-9), 0.7, zeta))
    above = float(booz.B(s0 * (1.0 + 1.0e-9), 0.7, zeta))
    np.testing.assert_allclose(below, above, rtol=0.0, atol=1.0e-6)

    # Harmonic scaling: the m-odd part of B scales as sqrt(s) in the
    # continuation region, so quartering s halves the poloidal asymmetry.
    def odd_part(s):
        return 0.5 * float(booz.B(s, 0.0, zeta) - booz.B(s, np.pi, zeta))

    ratio = odd_part(s0 / 4.0) / odd_part(s0 / 16.0)
    np.testing.assert_allclose(ratio, 2.0, rtol=0.05)

    # The continued radial derivative must be consistent with B itself.
    s_probe, step = s0 / 2.0, s0 * 1.0e-5
    finite_difference = (
        float(booz.B(s_probe + step, 0.7, zeta) - booz.B(s_probe - step, 0.7, zeta))
    ) / (2.0 * step)
    np.testing.assert_allclose(
        float(booz.dB_ds(s_probe, 0.7, zeta)), finite_difference, rtol=1.0e-5
    )


def test_fourier_quantities_match_direct_mode_sums():
    # The fused evaluation (shared phase table, skipped sine series,
    # factorized trigonometry, chunking) must reproduce the plain §7.1 mode
    # sums written out directly from the interpolated coefficients, through
    # both trigonometric branches, and chunked evaluation must equal
    # unchunked exactly.
    field = BoozerField.from_boozmn(boozmn_file_name)
    rng = np.random.default_rng(7)
    n = 23
    s = rng.uniform(0.05, 0.95, n)
    theta = rng.uniform(-np.pi, np.pi, n)
    zeta = rng.uniform(0.0, 2.0 * np.pi / field.nfp, n)

    phase = theta[:, np.newaxis] * field.xm - zeta[:, np.newaxis] * field.xn
    cosine = field.bmnc(s)
    sine = field.bmns(s)
    first = -cosine * np.sin(phase) + sine * np.cos(phase)
    k = np.asarray(field.iota(s))[:, np.newaxis] * field.xm - field.xn
    names = ("B", "dB_dtheta", "dB_dzeta", "D_B", "D2_B")
    expected = (
        np.sum(cosine * np.cos(phase) + sine * np.sin(phase), axis=-1),
        np.sum(field.xm * first, axis=-1),
        np.sum(-field.xn * first, axis=-1),
        np.sum(k * first, axis=-1),
        -np.sum(k**2 * (cosine * np.cos(phase) + sine * np.sin(phase)), axis=-1),
    )

    assert field._trig_factorized  # the W7-X mode table reuses few m and n
    for factorized in (True, False):
        field._trig_factorized = factorized
        bundle = field.fourier_quantities(s, theta, zeta, names)
        for name, reference, values in zip(names, expected, bundle):
            np.testing.assert_allclose(
                values, reference, rtol=1e-12, atol=1e-12, err_msg=name
            )
    field._trig_factorized = True

    # Forcing many small chunks must not change any value.
    whole = field.fourier_quantities(s, theta, zeta, names)
    field._FOURIER_CHUNK_ELEMENTS = 5 * field.xm.size
    chunked = field.fourier_quantities(s, theta, zeta, names)
    del field._FOURIER_CHUNK_ELEMENTS
    for name, big, small in zip(names, whole, chunked):
        np.testing.assert_array_equal(big, small, err_msg=name)

    # The protocol methods route through the same fused path; spot-check the
    # scalar shape contract they rely on.
    assert np.shape(field.B(0.3, 0.2, 0.1)) == ()
