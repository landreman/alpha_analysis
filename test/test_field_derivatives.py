from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
from scipy.io import netcdf_file

from alpha_analysis import BoozerField
from alpha_analysis.j_connectivity.field import BoozerFieldLike
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField


def _synthetic_field() -> SyntheticFourierField:
    return SyntheticFourierField(
        nfp=3,
        m=np.array([0, 1, 2, 3]),
        n=np.array([0, 3, -6, 9]),
        # Coefficients are ordered by ascending power of s.
        cosine_coefficients=np.array(
            [[2.0, 0.1], [0.3, -0.2], [-0.1, 0.4], [0.05, 0.0]]
        ),
        sine_coefficients=np.array(
            [[0.0, 0.0], [0.2, 0.1], [0.07, -0.05], [-0.03, 0.02]]
        ),
        iota_coefficients=np.array([0.7, 0.2]),
        G_coefficients=np.array([-4.0, 0.1]),
        I_coefficients=np.array([0.5]),
    )


def test_synthetic_fourier_derivatives_match_finite_differences():
    field = _synthetic_field()
    s = np.array([0.23, 0.51, 0.84])
    theta = np.array([0.31, 1.27, 5.83])
    zeta = np.array([0.17, 0.92, 1.71])
    h = 2.0e-6

    np.testing.assert_allclose(
        field.dB_ds(s, theta, zeta),
        (field.B(s + h, theta, zeta) - field.B(s - h, theta, zeta)) / (2 * h),
        rtol=2e-9,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        field.dB_dtheta(s, theta, zeta),
        (field.B(s, theta + h, zeta) - field.B(s, theta - h, zeta)) / (2 * h),
        rtol=2e-9,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        field.dB_dzeta(s, theta, zeta),
        (field.B(s, theta, zeta + h) - field.B(s, theta, zeta - h)) / (2 * h),
        rtol=2e-9,
        atol=2e-10,
    )


def test_parallel_derivatives_follow_field_line_and_include_sine_modes():
    field = _synthetic_field()
    s = np.array([0.18, 0.47, 0.79])
    theta = np.array([0.29, 2.13, 4.72])
    zeta = np.array([0.11, 0.73, 1.56])
    h = 2.0e-5
    iota = field.iota(s)

    forward = field.B(s, theta + iota * h, zeta + h)
    backward = field.B(s, theta - iota * h, zeta - h)
    np.testing.assert_allclose(
        field.D_B(s, theta, zeta),
        (forward - backward) / (2 * h),
        rtol=2e-8,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        field.D2_B(s, theta, zeta),
        (forward - 2 * field.B(s, theta, zeta) + backward) / h**2,
        rtol=3e-6,
        atol=3e-6,
    )


def test_periodicity_protocol_and_signed_C():
    field = _synthetic_field()
    s = np.array([0.2, 0.5, 0.8])
    theta = np.array([0.3, 1.1, 4.9])
    zeta = np.array([0.2, 0.7, 1.4])

    assert isinstance(field, BoozerFieldLike)
    np.testing.assert_allclose(
        field.B(s, theta + 2 * np.pi, zeta), field.B(s, theta, zeta)
    )
    np.testing.assert_allclose(
        field.B(s, theta, zeta + 2 * np.pi / field.nfp), field.B(s, theta, zeta)
    )
    np.testing.assert_allclose(field.C(s), field.G(s) + field.iota(s) * field.I(s))
    assert np.all(field.C(s) < 0.0)


def _write_asymmetric_boozmn(path) -> None:
    with netcdf_file(path, "w") as f:
        f.createDimension("radius", 4)
        f.createDimension("comput_surfs", 3)
        f.createDimension("pack_rad", 3)
        f.createDimension("mn_modes", 2)

        f.createVariable("jlist", "i", ("comput_surfs",))[:] = [2, 3, 4]
        for name, values in {
            "iota_b": [0.0, 0.5, 0.6, 0.7],
            "buco_b": [0.0, 0.1, 0.1, 0.1],
            "bvco_b": [0.0, 4.0, 4.0, 4.0],
        }.items():
            f.createVariable(name, "d", ("radius",))[:] = values
        f.createVariable("bmnc_b", "d", ("pack_rad", "mn_modes"))[:] = [
            [2.0 - 0.1 / 3, 0.3 + 0.2 / 3],
            [2.0, 0.3],
            [2.0 + 0.1 / 3, 0.3 - 0.2 / 3],
        ]
        f.createVariable("bmns_b", "d", ("pack_rad", "mn_modes"))[:] = [
            [0.0, 0.2 - 0.1 / 3],
            [0.0, 0.2],
            [0.0, 0.2 + 0.1 / 3],
        ]
        f.createVariable("ixm_b", "i", ("mn_modes",))[:] = [0, 1]
        f.createVariable("ixn_b", "i", ("mn_modes",))[:] = [0, 2]
        f.createVariable("nfp_b", "i", ())[...] = 2
        f.createVariable("rmnc_b", "d", ("pack_rad", "mn_modes"))[:] = 1.0
        f.createVariable("lasym__logical__", "i", ())[...] = 1


def test_boozer_field_loads_sine_modes_and_implements_field_protocol(tmp_path):
    path = tmp_path / "asymmetric_boozmn.nc"
    _write_asymmetric_boozmn(path)
    field = BoozerField.from_boozmn(path)
    theta = np.array([0.0, np.pi / 2])
    zeta = np.zeros(2)

    assert isinstance(field, BoozerFieldLike)
    assert field.asym is True
    np.testing.assert_allclose(field.B(0.5, theta, zeta), [2.3, 2.2], atol=1e-14)
    np.testing.assert_allclose(
        field.dB_dtheta(0.5, theta, zeta), [0.2, -0.3], atol=1e-14
    )
    # Keep the legacy API equivalent for asymmetric data as well.
    np.testing.assert_allclose(
        field.compute_B(0.5, theta, zeta), field.B(0.5, theta, zeta)
    )


def test_loaded_boozer_field_analytic_derivatives_match_finite_differences(tmp_path):
    path = tmp_path / "asymmetric_boozmn.nc"
    _write_asymmetric_boozmn(path)
    field = BoozerField.from_boozmn(path)
    s = np.array([0.31, 0.57, 0.76])
    theta = np.array([0.2, 1.4, 5.1])
    zeta = np.array([0.1, 0.8, 2.2])
    h = 2.0e-5

    np.testing.assert_allclose(
        field.dB_ds(s, theta, zeta),
        (field.B(s + h, theta, zeta) - field.B(s - h, theta, zeta)) / (2 * h),
        rtol=2e-9,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        field.dB_dtheta(s, theta, zeta),
        (field.B(s, theta + h, zeta) - field.B(s, theta - h, zeta)) / (2 * h),
        rtol=2e-9,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        field.dB_dzeta(s, theta, zeta),
        (field.B(s, theta, zeta + h) - field.B(s, theta, zeta - h)) / (2 * h),
        rtol=2e-9,
        atol=2e-10,
    )
    iota = field.iota(s)
    forward = field.B(s, theta + iota * h, zeta + h)
    backward = field.B(s, theta - iota * h, zeta - h)
    np.testing.assert_allclose(
        field.D_B(s, theta, zeta),
        (forward - backward) / (2 * h),
        rtol=2e-8,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        field.D2_B(s, theta, zeta),
        (forward - 2 * field.B(s, theta, zeta) + backward) / h**2,
        rtol=3e-6,
        atol=3e-6,
    )


def test_wout_adapter_transposes_and_loads_booz_xform_sine_modes(monkeypatch, tmp_path):
    class FakeBoozXform:
        def read_wout(self, _path):
            self.asym = True
            self.nfp = 2
            self.s_in = np.array([0.0, 1 / 3, 2 / 3, 1.0])
            self.iota = np.array([0.5, 0.55, 0.6, 0.65])
            self.Boozer_I_all = np.full(4, 0.1)
            self.Boozer_G_all = np.full(4, 4.0)

        def run(self):
            self.s_b = np.array([1 / 6, 0.5, 5 / 6])
            self.xm_b = np.array([0, 1])
            self.xn_b = np.array([0, 2])
            # booz_xform's public arrays have shape (mode, surface).
            self.bmnc_b = np.array([[2.0, 2.0, 2.0], [0.3, 0.3, 0.3]])
            self.bmns_b = np.array([[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]])
            self.rmnc_b = np.array([[5.5, 5.5, 5.5], [0.1, 0.1, 0.1]])

    monkeypatch.setitem(
        sys.modules, "booz_xform", SimpleNamespace(Booz_xform=FakeBoozXform)
    )
    field = BoozerField.from_wout(tmp_path / "unused-wout.nc")

    assert field.bmnc_data.shape == (3, 2)
    np.testing.assert_allclose(field.bmns_data[:, 1], 0.2)
    np.testing.assert_allclose(field.B(0.5, np.pi / 2, 0.0), 2.2)
    assert field.nfp == 2
    assert field.R00 == 5.5


def test_fourier_quantities_include_sine_modes_in_both_trig_branches(tmp_path):
    # Asymmetric data exercises the sine series of the fused evaluation; the
    # tiny mode table keeps the angle-difference factorization off by
    # default, so force it on to check that branch's algebra as well.
    path = tmp_path / "asymmetric_boozmn.nc"
    _write_asymmetric_boozmn(path)
    field = BoozerField.from_boozmn(path)
    assert field._has_sine
    assert not field._trig_factorized
    s = np.array([0.31, 0.57, 0.76])
    theta = np.array([0.2, 1.4, 5.1])
    zeta = np.array([0.1, 0.8, 2.2])
    names = ("B", "dB_ds", "dB_dtheta", "dB_dzeta", "D_B", "D2_B")
    reference = tuple(getattr(field, name)(s, theta, zeta) for name in names)
    field._trig_factorized = True
    bundle = field.fourier_quantities(s, theta, zeta, names)
    for name, expected, values in zip(names, reference, bundle):
        np.testing.assert_allclose(
            values, expected, rtol=1e-13, atol=1e-13, err_msg=name
        )
