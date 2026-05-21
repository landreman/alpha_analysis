import os

import numpy as np
from scipy.io import netcdf_file

from alpha_analysis import DATA_DIR, BoozerField

boozmn_file_name = os.path.join(DATA_DIR, "boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc")
wout_file_name = os.path.join(DATA_DIR, "wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc")

def test_load_boozmn():
    b = BoozerField.from_boozmn(boozmn_file_name)

    with netcdf_file(wout_file_name, mmap=False) as f:
        np.testing.assert_allclose(b.iota_data, f.variables["iotas"][()][1:], atol=0, rtol=1e-15)
        np.testing.assert_allclose(b.I_data, f.variables["buco"][()][1:], atol=0, rtol=1e-15)
        np.testing.assert_allclose(b.G_data, f.variables["bvco"][()][1:], atol=0, rtol=1e-15)

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
