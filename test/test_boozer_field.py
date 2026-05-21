import os

import numpy as np
from scipy.io import netcdf_file

from alpha_analysis import DATA_DIR, BoozerField

boozmn_file_name = os.path.join(DATA_DIR, "boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc")
wout_file_name = os.path.join(DATA_DIR, "wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc")

def test_load_boozmn():
    b = BoozerField.from_boozmn(boozmn_file_name)

    with netcdf_file(wout_file_name, mmap=False) as f:
        np.testing.assert_allclose(b.iota_grid, f.variables["iotas"][()][1:], atol=0, rtol=1e-15)
        np.testing.assert_allclose(b.I_grid, f.variables["buco"][()][1:], atol=0, rtol=1e-15)
        np.testing.assert_allclose(b.G_grid, f.variables["bvco"][()][1:], atol=0, rtol=1e-15)

