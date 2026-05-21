from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import netcdf_file
from scipy.interpolate import CubicSpline


class BoozerField:
    """Interpolated Boozer-coordinate field data.

    The full-grid profiles ``G``, ``I``, and ``iota`` are interpolated on the
    VMEC ``s`` grid. The Fourier coefficients ``bmnc`` are interpolated on the
    Boozer surface grid where the transformation was actually computed.
    """

    def __init__(
        self,
        boozmn_file: str | Path | None = None,
        wout_file: str | Path | None = None,
    ) -> None:
        self._reset()

        if boozmn_file is not None and wout_file is not None:
            raise ValueError("Specify only one of boozmn_file or wout_file")

        if boozmn_file is not None:
            self.load_boozmn(boozmn_file)
        elif wout_file is not None:
            self.load_wout(wout_file)

    def _reset(self) -> None:
        self.source: Path | None = None
        self.source_kind: str | None = None
        self.s_full: np.ndarray | None = None
        self.s_bmnc: np.ndarray | None = None
        self.G_grid: np.ndarray | None = None
        self.I_grid: np.ndarray | None = None
        self.iota_grid: np.ndarray | None = None
        self.bmnc_grid: np.ndarray | None = None
        self.mode_m: np.ndarray | None = None
        self.mode_n: np.ndarray | None = None
        self.asym: bool | None = None

        self._G_spline: CubicSpline | None = None
        self._I_spline: CubicSpline | None = None
        self._iota_spline: CubicSpline | None = None
        self._bmnc_spline: CubicSpline | None = None

    @classmethod
    def from_boozmn(cls, boozmn_file: str | Path) -> "BoozerField":
        return cls(boozmn_file=boozmn_file)

    @classmethod
    def from_wout(
        cls,
        wout_file: str | Path,
    ) -> "BoozerField":
        return cls(wout_file=wout_file)

    def load_boozmn(self, boozmn_file: str | Path) -> "BoozerField":
        path = Path(boozmn_file)
        f = netcdf_file(path, mmap=False)
        try:
            self._reset()
            self.source = path
            self.source_kind = "boozmn"

            jlist = f.variables["jlist"][()]
            iota_b = f.variables["iota_b"][()]
            ns = len(iota_b)
            jlist_should_be = np.arange(2, ns + 1)
            if not np.array_equal(jlist, jlist_should_be):
                raise ValueError(f"boozmn file must contain all surfaces. Found jlist={jlist}, expected {jlist_should_be}")

            self.s_full = np.linspace(0.0, 1.0, ns)
            self.ds = self.s_full[1] - self.s_full[0]
            self.s_half = self.s_full[1:] - 0.5 * self.ds
            self.iota_grid = f.variables["iota_b"][()][1:]
            self.I_grid = f.variables["buco_b"][()][1:]
            self.G_grid = f.variables["bvco_b"][()][1:]
            self.bmnc = f.variables["bmnc_b"][()]
            print("bmnc.shape", self.bmnc.shape)

            self._build_splines()
        finally:
            f.close()

        return self

    def load_wout(self, wout_file: str | Path, mboz=32, nboz=32) -> "BoozerField":
        import booz_xform
        bx = booz_xform.Booz_xform()
        bx.read_wout(str(wout_file))
        bx.mboz = mboz
        bx.nboz = nboz
        bx.run()

        self._reset()
        self.source = Path(wout_file)
        self.source_kind = "wout"

        self.asym = bool(getattr(bx, "asym", False))
        self.iota_grid = self._read_xform_1d(bx, ("iota", "iota_b"))
        self.I_grid = self._read_xform_1d(bx, ("Boozer_I_all", "buco_b", "I"))
        self.G_grid = self._read_xform_1d(bx, ("Boozer_G_all", "bvco_b", "G"))
        self.bmnc_grid = self._read_xform_2d(bx, ("bmnc_b",))
        self.mode_m = self._read_xform_1d(bx, ("xm_b", "ixm_b"), allow_missing=True)
        self.mode_n = self._read_xform_1d(bx, ("xn_b", "ixn_b"), allow_missing=True)

        s_bmnc = getattr(bx, "s_b", None)
        if s_bmnc is not None:
            self.s_bmnc = np.asarray(s_bmnc, dtype=float).copy()
        else:
            self.s_bmnc = np.linspace(0.0, 1.0, self.bmnc_grid.shape[1])

        self.s_full = np.linspace(0.0, 1.0, self.iota_grid.size)

        self._build_splines()
        return self

    def G(self, s: np.ndarray | float) -> np.ndarray | float:
        return self._evaluate_spline(self._G_spline, s)

    def I(self, s: np.ndarray | float) -> np.ndarray | float:
        return self._evaluate_spline(self._I_spline, s)

    def iota(self, s: np.ndarray | float) -> np.ndarray | float:
        return self._evaluate_spline(self._iota_spline, s)

    def bmnc(self, s: np.ndarray | float) -> np.ndarray | float:
        return self._evaluate_spline(self._bmnc_spline, s)

    def _build_splines(self) -> None:
        if self.s_full is None or self.G_grid is None or self.I_grid is None or self.iota_grid is None:
            raise ValueError("Full-grid profiles are not loaded")

        self._G_spline = CubicSpline(self.s_half, self.G_grid, axis=0, extrapolate=True)
        self._I_spline = CubicSpline(self.s_half, self.I_grid, axis=0, extrapolate=True)
        self._iota_spline = CubicSpline(self.s_half, self.iota_grid, axis=0, extrapolate=True)
        self._bmnc_spline = CubicSpline(self.s_half, self.bmnc, axis=0, extrapolate=True)

    @staticmethod
    def _evaluate_spline(spline: CubicSpline | None, s: np.ndarray | float) -> np.ndarray | float:
        if spline is None:
            raise ValueError("The requested field has not been loaded")

        value = np.asarray(spline(s))
        if value.ndim == 0:
            return value.item()
        return value

    @staticmethod
    def _read_1d_variable(dataset: Any, name: str) -> np.ndarray:
        variable = dataset.variables[name]
        value = np.array(variable[:], dtype=float, copy=True)
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        return value

    @staticmethod
    def _read_2d_variable(dataset: Any, name: str) -> np.ndarray:
        variable = dataset.variables[name]
        value = np.array(variable[:], dtype=float, copy=True)
        if value.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        return value

    @staticmethod
    def _read_optional_1d_variable(dataset: Any, name: str) -> np.ndarray | None:
        if name not in dataset.variables:
            return None
        return BoozerField._read_1d_variable(dataset, name)

    @staticmethod
    def _read_optional_scalar(dataset: Any, name: str, default: float | int) -> float:
        if name not in dataset.variables:
            return float(default)
        value = np.array(dataset.variables[name][:], dtype=float, copy=True)
        return float(value.reshape(-1)[0])

    def _read_bmnc_grid(self, dataset: Any, expected_count: int) -> np.ndarray:
        bmnc_grid = self._read_2d_variable(dataset, "bmnc_b")

        if "jlist" in dataset.variables:
            jlist = np.array(dataset.variables["jlist"][:], dtype=int, copy=True)
            compute_surfs = jlist - 2
            if compute_surfs.size != expected_count:
                raise ValueError("jlist and bmnc_b have incompatible radial sizes")

            ns_b = self._read_optional_scalar(dataset, "ns_b", default=0)
            if ns_b <= 1:
                raise ValueError("ns_b must be greater than 1 to reconstruct the bmnc grid")

            return (compute_surfs + 0.5) / (ns_b - 1.0)

        return np.linspace(0.0, 1.0, expected_count)

    @staticmethod
    def _read_xform_1d(xform: Any, names: tuple[str, ...], allow_missing: bool = False) -> np.ndarray:
        for name in names:
            if hasattr(xform, name):
                return np.asarray(getattr(xform, name), dtype=float).copy()
        if allow_missing:
            return np.array([], dtype=float)
        raise AttributeError(f"Could not find any of {names} on the booz_xform object")

    @staticmethod
    def _read_xform_2d(xform: Any, names: tuple[str, ...]) -> np.ndarray:
        for name in names:
            if hasattr(xform, name):
                return np.asarray(getattr(xform, name), dtype=float).copy()
        raise AttributeError(f"Could not find any of {names} on the booz_xform object")

    def __repr__(self) -> str:
        if self.source is None:
            return "BoozerField(unloaded)"
        return f"BoozerField(source_kind={self.source_kind!r}, source={str(self.source)!r})"
