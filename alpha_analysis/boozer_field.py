from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import netcdf_file
from scipy.interpolate import CubicSpline


@lru_cache(maxsize=64)
def _fourier_plan(quantities: tuple[str, ...]):
    """Cache which tables a fourier_quantities request needs.

    Returns ``(needed, need_base, need_first, need_value, need_ds, need_k)``
    where ``needed`` is the deduplicated quantity tuple; the flags select the
    trigonometric tables, coefficient splines, and ``k = m iota - n`` array.
    """
    needed = tuple(dict.fromkeys(quantities))
    need_base = any(name in ("B", "dB_ds", "D2_B") for name in needed)
    need_first = any(name in ("dB_dtheta", "dB_dzeta", "D_B") for name in needed)
    need_value = any(name != "dB_ds" for name in needed)
    need_ds = "dB_ds" in needed
    need_k = any(name in ("D_B", "D2_B") for name in needed)
    return needed, need_base, need_first, need_value, need_ds, need_k


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
        self.s_half: np.ndarray | None = None
        self.G_data: np.ndarray | None = None
        self.I_data: np.ndarray | None = None
        self.iota_data: np.ndarray | None = None
        self.bmnc_data: np.ndarray | None = None
        self.bmns_data: np.ndarray | None = None
        self.xm: np.ndarray | None = None
        self.xn: np.ndarray | None = None
        self.asym: bool | None = None
        self.nfp: int | None = None
        self.R00: float | None = None

        self._G_spline: CubicSpline | None = None
        self._I_spline: CubicSpline | None = None
        self._iota_spline: CubicSpline | None = None
        self._bmnc_spline: CubicSpline | None = None
        self._bmns_spline: CubicSpline | None = None
        self._coefficient_s0: float | None = None
        self._has_sine: bool = False
        self._xm_float: np.ndarray | None = None
        self._xn_float: np.ndarray | None = None
        self._m_unique: np.ndarray | None = None
        self._m_index: np.ndarray | None = None
        self._n_unique: np.ndarray | None = None
        self._n_index: np.ndarray | None = None
        self._trig_factorized: bool = False

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
                raise ValueError(
                    f"boozmn file must contain all surfaces. Found jlist={jlist}, expected {jlist_should_be}"
                )

            self.s_full = np.linspace(0.0, 1.0, ns)
            self.ds = self.s_full[1] - self.s_full[0]
            self.s_half = self.s_full[1:] - 0.5 * self.ds
            self.iota_data = f.variables["iota_b"][()][1:]
            self.I_data = f.variables["buco_b"][()][1:]
            self.G_data = f.variables["bvco_b"][()][1:]
            self.bmnc_data = f.variables["bmnc_b"][()]
            self.asym = bool(f.variables.get("lasym__logical__", np.array(0))[()])
            bmns_variable = f.variables.get("bmns_b")
            if bmns_variable is None:
                if self.asym:
                    raise ValueError("asymmetric boozmn data is missing bmns_b")
                self.bmns_data = np.zeros_like(self.bmnc_data)
            else:
                self.bmns_data = bmns_variable[()]
            self.xm = f.variables["ixm_b"][()]
            self.xn = f.variables["ixn_b"][()]
            self.nfp = int(f.variables["nfp_b"][()])
            self.R00 = float(f.variables["rmnc_b"][()][0, 0])
            self.s_bmnc = self.s_half.copy()

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
        self.iota_data = self._read_xform_1d(bx, ("iota", "iota_b"))
        self.I_data = self._read_xform_1d(bx, ("Boozer_I_all", "buco_b", "I"))
        self.G_data = self._read_xform_1d(bx, ("Boozer_G_all", "bvco_b", "G"))
        # booz_xform exposes (mode, surface), whereas boozmn NetCDF stores
        # (surface, mode).  Keep one internal convention for the spline axis.
        self.bmnc_data = self._read_xform_2d(bx, ("bmnc_b",)).T
        try:
            bmns = self._read_xform_2d(bx, ("bmns_b",))
        except AttributeError:
            if self.asym:
                raise ValueError(
                    "asymmetric booz_xform output is missing bmns_b"
                ) from None
            self.bmns_data = np.zeros_like(self.bmnc_data)
        else:
            if bmns.size == 0:
                if self.asym:
                    raise ValueError("asymmetric booz_xform output has empty bmns_b")
                self.bmns_data = np.zeros_like(self.bmnc_data)
            else:
                self.bmns_data = bmns.T
        self.xm = self._read_xform_1d(bx, ("xm_b", "ixm_b"), allow_missing=True)
        self.xn = self._read_xform_1d(bx, ("xn_b", "ixn_b"), allow_missing=True)

        s_bmnc = getattr(bx, "s_b", None)
        if s_bmnc is not None:
            self.s_bmnc = np.asarray(s_bmnc, dtype=float).copy()
        else:
            self.s_bmnc = np.linspace(0.0, 1.0, self.bmnc_data.shape[0])

        s_in = getattr(bx, "s_in", None)
        if s_in is None:
            self.s_full = np.linspace(0.0, 1.0, self.iota_data.size)
        else:
            self.s_full = np.asarray(s_in, dtype=float).copy()
        self.s_half = self.s_full.copy()
        self.nfp = int(bx.nfp)
        rmnc_b = self._read_xform_2d(bx, ("rmnc_b",))
        self.R00 = float(rmnc_b[0, 0])

        self._build_splines()
        return self

    def G(self, s: np.ndarray | float) -> np.ndarray | float:
        return self._evaluate_spline(self._G_spline, s)

    def I(self, s: np.ndarray | float) -> np.ndarray | float:
        return self._evaluate_spline(self._I_spline, s)

    def iota(self, s: np.ndarray | float) -> np.ndarray | float:
        return self._evaluate_spline(self._iota_spline, s)

    def bmnc(self, s: np.ndarray | float) -> np.ndarray | float:
        return self._evaluate_coefficient_spline(self._bmnc_spline, s)

    def bmns(self, s: np.ndarray | float) -> np.ndarray | float:
        """Return sine Fourier coefficients at normalized toroidal flux ``s``."""
        return self._evaluate_coefficient_spline(self._bmns_spline, s)

    def B(self, s, theta, zeta) -> np.ndarray:
        """Evaluate the general Fourier field pointwise (DESIGN.md §7.1).

        ``s``, ``theta``, and ``zeta`` are mutually broadcast; angles are radians.
        This protocol method is distinct from the legacy outer-grid semantics of
        :meth:`compute_B`.
        """
        return self._evaluate_fourier(s, theta, zeta, derivative="B")

    def dB_ds(self, s, theta, zeta) -> np.ndarray:
        """Return analytic ``partial_s B`` using differentiated radial splines (§7.1)."""
        return self._evaluate_fourier(s, theta, zeta, derivative="s")

    def dB_dtheta(self, s, theta, zeta) -> np.ndarray:
        """Return analytic ``partial_theta B`` in B per radian (§7.1)."""
        return self._evaluate_fourier(s, theta, zeta, derivative="theta")

    def dB_dzeta(self, s, theta, zeta) -> np.ndarray:
        """Return analytic ``partial_zeta B`` in B per radian (§7.1)."""
        return self._evaluate_fourier(s, theta, zeta, derivative="zeta")

    def D_B(self, s, theta, zeta) -> np.ndarray:
        """Return ``(iota partial_theta + partial_zeta) B`` (DESIGN.md §5.1)."""
        return self._evaluate_fourier(s, theta, zeta, derivative="parallel")

    def D2_B(self, s, theta, zeta) -> np.ndarray:
        """Return the second field-line derivative in zeta parametrization (§5.1)."""
        return self._evaluate_fourier(s, theta, zeta, derivative="parallel2")

    def C(self, s):
        """Return signed ``G + iota I`` in the loaded Boozer current units (§5.1)."""
        return self.G(s) + self.iota(s) * self.I(s)

    # Quantity names accepted by fourier_quantities, mapped from the legacy
    # _evaluate_fourier derivative keywords.
    _FOURIER_QUANTITIES = ("B", "dB_ds", "dB_dtheta", "dB_dzeta", "D_B", "D2_B")
    _QUANTITY_BY_DERIVATIVE = {
        "B": "B",
        "s": "dB_ds",
        "theta": "dB_dtheta",
        "zeta": "dB_dzeta",
        "parallel": "D_B",
        "parallel2": "D2_B",
    }
    # Caps each (points, modes) temporary array at ~8 MB of doubles so that
    # arbitrarily large point batches cannot exhaust memory.
    _FOURIER_CHUNK_ELEMENTS = 2**20

    def _evaluate_fourier(self, s, theta, zeta, *, derivative: str) -> np.ndarray:
        quantity = self._QUANTITY_BY_DERIVATIVE.get(derivative)
        if quantity is None:
            raise ValueError(f"unknown Fourier derivative {derivative!r}")
        return self.fourier_quantities(s, theta, zeta, (quantity,))[0]

    def fourier_quantities(
        self, s, theta, zeta, quantities: tuple[str, ...]
    ) -> tuple[np.ndarray, ...]:
        """Evaluate several §7.1 Fourier quantities from one shared phase table.

        ``quantities`` names protocol methods among ``("B", "dB_ds",
        "dB_dtheta", "dB_dzeta", "D_B", "D2_B")``; the matching arrays are
        returned in order, in the broadcast shape of ``s``, ``theta``,
        ``zeta`` (angles in radians, ``s`` normalized toroidal flux). One
        set of trigonometric tables serves every requested quantity, the
        sine series is skipped entirely for stellarator-symmetric data, and
        points are processed in bounded chunks so large batches cannot
        exhaust memory. Each result equals the corresponding individual
        method's up to floating-point rounding of the trigonometric factors
        and summation order.
        """
        unknown = [name for name in quantities if name not in self._FOURIER_QUANTITIES]
        if unknown:
            raise ValueError(f"unknown Fourier quantities {unknown!r}")
        s_arr, theta_arr, zeta_arr = np.broadcast_arrays(
            np.asarray(s, dtype=float),
            np.asarray(theta, dtype=float),
            np.asarray(zeta, dtype=float),
        )
        shape = s_arr.shape
        s_flat = s_arr.ravel()
        theta_flat = theta_arr.ravel()
        zeta_flat = zeta_arr.ravel()
        n_points = s_flat.size
        n_modes = self.xm.size
        chunk = max(1, self._FOURIER_CHUNK_ELEMENTS // max(1, n_modes))
        if n_points <= chunk:
            computed = self._fourier_chunk(s_flat, theta_flat, zeta_flat, quantities)
            return tuple(computed[name].reshape(shape) for name in quantities)
        results = {name: np.empty(n_points) for name in set(quantities)}
        for start in range(0, n_points, chunk):
            stop = min(n_points, start + chunk)
            computed = self._fourier_chunk(
                s_flat[start:stop],
                theta_flat[start:stop],
                zeta_flat[start:stop],
                quantities,
            )
            for name, value in computed.items():
                results[name][start:stop] = value
        return tuple(results[name].reshape(shape) for name in quantities)

    def _fourier_chunk(
        self,
        s: np.ndarray,
        theta: np.ndarray,
        zeta: np.ndarray,
        quantities: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        """Evaluate one bounded chunk of points; returns arrays by quantity."""
        # "Base" quantities contract the coefficients against cos(phase)
        # (plus sin(phase) for sine modes); "first"-derivative quantities
        # swap the trigonometric tables and carry a mode-number weight.
        needed, need_base, need_first, need_value, need_ds, need_k = _fourier_plan(
            quantities
        )
        has_sine = self._has_sine
        xm = self._xm_float
        xn = self._xn_float
        need_cos = need_base or (has_sine and need_first)
        need_sin = need_first or (has_sine and need_base)
        if self._trig_factorized:
            # cos(m theta - n zeta) and its sine from the angle-difference
            # identities over the unique m and unique n values: a few
            # hundred transcendental evaluations instead of one (or two)
            # per mode per point.
            cos_m_theta = np.cos(theta[:, np.newaxis] * self._m_unique)
            sin_m_theta = np.sin(theta[:, np.newaxis] * self._m_unique)
            cos_n_zeta = np.cos(zeta[:, np.newaxis] * self._n_unique)
            sin_n_zeta = np.sin(zeta[:, np.newaxis] * self._n_unique)
            cos_m = cos_m_theta[:, self._m_index]
            sin_m = sin_m_theta[:, self._m_index]
            cos_n = cos_n_zeta[:, self._n_index]
            sin_n = sin_n_zeta[:, self._n_index]
            cos_phase = cos_m * cos_n + sin_m * sin_n if need_cos else None
            sin_phase = sin_m * cos_n - cos_m * sin_n if need_sin else None
        else:
            phase = theta[:, np.newaxis] * xm - zeta[:, np.newaxis] * xn
            cos_phase = np.cos(phase) if need_cos else None
            sin_phase = np.sin(phase) if need_sin else None
        cosine = (
            self._evaluate_coefficient_spline(self._bmnc_spline, s)
            if need_value
            else None
        )
        sine = (
            self._evaluate_coefficient_spline(self._bmns_spline, s)
            if need_value and has_sine
            else None
        )
        if need_ds:
            cosine_ds = self._evaluate_coefficient_spline(self._bmnc_spline, s, 1)
            sine_ds = (
                self._evaluate_coefficient_spline(self._bmns_spline, s, 1)
                if has_sine
                else None
            )
        if need_k:
            k = np.atleast_1d(np.asarray(self.iota(s)))[:, np.newaxis] * xm - xn
        table_shape = (len(s), xm.size)

        def contracted(coefficients, table, weight=None):
            if weight is None:
                return np.einsum("ij,ij->i", coefficients, table)
            weight = np.broadcast_to(weight, table_shape)
            return np.einsum("ij,ij,ij->i", coefficients, table, weight)

        computed: dict[str, np.ndarray] = {}
        for name in needed:
            if name == "B":
                value = contracted(cosine, cos_phase)
                if has_sine:
                    value += contracted(sine, sin_phase)
            elif name == "dB_ds":
                value = contracted(cosine_ds, cos_phase)
                if has_sine:
                    value += contracted(sine_ds, sin_phase)
            elif name == "dB_dtheta":
                value = -contracted(cosine, sin_phase, xm)
                if has_sine:
                    value += contracted(sine, cos_phase, xm)
            elif name == "dB_dzeta":
                value = contracted(cosine, sin_phase, xn)
                if has_sine:
                    value -= contracted(sine, cos_phase, xn)
            elif name == "D_B":
                value = -contracted(cosine, sin_phase, k)
                if has_sine:
                    value += contracted(sine, cos_phase, k)
            else:  # D2_B
                value = -contracted(cosine, cos_phase, k * k)
                if has_sine:
                    value -= contracted(sine, sin_phase, k * k)
            computed[name] = value
        return computed

    def compute_B(
        self,
        s: np.ndarray | float,
        theta: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=float)
        phi_arr = np.asarray(phi, dtype=float)
        if theta_arr.shape != phi_arr.shape:
            raise ValueError("theta and phi must have the same shape")
        if theta_arr.ndim not in (1, 2):
            raise ValueError("theta and phi must be 1d or 2d arrays")

        theta_flat = theta_arr.reshape(-1)
        phi_flat = phi_arr.reshape(-1)

        xm = self.xm
        xn = self.xn
        bmnc_eval = np.asarray(self.bmnc(s), dtype=float)
        if bmnc_eval.ndim == 1:
            bmnc_eval = bmnc_eval[np.newaxis, :]
            scalar_s = True
        else:
            scalar_s = False
        if bmnc_eval.shape[1] != xm.size:
            raise ValueError(
                f"bmnc mode count ({bmnc_eval.shape[1]}) does not match xm/xn length ({xm.size})"
            )

        phase = (
            xm[:, np.newaxis] * theta_flat[np.newaxis, :]
            - xn[:, np.newaxis] * phi_flat[np.newaxis, :]
        )
        bmns_eval = np.asarray(self.bmns(s), dtype=float)
        if bmns_eval.ndim == 1:
            bmns_eval = bmns_eval[np.newaxis, :]
        B_flat = bmnc_eval @ np.cos(phase) + bmns_eval @ np.sin(phase)

        if theta_arr.ndim == 1:
            if scalar_s:
                return B_flat[0]
            return B_flat

        B = B_flat.reshape((B_flat.shape[0],) + theta_arr.shape)
        if scalar_s:
            return B[0]
        return B

    def surface(self, s: float) -> "BoozerSurface":
        return BoozerSurface(self, s)

    def get_min_max(
        self,
        n_s: int = 10,
        n_theta: int = 64,
        n_phi: int = 65,
    ):
        """Get the global minimum and maximum |B| throughout the volume."""
        s = np.linspace(0.0, 1.0, n_s)
        # It is best if n_theta is even since B_max is often at theta=pi.
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        phi = np.linspace(0.0, 2.0 * np.pi / self.nfp, n_phi, endpoint=False)
        phi2d, theta2d = np.meshgrid(phi, theta)
        B = self.compute_B(s, theta2d, phi2d)
        return np.min(B), np.max(B)

    def _build_splines(self) -> None:
        if (
            self.s_full is None
            or self.s_half is None
            or self.G_data is None
            or self.I_data is None
            or self.iota_data is None
            or self.bmnc_data is None
            or self.bmns_data is None
        ):
            raise ValueError("Data has not been loaded")

        self._G_spline = CubicSpline(self.s_half, self.G_data, axis=0, extrapolate=True)
        self._I_spline = CubicSpline(self.s_half, self.I_data, axis=0, extrapolate=True)
        self._iota_spline = CubicSpline(
            self.s_half, self.iota_data, axis=0, extrapolate=True
        )
        fourier_s = self.s_half if self.s_bmnc is None else self.s_bmnc
        self._bmnc_spline = CubicSpline(
            fourier_s, self.bmnc_data, axis=0, extrapolate=True
        )
        self._bmns_spline = CubicSpline(
            fourier_s, self.bmns_data, axis=0, extrapolate=True
        )
        # Identically-zero sine data lets fourier_quantities skip the sine
        # series without changing any value.
        self._has_sine = bool(np.any(self.bmns_data))
        self._xm_float = np.asarray(self.xm, dtype=float)
        self._xn_float = np.asarray(self.xn, dtype=float)
        self._m_unique, self._m_index = np.unique(self._xm_float, return_inverse=True)
        self._n_unique, self._n_index = np.unique(self._xn_float, return_inverse=True)
        # The angle-difference factorization only pays off when the mode
        # table reuses few distinct m and n values.
        self._trig_factorized = (
            self._m_unique.size + self._n_unique.size <= self._xm_float.size // 4
        )
        # Innermost coefficient surface: below it the axis-regular harmonic
        # continuation of DESIGN.md §7.3 replaces spline extrapolation.
        first = float(fourier_s[0])
        self._coefficient_s0 = first if first > 0.0 else None

    def _evaluate_coefficient_spline(
        self,
        spline: CubicSpline | None,
        s: np.ndarray | float,
        derivative_order: int = 0,
    ) -> np.ndarray:
        """Evaluate Fourier-coefficient splines with an axis-regular core.

        Below the innermost coefficient surface ``s0`` the cubic spline would
        extrapolate without physical constraint, leaving ``m != 0`` harmonics
        finite at the axis and making ``|B|`` multivalued there. DESIGN.md
        §7.3 (option 1) requires the poloidal-harmonic scaling ``rho^|m|``
        with ``rho = sqrt(s)``, so each ``m != 0`` coefficient is continued
        for ``s < s0`` as ``c(s0) (s/s0)^{|m|/2}``: smooth in the logical
        ``(x, y)`` plane and continuous at ``s0``. ``m = 0`` harmonics keep
        their spline values. The ``s`` derivative of an ``|m| = 1`` harmonic
        is genuinely singular at ``s = 0`` and evaluates to ``inf`` there
        rather than a finite substitute.
        """
        if spline is None:
            raise ValueError("The requested field has not been loaded")
        s_arr = np.asarray(s, dtype=float)
        flat = np.atleast_1d(s_arr).ravel()
        value = np.asarray(spline(flat, nu=derivative_order), dtype=float)
        s0 = self._coefficient_s0
        if s0 is not None and self.xm is not None:
            mask = flat < s0
            if np.any(mask):
                if derivative_order not in (0, 1):
                    raise ValueError(
                        "axis-regular coefficient continuation supports "
                        "derivative orders 0 and 1 only"
                    )
                m = np.abs(np.asarray(self.xm, dtype=float))
                nonzero = m > 0.0
                exponent = 0.5 * m[nonzero]
                anchor = np.asarray(spline(s0), dtype=float)[nonzero]
                ratio = (flat[mask] / s0)[:, np.newaxis]
                with np.errstate(divide="ignore"):
                    if derivative_order == 0:
                        continued = anchor * ratio**exponent
                    else:
                        continued = anchor * exponent * ratio ** (exponent - 1.0) / s0
                block = value[mask]
                block[:, nonzero] = continued
                value[mask] = block
        return value.reshape(s_arr.shape + value.shape[-1:])

    @staticmethod
    def _evaluate_spline(
        spline: CubicSpline | None, s: np.ndarray | float, derivative_order: int = 0
    ) -> np.ndarray | float:
        if spline is None:
            raise ValueError("The requested field has not been loaded")

        value = np.asarray(spline(s, nu=derivative_order))
        if value.ndim == 0:
            return value.item()
        return value

    @staticmethod
    def _read_xform_1d(
        xform: Any, names: tuple[str, ...], allow_missing: bool = False
    ) -> np.ndarray:
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


class BoozerSurface:
    """A Boozer-coordinate flux surface with interpolated field data."""

    def __init__(self, booz: BoozerField, s: float) -> None:
        self.booz = booz
        self.s = s
        self.G = booz.G(s)
        self.I = booz.I(s)
        self.iota = booz.iota(s)
        self.bmnc = booz.bmnc(s)
        self.bmns = booz.bmns(s)
        self.nfp = booz.nfp
        self.R00 = booz.R00

    def compute_B(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=float)
        phi_arr = np.asarray(phi, dtype=float)
        if theta_arr.shape != phi_arr.shape:
            raise ValueError("theta and phi must have the same shape")
        if theta_arr.ndim not in (1, 2):
            raise ValueError("theta and phi must be 1d or 2d arrays")

        theta_flat = theta_arr.reshape(-1)
        phi_flat = phi_arr.reshape(-1)

        xm = self.booz.xm
        xn = self.booz.xn
        bmnc_eval = np.asarray(self.bmnc, dtype=float)
        if bmnc_eval.ndim != 1:
            raise ValueError("BoozerSurface bmnc data must be one-dimensional")
        if bmnc_eval.size != xm.size:
            raise ValueError(
                f"bmnc mode count ({bmnc_eval.size}) does not match xm/xn length ({xm.size})"
            )

        phase = (
            xm[:, np.newaxis] * theta_flat[np.newaxis, :]
            - xn[:, np.newaxis] * phi_flat[np.newaxis, :]
        )
        bmns_eval = np.asarray(self.bmns, dtype=float)
        B_flat = bmnc_eval @ np.cos(phase) + bmns_eval @ np.sin(phase)

        if theta_arr.ndim == 1:
            return B_flat
        return B_flat.reshape(theta_arr.shape)

    def compute_B_tensor_alpha_phi(
        self,
        alpha: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        """Compute B(alpha, phi) optimized for a tensor product of 1d alpha and
        1d phi arrays."""
        phi_arr = np.asarray(phi, dtype=float)
        alpha_arr = np.asarray(alpha, dtype=float)
        xm = self.booz.xm
        xn = self.booz.xn
        bmnc_eval = np.asarray(self.bmnc, dtype=float)
        bmns_eval = np.asarray(self.bmns, dtype=float)
        if bmnc_eval.ndim != 1:
            raise ValueError("BoozerSurface bmnc data must be one-dimensional")
        if bmnc_eval.size != xm.size:
            raise ValueError(
                f"bmnc mode count ({bmnc_eval.size}) does not match xm/xn length ({xm.size})"
            )
        k = xm * self.iota - xn
        k_phi = np.outer(k, phi_arr)
        cos_k_phi = np.cos(k_phi)
        sin_k_phi = np.sin(k_phi)

        xm_alpha = np.outer(alpha_arr, xm)
        cos_alpha = np.cos(xm_alpha)
        sin_alpha = np.sin(xm_alpha)
        cos_part = (
            cos_alpha * bmnc_eval[np.newaxis, :] + sin_alpha * bmns_eval[np.newaxis, :]
        )
        sin_part = (
            cos_alpha * bmns_eval[np.newaxis, :] - sin_alpha * bmnc_eval[np.newaxis, :]
        )
        return cos_part @ cos_k_phi + sin_part @ sin_k_phi
