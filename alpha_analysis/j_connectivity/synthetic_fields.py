"""Analytic synthetic Fourier fields for scientific tests (DESIGN.md §20.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticFourierField:
    """Finite Boozer Fourier series with polynomial radial coefficients.

    ``cosine_coefficients[k, p]`` and ``sine_coefficients[k, p]`` multiply
    ``s**p`` for mode ``m[k] theta - n[k] zeta``.  Magnetic-field coefficients
    and derivatives have the units of ``B``; angles are radians and ``s`` is
    normalized toroidal flux.  The analytic derivatives implement DESIGN.md
    §§5.1 and 7.1.
    """

    nfp: int
    m: NDArray[np.int64]
    n: NDArray[np.int64]
    cosine_coefficients: FloatArray
    sine_coefficients: FloatArray
    iota_coefficients: FloatArray
    G_coefficients: FloatArray
    I_coefficients: FloatArray

    def __post_init__(self) -> None:
        m = np.asarray(self.m, dtype=np.int64)
        n = np.asarray(self.n, dtype=np.int64)
        cosine = np.asarray(self.cosine_coefficients, dtype=float)
        sine = np.asarray(self.sine_coefficients, dtype=float)
        if self.nfp < 1:
            raise ValueError("nfp must be positive")
        if m.ndim != 1 or n.shape != m.shape:
            raise ValueError("m and n must be one-dimensional arrays of equal length")
        if cosine.ndim != 2 or sine.shape != cosine.shape:
            raise ValueError("cosine and sine coefficients must have the same 2d shape")
        if cosine.shape[0] != m.size:
            raise ValueError("each Fourier mode must have one radial polynomial")
        if np.any(n % self.nfp != 0):
            raise ValueError("toroidal mode numbers n must be multiples of nfp")
        for name in ("iota_coefficients", "G_coefficients", "I_coefficients"):
            if np.asarray(getattr(self, name)).ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
        object.__setattr__(self, "m", m)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "cosine_coefficients", cosine)
        object.__setattr__(self, "sine_coefficients", sine)

    @staticmethod
    def _polynomial(coefficients: ArrayLike, s: ArrayLike) -> FloatArray:
        return np.polynomial.polynomial.polyval(
            np.asarray(s, dtype=float), coefficients
        )

    @staticmethod
    def _mode_polynomials(coefficients: FloatArray, s: FloatArray) -> FloatArray:
        result = np.zeros(s.shape + (coefficients.shape[0],), dtype=float)
        for power in range(coefficients.shape[1] - 1, -1, -1):
            result = result * s[..., np.newaxis] + coefficients[:, power]
        return result

    @staticmethod
    def _radial_derivative(coefficients: FloatArray) -> FloatArray:
        if coefficients.shape[1] == 1:
            return np.zeros((coefficients.shape[0], 1))
        powers = np.arange(1, coefficients.shape[1], dtype=float)
        return coefficients[:, 1:] * powers

    def _coordinates(
        self, s: ArrayLike, theta: ArrayLike, zeta: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        s_arr, theta_arr, zeta_arr = np.broadcast_arrays(
            np.asarray(s, dtype=float),
            np.asarray(theta, dtype=float),
            np.asarray(zeta, dtype=float),
        )
        phase = theta_arr[..., np.newaxis] * self.m - zeta_arr[..., np.newaxis] * self.n
        return s_arr, phase

    def _amplitudes(self, s: FloatArray) -> tuple[FloatArray, FloatArray]:
        return (
            self._mode_polynomials(self.cosine_coefficients, s),
            self._mode_polynomials(self.sine_coefficients, s),
        )

    def B(self, s: ArrayLike, theta: ArrayLike, zeta: ArrayLike) -> FloatArray:
        """Evaluate ``sum(Bc cos(chi) + Bs sin(chi))`` (DESIGN.md §7.1)."""
        s_arr, phase = self._coordinates(s, theta, zeta)
        cosine, sine = self._amplitudes(s_arr)
        return np.sum(cosine * np.cos(phase) + sine * np.sin(phase), axis=-1)

    def dB_ds(self, s: ArrayLike, theta: ArrayLike, zeta: ArrayLike) -> FloatArray:
        """Return the analytic derivative with respect to normalized flux ``s``."""
        s_arr, phase = self._coordinates(s, theta, zeta)
        cosine = self._mode_polynomials(
            self._radial_derivative(self.cosine_coefficients), s_arr
        )
        sine = self._mode_polynomials(
            self._radial_derivative(self.sine_coefficients), s_arr
        )
        return np.sum(cosine * np.cos(phase) + sine * np.sin(phase), axis=-1)

    def dB_dtheta(self, s: ArrayLike, theta: ArrayLike, zeta: ArrayLike) -> FloatArray:
        """Return analytic ``partial_theta B`` in B per radian."""
        s_arr, phase = self._coordinates(s, theta, zeta)
        cosine, sine = self._amplitudes(s_arr)
        return np.sum(
            self.m * (-cosine * np.sin(phase) + sine * np.cos(phase)), axis=-1
        )

    def dB_dzeta(self, s: ArrayLike, theta: ArrayLike, zeta: ArrayLike) -> FloatArray:
        """Return analytic ``partial_zeta B`` in B per radian."""
        s_arr, phase = self._coordinates(s, theta, zeta)
        cosine, sine = self._amplitudes(s_arr)
        return np.sum(self.n * (cosine * np.sin(phase) - sine * np.cos(phase)), axis=-1)

    def D_B(self, s: ArrayLike, theta: ArrayLike, zeta: ArrayLike) -> FloatArray:
        """Return ``(iota partial_theta + partial_zeta) B`` (DESIGN.md §5.1)."""
        s_arr, phase = self._coordinates(s, theta, zeta)
        cosine, sine = self._amplitudes(s_arr)
        k = self.m * self.iota(s_arr)[..., np.newaxis] - self.n
        return np.sum(k * (-cosine * np.sin(phase) + sine * np.cos(phase)), axis=-1)

    def D2_B(self, s: ArrayLike, theta: ArrayLike, zeta: ArrayLike) -> FloatArray:
        """Return ``(iota partial_theta + partial_zeta)^2 B`` (§§5.1, 7.1)."""
        s_arr, phase = self._coordinates(s, theta, zeta)
        cosine, sine = self._amplitudes(s_arr)
        k = self.m * self.iota(s_arr)[..., np.newaxis] - self.n
        return -np.sum(k**2 * (cosine * np.cos(phase) + sine * np.sin(phase)), axis=-1)

    def iota(self, s: ArrayLike) -> FloatArray:
        """Return rotational transform as a polynomial in normalized flux."""
        return self._polynomial(self.iota_coefficients, s)

    def G(self, s: ArrayLike) -> FloatArray:
        """Return the Boozer toroidal-current profile in its supplied units."""
        return self._polynomial(self.G_coefficients, s)

    def I(self, s: ArrayLike) -> FloatArray:
        """Return the Boozer poloidal-current profile in its supplied units."""
        return self._polynomial(self.I_coefficients, s)

    def C(self, s: ArrayLike) -> FloatArray:
        """Return signed ``G + iota I``; no field-direction sign is discarded."""
        return self.G(s) + self.iota(s) * self.I(s)
