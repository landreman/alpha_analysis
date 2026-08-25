"""Magnetic-field protocol for the J-connectivity pipeline (DESIGN.md §7.2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@runtime_checkable
class BoozerFieldLike(Protocol):
    """Vectorized Boozer-field interface used by all numerical stages.

    Angles are radians, ``s`` is normalized toroidal flux, and
    ``D_B = (iota * partial_theta + partial_zeta) B`` follows DESIGN.md §5.1.
    Implementations accept mutually broadcastable scalar or array coordinates.
    """

    nfp: int

    def B(self, s: ArrayLike, theta: ArrayLike, zeta: ArrayLike) -> FloatArray: ...

    def dB_ds(self, s: ArrayLike, theta: ArrayLike, zeta: ArrayLike) -> FloatArray: ...

    def dB_dtheta(
        self, s: ArrayLike, theta: ArrayLike, zeta: ArrayLike
    ) -> FloatArray: ...

    def dB_dzeta(
        self, s: ArrayLike, theta: ArrayLike, zeta: ArrayLike
    ) -> FloatArray: ...

    def D_B(self, s: ArrayLike, theta: ArrayLike, zeta: ArrayLike) -> FloatArray: ...

    def D2_B(self, s: ArrayLike, theta: ArrayLike, zeta: ArrayLike) -> FloatArray: ...

    def iota(self, s: ArrayLike) -> FloatArray: ...

    def G(self, s: ArrayLike) -> FloatArray: ...

    def I(self, s: ArrayLike) -> FloatArray: ...

    def C(self, s: ArrayLike) -> FloatArray:
        """Return signed ``G + iota I`` in Boozer current units."""
        ...
