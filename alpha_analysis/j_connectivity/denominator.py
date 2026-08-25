"""Independent denominator and global-field bounds (DESIGN.md §13)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .field import BoozerFieldLike

FloatArray = NDArray[np.float64]


class SourceProfile(Protocol):
    """Callable source profile ``h(rho)`` for dimensionless ``0 <= rho <= 1``."""

    def __call__(self, rho: ArrayLike) -> FloatArray: ...


@dataclass(frozen=True)
class UniformSourceProfile:
    """Nonnegative constant source profile, with arbitrary normalization."""

    value: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.value) or self.value < 0.0:
            raise ValueError("source-profile value must be finite and nonnegative")

    def __call__(self, rho: ArrayLike) -> FloatArray:
        return np.full(np.asarray(rho).shape, self.value, dtype=float)


@dataclass(frozen=True)
class DenominatorConfig:
    """Tensor-product resolution for DESIGN.md §13.1."""

    n_s: int = 16
    n_theta: int = 32
    n_zeta: int = 32

    def __post_init__(self) -> None:
        if self.n_s < 1:
            raise ValueError("n_s must be positive")
        if self.n_theta < 2 or self.n_zeta < 2:
            raise ValueError("periodic quadrature resolutions must be at least two")


@dataclass(frozen=True)
class DenominatorEstimate:
    """One tensor-product estimate of the normalized volume factor ``V_h``."""

    V_h: float
    nodes_s: FloatArray
    weights_s: FloatArray
    n_theta: int
    n_zeta: int


@dataclass(frozen=True)
class DenominatorConvergence:
    """Ordered denominator estimates and successive absolute changes."""

    estimates: tuple[DenominatorEstimate, ...]
    absolute_changes: FloatArray


@dataclass(frozen=True)
class BoundsConfig:
    """Grid, refinement, and safety-margin controls for DESIGN.md §13.2."""

    n_s: int = 17
    n_theta: int = 32
    n_zeta: int = 32
    candidate_count: int = 8
    safety_factor: float = 2.0
    absolute_margin: float = 0.0
    user_b_min: float | None = None
    user_b_max: float | None = None

    def __post_init__(self) -> None:
        if self.n_s < 2:
            raise ValueError("n_s must be at least two")
        if self.n_theta < 2 or self.n_zeta < 2:
            raise ValueError("angular grid resolutions must be at least two")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if self.safety_factor < 0.0 or self.absolute_margin < 0.0:
            raise ValueError("safety margins must be nonnegative")
        for name in ("user_b_min", "user_b_max"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if (
            self.user_b_min is not None
            and self.user_b_max is not None
            and self.user_b_min >= self.user_b_max
        ):
            raise ValueError("user_b_min must be less than user_b_max")


@dataclass(frozen=True)
class GlobalBBounds:
    """Refined extrema, safe pitch bracket, and radial extrema profiles."""

    refined_min: float
    refined_max: float
    lower: float
    upper: float
    interpolation_error: float
    safety_margin: float
    minimum_location: FloatArray
    maximum_location: FloatArray
    profile_s: FloatArray
    profile_min: FloatArray
    profile_max: FloatArray


def compute_denominator(
    field: BoozerFieldLike,
    source_profile: SourceProfile,
    config: DenominatorConfig = DenominatorConfig(),
) -> DenominatorEstimate:
    """Compute ``V_h`` by the tensor rule in DESIGN.md §13.1.

    ``s`` is normalized toroidal flux, angles are radians, and the source is
    evaluated at ``rho=sqrt(s)``.  The result has the units of ``|C|/B**2``.
    Gauss-Legendre quadrature is used radially and endpoint-free periodic
    trapezoidal rules are used in both angles.
    """
    legendre_nodes, legendre_weights = np.polynomial.legendre.leggauss(config.n_s)
    nodes_s = 0.5 * (legendre_nodes + 1.0)
    weights_s = 0.5 * legendre_weights
    theta = np.linspace(0.0, 2.0 * np.pi, config.n_theta, endpoint=False)
    zeta_period = 2.0 * np.pi / field.nfp
    zeta = np.linspace(0.0, zeta_period, config.n_zeta, endpoint=False)

    source = _source_values(source_profile, np.sqrt(nodes_s))
    C = np.asarray(field.C(nodes_s), dtype=float)
    try:
        C = np.broadcast_to(C, nodes_s.shape)
    except ValueError as error:
        raise ValueError("C(s) must return values broadcastable to s") from error
    if not np.all(np.isfinite(C)):
        raise ValueError("C(s) returned non-finite values")

    s_grid = nodes_s[:, np.newaxis, np.newaxis]
    theta_grid = theta[np.newaxis, :, np.newaxis]
    zeta_grid = zeta[np.newaxis, np.newaxis, :]
    B = np.asarray(field.B(s_grid, theta_grid, zeta_grid), dtype=float)
    expected_shape = (config.n_s, config.n_theta, config.n_zeta)
    try:
        B = np.broadcast_to(B, expected_shape)
    except ValueError as error:
        raise ValueError("B must broadcast over the tensor-product grid") from error
    if not np.all(np.isfinite(B)) or np.any(B <= 0.0):
        raise ValueError("B must be finite and strictly positive in the denominator")

    angular_integral = (2.0 * np.pi) * zeta_period * np.mean(B**-2, axis=(1, 2))
    value = np.sum(weights_s * source * np.abs(C) * angular_integral)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("V_h must be finite and positive; check the source and C(s)")
    return DenominatorEstimate(
        V_h=float(value),
        nodes_s=nodes_s,
        weights_s=weights_s,
        n_theta=config.n_theta,
        n_zeta=config.n_zeta,
    )


def denominator_convergence(
    field: BoozerFieldLike,
    source_profile: SourceProfile,
    configs: tuple[DenominatorConfig, ...],
) -> DenominatorConvergence:
    """Evaluate an explicitly supplied, independently controlled resolution study."""
    if not configs:
        raise ValueError("at least one denominator configuration is required")
    estimates = tuple(
        compute_denominator(field, source_profile, config) for config in configs
    )
    values = np.array([estimate.V_h for estimate in estimates])
    return DenominatorConvergence(
        estimates=estimates,
        absolute_changes=np.abs(np.diff(values)),
    )


def find_global_B_bounds(
    field: BoozerFieldLike,
    config: BoundsConfig = BoundsConfig(),
) -> GlobalBBounds:
    """Find refined global extrema and a safe bracket (DESIGN.md §13.2).

    Grid extrema seed bounded local optimization in ``(s, theta, zeta)``.
    Angular variables are periodic in radians.  The safety margin is
    ``absolute_margin + safety_factor * interpolation_error``, where the error
    is the largest change from a grid extremum to its locally refined value.
    Explicit user bounds replace the corresponding computed bracket endpoint,
    but are rejected if they do not contain the refined extrema.
    """
    s = np.linspace(0.0, 1.0, config.n_s)
    theta = np.linspace(0.0, 2.0 * np.pi, config.n_theta, endpoint=False)
    zeta_period = 2.0 * np.pi / field.nfp
    zeta = np.linspace(0.0, zeta_period, config.n_zeta, endpoint=False)
    s_grid, theta_grid, zeta_grid = np.meshgrid(s, theta, zeta, indexing="ij")
    B_grid = _finite_B(field, s_grid, theta_grid, zeta_grid)

    minimum_value, minimum_location = _refine_candidates(
        field,
        B_grid,
        s_grid,
        theta_grid,
        zeta_grid,
        config.candidate_count,
        maximize=False,
    )
    maximum_value, maximum_location = _refine_candidates(
        field,
        B_grid,
        s_grid,
        theta_grid,
        zeta_grid,
        config.candidate_count,
        maximize=True,
    )

    profile_min = np.empty_like(s)
    profile_max = np.empty_like(s)
    profile_grid_min = np.min(B_grid, axis=(1, 2))
    profile_grid_max = np.max(B_grid, axis=(1, 2))
    for radial_index, radial_value in enumerate(s):
        profile_min[radial_index] = _refine_angular_candidates(
            field,
            radial_value,
            B_grid[radial_index],
            theta,
            zeta,
            config.candidate_count,
            maximize=False,
        )
        profile_max[radial_index] = _refine_angular_candidates(
            field,
            radial_value,
            B_grid[radial_index],
            theta,
            zeta,
            config.candidate_count,
            maximize=True,
        )

    interpolation_error = float(
        max(
            abs(minimum_value - float(np.min(B_grid))),
            abs(maximum_value - float(np.max(B_grid))),
            np.max(np.abs(profile_min - profile_grid_min)),
            np.max(np.abs(profile_max - profile_grid_max)),
        )
    )
    safety_margin = config.absolute_margin + config.safety_factor * interpolation_error
    lower = minimum_value - safety_margin
    upper = maximum_value + safety_margin
    if config.user_b_min is not None:
        if config.user_b_min > minimum_value:
            raise ValueError(
                "user-supplied B bounds do not bracket the refined extrema"
            )
        lower = config.user_b_min
    if config.user_b_max is not None:
        if config.user_b_max < maximum_value:
            raise ValueError(
                "user-supplied B bounds do not bracket the refined extrema"
            )
        upper = config.user_b_max
    if lower >= upper:
        raise ValueError("global B bounds must have positive width")

    return GlobalBBounds(
        refined_min=minimum_value,
        refined_max=maximum_value,
        lower=float(lower),
        upper=float(upper),
        interpolation_error=interpolation_error,
        safety_margin=float(safety_margin),
        minimum_location=minimum_location,
        maximum_location=maximum_location,
        profile_s=s,
        profile_min=profile_min,
        profile_max=profile_max,
    )


def _source_values(source_profile: SourceProfile, rho: FloatArray) -> FloatArray:
    values = np.asarray(source_profile(rho), dtype=float)
    try:
        values = np.broadcast_to(values, rho.shape)
    except ValueError as error:
        raise ValueError(
            "source profile must return values broadcastable to rho"
        ) from error
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("source profile must be finite and nonnegative")
    return values


def _finite_B(field, s, theta, zeta) -> FloatArray:
    values = np.asarray(field.B(s, theta, zeta), dtype=float)
    shape = np.broadcast(s, theta, zeta).shape
    try:
        values = np.broadcast_to(values, shape)
    except ValueError as error:
        raise ValueError("B must broadcast over the extrema-search grid") from error
    if not np.all(np.isfinite(values)):
        raise ValueError("B returned non-finite values during the extrema search")
    if np.any(values <= 0.0):
        raise ValueError("B must be strictly positive during the extrema search")
    return values


def _refine_candidates(
    field,
    values,
    s_grid,
    theta_grid,
    zeta_grid,
    candidate_count,
    *,
    maximize,
):
    order = np.argsort(values, axis=None)
    if maximize:
        order = order[::-1]
    count = min(candidate_count, order.size)
    period = 2.0 * np.pi / field.nfp
    sign = -1.0 if maximize else 1.0
    best_value = -np.inf if maximize else np.inf
    best_location = None

    def objective(point):
        value = field.B(point[0], point[1] % (2.0 * np.pi), point[2] % period)
        return sign * float(np.asarray(value))

    for flat_index in order[:count]:
        index = np.unravel_index(flat_index, values.shape)
        initial = np.array(
            [s_grid[index], theta_grid[index], zeta_grid[index]], dtype=float
        )
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=((0.0, 1.0), (0.0, 2.0 * np.pi), (0.0, period)),
            options={"ftol": 1.0e-14, "gtol": 1.0e-10, "maxiter": 300},
        )
        location = np.array(
            [result.x[0], result.x[1] % (2.0 * np.pi), result.x[2] % period]
        )
        refined = float(field.B(*location))
        is_better = refined > best_value if maximize else refined < best_value
        if is_better:
            best_value = refined
            best_location = location
    return float(best_value), best_location


def _refine_angular_candidates(
    field, s, values, theta, zeta, candidate_count, *, maximize
):
    order = np.argsort(values, axis=None)
    if maximize:
        order = order[::-1]
    count = min(candidate_count, order.size)
    period = 2.0 * np.pi / field.nfp
    sign = -1.0 if maximize else 1.0
    best_value = -np.inf if maximize else np.inf

    def objective(point):
        value = field.B(s, point[0] % (2.0 * np.pi), point[1] % period)
        return sign * float(np.asarray(value))

    for flat_index in order[:count]:
        index = np.unravel_index(flat_index, values.shape)
        result = minimize(
            objective,
            np.array([theta[index[0]], zeta[index[1]]]),
            method="L-BFGS-B",
            bounds=((0.0, 2.0 * np.pi), (0.0, period)),
            options={"ftol": 1.0e-14, "gtol": 1.0e-10, "maxiter": 200},
        )
        refined = float(field.B(s, result.x[0] % (2.0 * np.pi), result.x[1] % period))
        is_better = refined > best_value if maximize else refined < best_value
        if is_better:
            best_value = refined
    return float(best_value)
