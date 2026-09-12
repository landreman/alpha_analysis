"""Independent trapped-population accounting (DESIGN.md §§12.1--12.4).

This module deliberately does not classify accessibility.  It evaluates the total
trapped population for a declared trapping model, preserves quadrature/assumption
scope, and supplies interval-safe ledger algebra for later reachability stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .denominator import SourceProfile
from .field import BoozerFieldLike

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
LinewiseTrapping = Callable[
    [FloatArray, FloatArray, FloatArray, float, FloatArray], ArrayLike
]


@dataclass(frozen=True)
class PopulationQuadratureConfig:
    """Tensor-product resolution for the population integrals in §12.1.

    ``s`` is dimensionless normalized toroidal flux and both angles are radians.
    Gauss--Legendre nodes are used in ``s``; endpoint-free periodic trapezoidal
    nodes are used in ``theta`` and one field period of ``zeta``.
    """

    n_s: int = 16
    n_theta: int = 64
    n_zeta: int = 64

    def __post_init__(self) -> None:
        if self.n_s < 1:
            raise ValueError("n_s must be positive")
        if self.n_theta < 2 or self.n_zeta < 2:
            raise ValueError("periodic quadrature resolutions must be at least two")


@dataclass(frozen=True)
class PopulationContext:
    """Reusable source-weighted spatial quadrature for one Boozer field.

    ``surface_maximum`` has field units and is sampled at ``nodes_s``.  Its
    provenance is kept separately because an optimizer or sampled maximum is not
    automatically a certified upper bound (DESIGN.md §§12.1 and 13.4).
    """

    config: PopulationQuadratureConfig
    nodes_s: FloatArray
    weights_s: FloatArray
    theta: FloatArray
    zeta: FloatArray
    B: FloatArray
    source_values: FloatArray
    absolute_C: FloatArray
    surface_maximum: FloatArray
    source_name: str
    surface_maximum_scope: str
    surface_maximum_is_certified_upper: bool
    angular_measure_per_node: float


@dataclass(frozen=True)
class PopulationSliceEstimate:
    """Quadrature estimate of ``Q_total(b)`` from DESIGN.md §12.1.

    ``total_weight`` is unnormalized and has the units of
    ``h |C| / B`` times the three-coordinate measure.  It is not an
    accessibility result.  ``radial_density`` excludes Gauss weights so it can be
    plotted as a function of normalized flux ``s``.
    """

    b: float
    total_weight: float
    nodes_s: FloatArray
    radial_density: FloatArray
    source_name: str
    trapping_scope: str
    surface_maximum_scope: str
    bound_scope: str
    uncontrolled_errors: tuple[str, ...]


@dataclass(frozen=True)
class PitchBandEstimate:
    """Estimate of a whole pitch band's trapped weight and normalized fraction.

    ``pitch_weight`` is ``integral Q_total(b) b**-2 db``.  Its division by
    ``2 V_h`` yields ``fraction``.  The analytic pitch antiderivative retains the
    required ``b**-2`` Jacobian from DESIGN.md §§4.3 and 12.4.
    """

    b_lower: float
    b_upper: float
    pitch_weight: float
    denominator_estimate: float
    fraction: float
    nodes_s: FloatArray
    radial_fraction_density: FloatArray
    source_name: str
    trapping_scope: str
    surface_maximum_scope: str
    bound_scope: str
    uncontrolled_errors: tuple[str, ...]


@dataclass(frozen=True)
class WeightBounds:
    """Finite nonnegative lower/upper bounds for one population weight."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError("weight bounds must be finite")
        if self.lower < 0.0 or self.upper < self.lower:
            raise ValueError("weight bounds require 0 <= lower <= upper")


@dataclass(frozen=True)
class OwnedWeightBounds:
    """Disjoint cell/branch IDs with componentwise population bounds.

    Integer owner IDs are the mechanism that prevents a multiply covered state
    from being credited to two ledger categories.  The weights use the same units
    as ``Q_total``.
    """

    owner_ids: IntArray
    lower: FloatArray
    upper: FloatArray

    def __post_init__(self) -> None:
        owner_ids = np.asarray(self.owner_ids)
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if owner_ids.ndim != 1 or lower.ndim != 1 or upper.ndim != 1:
            raise ValueError("owned weights must be one-dimensional")
        if lower.shape != owner_ids.shape or upper.shape != owner_ids.shape:
            raise ValueError("owner IDs and weight bounds must have equal length")
        if not np.issubdtype(owner_ids.dtype, np.integer):
            raise ValueError("owner IDs must be integers")
        owner_ids = owner_ids.astype(np.int64, copy=False)
        if np.unique(owner_ids).size != owner_ids.size:
            raise ValueError("owner IDs overlap within one population category")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("owned weight bounds must be finite")
        if np.any(lower < 0.0) or np.any(upper < lower):
            raise ValueError("owned weights require 0 <= lower <= upper")
        object.__setattr__(self, "owner_ids", owner_ids)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def lower_total(self) -> float:
        return float(np.sum(self.lower))

    @property
    def upper_total(self) -> float:
        return float(np.sum(self.upper))


@dataclass(frozen=True)
class PopulationLedger:
    """Closed lower/upper population accounting from DESIGN.md §12.2."""

    total: WeightBounds
    reachable: OwnedWeightBounds
    nonreachable: OwnedWeightBounds
    unresolved_covered: OwnedWeightBounds
    covered_lower: float
    missing_weight_upper: float
    Q_lower: float
    Q_upper: float


@dataclass(frozen=True)
class PitchBandFractionBounds:
    """Certified omitted-band fraction after supplied numerator/denominator errors."""

    lower: float
    upper: float
    pitch_weight_lower: float
    pitch_weight_upper: float
    denominator: WeightBounds
    bound_scope: str


def build_population_context(
    field: BoozerFieldLike,
    source_profile: SourceProfile,
    config: PopulationQuadratureConfig = PopulationQuadratureConfig(),
    *,
    source_name: str,
    surface_maximum: ArrayLike | Callable[[FloatArray], ArrayLike] | None = None,
    surface_maximum_scope: str = "sampled-angular-grid-estimate",
    surface_maximum_is_certified_upper: bool = False,
) -> PopulationContext:
    """Evaluate reusable spatial data for the §12.1 population integrals.

    The source callable receives ``rho=sqrt(s)``, never ``s``.  When
    ``surface_maximum`` is omitted, the angular grid maximum is retained only as
    an estimate.  Callers must opt in explicitly before treating a supplied
    profile as a certified upper bound.
    """
    if not source_name.strip():
        raise ValueError("source_name must declare the source used")
    if field.nfp < 1:
        raise ValueError("field.nfp must be positive")

    legendre_nodes, legendre_weights = np.polynomial.legendre.leggauss(config.n_s)
    nodes_s = 0.5 * (legendre_nodes + 1.0)
    weights_s = 0.5 * legendre_weights
    theta = np.linspace(0.0, 2.0 * np.pi, config.n_theta, endpoint=False)
    zeta_period = 2.0 * np.pi / field.nfp
    zeta = np.linspace(0.0, zeta_period, config.n_zeta, endpoint=False)

    source_values = _source_values(source_profile, np.sqrt(nodes_s))
    absolute_C = _profile_values(np.abs(field.C(nodes_s)), nodes_s, "C(s)")
    s_grid = nodes_s[:, np.newaxis, np.newaxis]
    theta_grid = theta[np.newaxis, :, np.newaxis]
    zeta_grid = zeta[np.newaxis, np.newaxis, :]
    B = np.asarray(field.B(s_grid, theta_grid, zeta_grid), dtype=float)
    expected_shape = (config.n_s, config.n_theta, config.n_zeta)
    try:
        B = np.broadcast_to(B, expected_shape)
    except ValueError as error:
        raise ValueError("B must broadcast over the population grid") from error
    if not np.all(np.isfinite(B)) or np.any(B <= 0.0):
        raise ValueError("B must be finite and positive on the population grid")

    sampled_maximum = np.max(B, axis=(1, 2))
    if surface_maximum is None:
        maximum = sampled_maximum
        surface_maximum_is_certified_upper = False
    else:
        supplied = (
            surface_maximum(nodes_s) if callable(surface_maximum) else surface_maximum
        )
        maximum = _profile_values(supplied, nodes_s, "surface maximum")
        tolerance = 64.0 * np.finfo(float).eps * np.maximum(1.0, sampled_maximum)
        if np.any(maximum + tolerance < sampled_maximum):
            raise ValueError("surface maximum lies below a sampled field value")
    if not surface_maximum_scope.strip():
        raise ValueError("surface_maximum_scope must describe the maximum provenance")

    return PopulationContext(
        config=config,
        nodes_s=nodes_s,
        weights_s=weights_s,
        theta=theta,
        zeta=zeta,
        B=np.asarray(B),
        source_values=source_values,
        absolute_C=absolute_C,
        surface_maximum=maximum,
        source_name=source_name,
        surface_maximum_scope=surface_maximum_scope,
        surface_maximum_is_certified_upper=surface_maximum_is_certified_upper,
        angular_measure_per_node=(2.0 * np.pi * zeta_period)
        / (config.n_theta * config.n_zeta),
    )


def compute_population_slice(
    context: PopulationContext,
    b: float,
    *,
    linewise_trapped: LinewiseTrapping | None = None,
    dense_line_assumption: bool = False,
) -> PopulationSliceEstimate:
    """Estimate total trapped weight at conserved bounce field ``b`` (§12.1).

    A supplied ``linewise_trapped`` predicate is authoritative for each allowed
    quadrature point.  Otherwise the surface-maximum shortcut is equality only
    when ``dense_line_assumption`` is declared.  Without that assumption, a
    certified surface maximum produces an upper-count model, not linewise truth.
    """
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError("b must be finite and positive")
    if linewise_trapped is not None and dense_line_assumption:
        raise ValueError("choose linewise trapping or the dense-line assumption")

    s_grid, theta_grid, zeta_grid = _coordinate_grids(context)
    allowed = context.B < b
    if linewise_trapped is not None:
        supplied = np.asarray(
            linewise_trapped(s_grid, theta_grid, zeta_grid, float(b), context.B)
        )
        try:
            trapped = np.broadcast_to(supplied, context.B.shape).astype(
                bool, copy=False
            )
        except ValueError as error:
            raise ValueError(
                "linewise trapping mask must broadcast over the grid"
            ) from error
        if np.any(trapped & ~allowed):
            raise ValueError("linewise trapping mask includes points with B >= b")
        trapping_scope = "linewise_mask"
        bound_scope = "estimate"
        assumptions = ("linewise trapping predicate supplied by caller",)
    else:
        trapped = allowed & (context.surface_maximum[:, np.newaxis, np.newaxis] > b)
        if dense_line_assumption:
            trapping_scope = "dense_line_surface_maximum"
            bound_scope = "estimate"
            assumptions = (
                "field lines are dense on almost every contributing surface",
            )
        else:
            if not context.surface_maximum_is_certified_upper:
                raise ValueError(
                    "surface-maximum upper counting requires a certified upper profile"
                )
            trapping_scope = "surface_maximum_upper_only"
            bound_scope = "quadrature_estimate_of_upper_model"
            assumptions = (
                "certified surface maximum is only an upper count for linewise trapping",
            )

    integrand = np.zeros_like(context.B)
    denominator = np.sqrt(1.0 - context.B[trapped] / b)
    integrand[trapped] = 1.0 / (context.B[trapped] * denominator)
    radial_density = (
        context.source_values
        * context.absolute_C
        * context.angular_measure_per_node
        * np.sum(integrand, axis=(1, 2))
    )
    total_weight = float(np.sum(context.weights_s * radial_density))
    if not np.isfinite(total_weight) or total_weight < 0.0:
        raise ValueError("population quadrature produced an invalid weight")

    return PopulationSliceEstimate(
        b=float(b),
        total_weight=total_weight,
        nodes_s=context.nodes_s,
        radial_density=radial_density,
        source_name=context.source_name,
        trapping_scope=trapping_scope,
        surface_maximum_scope=context.surface_maximum_scope,
        bound_scope=bound_scope,
        uncontrolled_errors=(
            "field representation / interpolation",
            "surface-maximum profile unless independently certified",
            "spatial quadrature",
        )
        + assumptions,
    )


def compute_pitch_band_estimate(
    context: PopulationContext,
    b_lower: float,
    b_upper: float,
    denominator_estimate: float,
    *,
    dense_line_assumption: bool = False,
) -> PitchBandEstimate:
    """Integrate a whole omitted pitch band using the exact ``b**-2`` primitive.

    At a spatial point with local field ``B`` and trapping ceiling ``M``,

    ``integral db / (B b**2 sqrt(1-B/b)) = 2 sqrt(1-B/b) / B**2``.

    The integration interval is intersected with ``B < b < M``.  Dividing the
    resulting pitch weight by ``2 V_h`` gives the fraction available to the
    omitted band (DESIGN.md §§4.3, 12.1, and 12.4).
    """
    if (
        not np.isfinite(b_lower)
        or not np.isfinite(b_upper)
        or b_lower <= 0.0
        or b_upper <= b_lower
    ):
        raise ValueError("pitch band requires finite 0 < b_lower < b_upper")
    if not np.isfinite(denominator_estimate) or denominator_estimate <= 0.0:
        raise ValueError("denominator estimate must be finite and positive")
    if dense_line_assumption:
        trapping_scope = "dense_line_surface_maximum"
        bound_scope = "estimate"
        assumptions = ("dense-line surface-maximum equality",)
    else:
        if not context.surface_maximum_is_certified_upper:
            raise ValueError(
                "upper band counting requires a certified surface-maximum profile"
            )
        trapping_scope = "surface_maximum_upper_only"
        bound_scope = "quadrature_estimate_of_upper_model"
        assumptions = ("surface maximum supplies only an upper trapping ceiling",)

    lower = np.maximum(context.B, b_lower)
    upper = np.minimum(
        np.broadcast_to(
            context.surface_maximum[:, np.newaxis, np.newaxis], context.B.shape
        ),
        b_upper,
    )
    active = upper > lower
    u_difference = np.zeros_like(context.B)
    u_difference[active] = np.sqrt(1.0 - context.B[active] / upper[active]) - np.sqrt(
        1.0 - context.B[active] / lower[active]
    )
    spatial_integrand = u_difference / context.B**2
    radial_fraction_density = (
        context.source_values
        * context.absolute_C
        * context.angular_measure_per_node
        * np.sum(spatial_integrand, axis=(1, 2))
        / denominator_estimate
    )
    half_pitch_weight = float(
        denominator_estimate * np.sum(context.weights_s * radial_fraction_density)
    )
    pitch_weight = 2.0 * half_pitch_weight
    fraction = half_pitch_weight / denominator_estimate
    if not np.isfinite(fraction) or fraction < 0.0:
        raise ValueError("pitch-band quadrature produced an invalid fraction")

    return PitchBandEstimate(
        b_lower=float(b_lower),
        b_upper=float(b_upper),
        pitch_weight=pitch_weight,
        denominator_estimate=float(denominator_estimate),
        fraction=fraction,
        nodes_s=context.nodes_s,
        radial_fraction_density=radial_fraction_density,
        source_name=context.source_name,
        trapping_scope=trapping_scope,
        surface_maximum_scope=context.surface_maximum_scope,
        bound_scope=bound_scope,
        uncontrolled_errors=(
            "field representation / interpolation",
            "surface-maximum profile unless independently certified",
            "spatial quadrature",
            "denominator quadrature",
        )
        + assumptions,
    )


def build_population_ledger(
    total: WeightBounds,
    reachable: OwnedWeightBounds,
    nonreachable: OwnedWeightBounds,
    unresolved_covered: OwnedWeightBounds | None = None,
) -> PopulationLedger:
    """Close the disjoint lower/upper ledger according to DESIGN.md §12.2.

    ``Q_lower`` is definitely reachable lower weight.  ``Q_upper`` subtracts
    only definitely nonreachable lower weight from the independent total upper
    bound.  Missing weight uses ``total.upper - covered.lower``; owner overlap or
    a materially negative residual is rejected rather than clamped.
    """
    if unresolved_covered is None:
        unresolved_covered = OwnedWeightBounds(
            owner_ids=np.empty(0, dtype=np.int64),
            lower=np.empty(0),
            upper=np.empty(0),
        )
    categories = (reachable, nonreachable, unresolved_covered)
    owner_ids = np.concatenate([category.owner_ids for category in categories])
    if np.unique(owner_ids).size != owner_ids.size:
        raise ValueError("population owner IDs overlap across ledger categories")

    covered_lower = float(sum(category.lower_total for category in categories))
    scale = max(1.0, total.upper, covered_lower)
    tolerance = 64.0 * np.finfo(float).eps * scale
    if covered_lower > total.upper + tolerance:
        raise ValueError("covered lower weight exceeds total upper bound")
    missing_weight_upper = total.upper - covered_lower
    if missing_weight_upper < 0.0:
        missing_weight_upper = 0.0

    Q_lower = reachable.lower_total
    Q_upper = total.upper - nonreachable.lower_total
    if Q_lower > Q_upper + tolerance:
        raise ValueError(
            "reachable lower weight exceeds possible-reachable upper bound"
        )
    if Q_upper < 0.0:
        raise ValueError("nonreachable lower weight exceeds total upper bound")
    return PopulationLedger(
        total=total,
        reachable=reachable,
        nonreachable=nonreachable,
        unresolved_covered=unresolved_covered,
        covered_lower=covered_lower,
        missing_weight_upper=missing_weight_upper,
        Q_lower=Q_lower,
        Q_upper=Q_upper,
    )


def enclose_pitch_band_fraction(
    estimate: PitchBandEstimate,
    *,
    pitch_weight_absolute_error: float,
    denominator: WeightBounds,
    bound_scope: str,
) -> PitchBandFractionBounds:
    """Turn supplied certified errors into an omitted-band fraction enclosure.

    The lower ratio uses the denominator upper bound and the upper ratio uses the
    positive denominator lower bound, as required by DESIGN.md §13.4.  This
    function does not promote convergence differences into error bounds: the
    caller must supply a justified absolute error and describe its scope.
    """
    if denominator.lower <= 0.0:
        raise ValueError("denominator lower bound must be positive")
    if (
        not np.isfinite(pitch_weight_absolute_error)
        or pitch_weight_absolute_error < 0.0
    ):
        raise ValueError("pitch-weight error must be finite and nonnegative")
    if not bound_scope.strip():
        raise ValueError("bound_scope must describe the certified error scope")
    pitch_weight_lower = max(0.0, estimate.pitch_weight - pitch_weight_absolute_error)
    pitch_weight_upper = estimate.pitch_weight + pitch_weight_absolute_error
    lower = pitch_weight_lower / (2.0 * denominator.upper)
    upper = pitch_weight_upper / (2.0 * denominator.lower)
    return PitchBandFractionBounds(
        lower=lower,
        upper=upper,
        pitch_weight_lower=pitch_weight_lower,
        pitch_weight_upper=pitch_weight_upper,
        denominator=denominator,
        bound_scope=bound_scope,
    )


def _coordinate_grids(
    context: PopulationContext,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    return (
        np.broadcast_to(context.nodes_s[:, np.newaxis, np.newaxis], context.B.shape),
        np.broadcast_to(context.theta[np.newaxis, :, np.newaxis], context.B.shape),
        np.broadcast_to(context.zeta[np.newaxis, np.newaxis, :], context.B.shape),
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
    return np.asarray(values)


def _profile_values(values: ArrayLike, nodes_s: FloatArray, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    try:
        array = np.broadcast_to(array, nodes_s.shape)
    except ValueError as error:
        raise ValueError(f"{name} must return values broadcastable to s") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} returned non-finite values")
    return np.asarray(array)
