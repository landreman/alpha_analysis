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
BOUND_SCOPES = frozenset(
    {"model_enclosure", "field_enclosure", "estimate", "statistical_interval"}
)


@dataclass(frozen=True)
class LinewiseTrappingMasks:
    """Definite/possible linewise trapping with explicit unresolved reasons.

    The masks use the population grid. ``definitely_trapped`` must be a subset
    of ``possibly_trapped``. A nonempty difference represents unresolved line
    tracing and requires at least one reason; it is never converted to zero
    population (DESIGN.md §§12.1 and 21.2).
    """

    definitely_trapped: ArrayLike
    possibly_trapped: ArrayLike
    unresolved_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(not reason.strip() for reason in self.unresolved_reasons):
            raise ValueError("linewise unresolved reasons must be nonempty strings")


LinewiseTrapping = Callable[
    [FloatArray, FloatArray, FloatArray, float, FloatArray], LinewiseTrappingMasks
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
    surface_maximum_is_certified_exact: bool
    angular_measure_per_node: float


@dataclass(frozen=True)
class PopulationSliceEstimate:
    """Quadrature estimate of ``Q_total(b)`` from DESIGN.md §12.1.

    The total weights are unnormalized and have the units of
    ``h |C| / B`` times the three-coordinate measure.  It is not an
    accessibility result.  The radial densities exclude Gauss weights so they
    can be plotted as functions of normalized flux ``s``. The interval is an
    unresolved-model interval, not a numerical enclosure unless its recorded
    errors are subsequently bounded.
    """

    b: float
    total_weight_lower: float
    total_weight_upper: float
    nodes_s: FloatArray
    radial_density_lower: FloatArray
    radial_density_upper: FloatArray
    source_name: str
    trapping_scope: str
    surface_maximum_scope: str
    bound_scope: str
    uncontrolled_errors: tuple[str, ...]
    surface_maximum_is_certified_upper: bool
    surface_maximum_is_certified_exact: bool

    @property
    def total_weight(self) -> float:
        """Return a scalar only when trapping has no unresolved population."""
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, self.total_weight_upper)
        if self.total_weight_upper - self.total_weight_lower > tolerance:
            raise ValueError(
                "population is interval-valued; use total_weight_lower/upper"
            )
        return self.total_weight_lower

    @property
    def radial_density(self) -> FloatArray:
        """Return scalar radial density only when lower and upper agree."""
        scale = np.maximum(1.0, np.abs(self.radial_density_upper))
        tolerance = 64.0 * np.finfo(float).eps * scale
        if np.any(self.radial_density_upper - self.radial_density_lower > tolerance):
            raise ValueError(
                "population is interval-valued; use radial_density_lower/upper"
            )
        return self.radial_density_lower


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
    surface_maximum_is_certified_upper: bool
    surface_maximum_is_certified_exact: bool
    dense_line_certification: str | None


@dataclass(frozen=True)
class WeightBounds:
    """Finite nonnegative bounds with explicit certification/error scope."""

    lower: float
    upper: float
    bound_scope: str
    is_certified: bool
    uncontrolled_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError("weight bounds must be finite")
        if self.lower < 0.0 or self.upper < self.lower:
            raise ValueError("weight bounds require 0 <= lower <= upper")
        _validate_bound_scope(self.bound_scope)
        enclosure_scope = self.bound_scope in {"model_enclosure", "field_enclosure"}
        if self.is_certified != enclosure_scope:
            raise ValueError(
                "is_certified must be true exactly for model/field enclosure scope"
            )
        if any(not error.strip() for error in self.uncontrolled_errors):
            raise ValueError("uncontrolled errors must be nonempty strings")
        if self.is_certified and self.uncontrolled_errors:
            raise ValueError("certified bounds cannot retain uncontrolled errors")


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
    bound_scope: str
    is_certified: bool
    uncontrolled_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_bound_scope(self.bound_scope)
        if self.is_certified != (
            self.bound_scope in {"model_enclosure", "field_enclosure"}
        ):
            raise ValueError("owned weight certification must match its bound_scope")
        if any(not error.strip() for error in self.uncontrolled_errors):
            raise ValueError("owned weight errors must be nonempty strings")
        if self.is_certified and self.uncontrolled_errors:
            raise ValueError(
                "certified owned weights cannot retain uncontrolled errors"
            )
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
    bound_scope: str
    uncontrolled_errors: tuple[str, ...]


@dataclass(frozen=True)
class PitchBandFractionBounds:
    """Omitted-band enclosure with source and precursor-estimate provenance.

    ``estimate_uncontrolled_errors`` describes errors in the precursor estimate;
    a claimed enclosure requires the caller's supplied absolute error to control
    the applicable numerical/field errors (DESIGN.md §13.4).
    """

    lower: float
    upper: float
    pitch_weight_lower: float
    pitch_weight_upper: float
    denominator: WeightBounds
    bound_scope: str
    source_name: str
    trapping_scope: str
    surface_maximum_scope: str
    dense_line_certification: str | None
    estimate_uncontrolled_errors: tuple[str, ...]


def build_population_context(
    field: BoozerFieldLike,
    source_profile: SourceProfile,
    config: PopulationQuadratureConfig = PopulationQuadratureConfig(),
    *,
    source_name: str,
    surface_maximum: ArrayLike | Callable[[FloatArray], ArrayLike] | None = None,
    surface_maximum_scope: str = "sampled-angular-grid-estimate",
    surface_maximum_is_certified_upper: bool = False,
    surface_maximum_is_certified_exact: bool = False,
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
        surface_maximum_is_certified_exact = False
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
    if surface_maximum_is_certified_exact:
        surface_maximum_is_certified_upper = True

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
        surface_maximum_is_certified_exact=surface_maximum_is_certified_exact,
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
        supplied = linewise_trapped(s_grid, theta_grid, zeta_grid, float(b), context.B)
        if not isinstance(supplied, LinewiseTrappingMasks):
            raise TypeError(
                "linewise_trapped must return LinewiseTrappingMasks so unresolved "
                "traces remain explicit"
            )
        definitely_trapped = _broadcast_bool_mask(
            supplied.definitely_trapped, context.B.shape, "definitely_trapped"
        )
        possibly_trapped = _broadcast_bool_mask(
            supplied.possibly_trapped, context.B.shape, "possibly_trapped"
        )
        if np.any(definitely_trapped & ~possibly_trapped):
            raise ValueError("definitely_trapped must be a subset of possibly_trapped")
        if np.any(possibly_trapped & ~allowed):
            raise ValueError("linewise trapping masks include points with B >= b")
        unresolved = possibly_trapped & ~definitely_trapped
        if np.any(unresolved) and not supplied.unresolved_reasons:
            raise ValueError("uncertain linewise trapping requires a recorded reason")
        trapping_scope = "linewise_masks"
        bound_scope = "estimate"
        assumptions = ("linewise trapping predicate supplied by caller",)
        unresolved_errors = tuple(
            f"unresolved linewise trapping: {reason}"
            for reason in supplied.unresolved_reasons
        )
    else:
        surface_trapped = allowed & (
            context.surface_maximum[:, np.newaxis, np.newaxis] > b
        )
        if dense_line_assumption:
            definitely_trapped = surface_trapped
            possibly_trapped = surface_trapped
            trapping_scope = "dense_line_surface_maximum"
            bound_scope = "estimate"
            assumptions = (
                "field lines are dense on almost every contributing surface",
            )
            unresolved_errors = ()
        else:
            if not context.surface_maximum_is_certified_upper:
                raise ValueError(
                    "surface-maximum upper counting requires a certified upper profile"
                )
            definitely_trapped = np.zeros_like(surface_trapped)
            possibly_trapped = surface_trapped
            trapping_scope = "surface_maximum_upper_only"
            bound_scope = "estimate"
            assumptions = (
                "certified surface maximum is only an upper count for linewise trapping",
            )
            unresolved_errors = (
                "linewise trapping is bounded only by the surface-maximum upper model",
            )

    radial_density_lower = _slice_radial_density(context, b, definitely_trapped)
    radial_density_upper = _slice_radial_density(context, b, possibly_trapped)
    total_weight_lower = float(np.sum(context.weights_s * radial_density_lower))
    total_weight_upper = float(np.sum(context.weights_s * radial_density_upper))
    if (
        not np.isfinite(total_weight_lower)
        or not np.isfinite(total_weight_upper)
        or total_weight_lower < 0.0
        or total_weight_upper < total_weight_lower
    ):
        raise ValueError("population quadrature produced an invalid weight")

    return PopulationSliceEstimate(
        b=float(b),
        total_weight_lower=total_weight_lower,
        total_weight_upper=total_weight_upper,
        nodes_s=context.nodes_s,
        radial_density_lower=radial_density_lower,
        radial_density_upper=radial_density_upper,
        source_name=context.source_name,
        trapping_scope=trapping_scope,
        surface_maximum_scope=context.surface_maximum_scope,
        bound_scope=bound_scope,
        uncontrolled_errors=(
            "field representation / interpolation",
            "surface-maximum profile unless independently certified",
            "spatial quadrature",
        )
        + assumptions
        + unresolved_errors,
        surface_maximum_is_certified_upper=(context.surface_maximum_is_certified_upper),
        surface_maximum_is_certified_exact=(context.surface_maximum_is_certified_exact),
    )


def compute_pitch_band_estimate(
    context: PopulationContext,
    b_lower: float,
    b_upper: float,
    denominator_estimate: float,
    *,
    dense_line_assumption: bool = False,
    dense_line_certification: str | None = None,
) -> PitchBandEstimate:
    """Integrate a whole omitted pitch band using the exact ``b**-2`` primitive.

    At a spatial point with local field ``B`` and trapping ceiling ``M``,

    ``integral db / (B b**2 sqrt(1-B/b)) = 2 sqrt(1-B/b) / B**2``.

    The integration interval is intersected with ``B < b < M``.  Dividing the
    resulting pitch weight by ``2 V_h`` gives the fraction available to the
    omitted band (DESIGN.md §§4.3, 12.1, and 12.4). A declared dense-line
    assumption is not a proof: an independently justified certificate must be
    supplied to permit a positive lower bound downstream.
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
    if dense_line_certification is not None and (
        not dense_line_assumption or not dense_line_certification.strip()
    ):
        raise ValueError(
            "dense-line certification requires the dense-line assumption and a "
            "nonempty justification"
        )
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
        bound_scope = "estimate"
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
        surface_maximum_is_certified_upper=(context.surface_maximum_is_certified_upper),
        surface_maximum_is_certified_exact=(context.surface_maximum_is_certified_exact),
        dense_line_certification=dense_line_certification,
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
    if not total.is_certified:
        raise ValueError(
            "population ledger requires a certified total upper bound; an "
            "uncontrolled estimate cannot define Q_upper"
        )
    if unresolved_covered is None:
        unresolved_covered = OwnedWeightBounds(
            owner_ids=np.empty(0, dtype=np.int64),
            lower=np.empty(0),
            upper=np.empty(0),
            bound_scope=total.bound_scope,
            is_certified=True,
        )
    categories = (reachable, nonreachable, unresolved_covered)
    if any(not category.is_certified for category in categories):
        raise ValueError("population ledger requires certified owned weight bounds")
    bound_scope = (
        "model_enclosure"
        if "model_enclosure"
        in (total.bound_scope, *(c.bound_scope for c in categories))
        else "field_enclosure"
    )
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
        bound_scope=bound_scope,
        uncontrolled_errors=total.uncontrolled_errors,
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
    if not denominator.is_certified:
        raise ValueError("denominator bounds must be certified for an enclosure")
    if (
        bound_scope == "field_enclosure"
        and denominator.bound_scope != "field_enclosure"
    ):
        raise ValueError("field enclosure requires a field-enclosed denominator")
    if denominator.lower <= 0.0:
        raise ValueError("denominator lower bound must be positive")
    if (
        not np.isfinite(pitch_weight_absolute_error)
        or pitch_weight_absolute_error < 0.0
    ):
        raise ValueError("pitch-weight error must be finite and nonnegative")
    _validate_bound_scope(bound_scope)
    if bound_scope not in {"model_enclosure", "field_enclosure"}:
        raise ValueError("pitch-band bounds require model or field enclosure scope")
    if not estimate.surface_maximum_is_certified_upper:
        raise ValueError("a band enclosure requires a certified surface maximum")
    if (
        estimate.trapping_scope == "dense_line_surface_maximum"
        and estimate.dense_line_certification is not None
        and estimate.surface_maximum_is_certified_exact
    ):
        pitch_weight_lower = max(
            0.0, estimate.pitch_weight - pitch_weight_absolute_error
        )
    elif estimate.trapping_scope in {
        "dense_line_surface_maximum",
        "surface_maximum_upper_only",
    }:
        pitch_weight_lower = 0.0
    else:
        raise ValueError("pitch-band estimate has no certifiable trapping scope")
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
        source_name=estimate.source_name,
        trapping_scope=estimate.trapping_scope,
        surface_maximum_scope=estimate.surface_maximum_scope,
        dense_line_certification=estimate.dense_line_certification,
        estimate_uncontrolled_errors=estimate.uncontrolled_errors,
    )


def _coordinate_grids(
    context: PopulationContext,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    return (
        np.broadcast_to(context.nodes_s[:, np.newaxis, np.newaxis], context.B.shape),
        np.broadcast_to(context.theta[np.newaxis, :, np.newaxis], context.B.shape),
        np.broadcast_to(context.zeta[np.newaxis, np.newaxis, :], context.B.shape),
    )


def _broadcast_bool_mask(
    values: ArrayLike, shape: tuple[int, ...], name: str
) -> BoolArray:
    array = np.asarray(values)
    if array.dtype != np.bool_:
        raise ValueError(f"{name} must contain booleans")
    try:
        array = np.broadcast_to(array, shape)
    except ValueError as error:
        raise ValueError(f"{name} must broadcast over the population grid") from error
    return np.asarray(array)


def _slice_radial_density(
    context: PopulationContext, b: float, trapped: BoolArray
) -> FloatArray:
    integrand = np.zeros_like(context.B)
    denominator = np.sqrt(1.0 - context.B[trapped] / b)
    integrand[trapped] = 1.0 / (context.B[trapped] * denominator)
    return (
        context.source_values
        * context.absolute_C
        * context.angular_measure_per_node
        * np.sum(integrand, axis=(1, 2))
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


def _validate_bound_scope(bound_scope: str) -> None:
    if bound_scope not in BOUND_SCOPES:
        allowed = ", ".join(sorted(BOUND_SCOPES))
        raise ValueError(f"bound_scope must be one of: {allowed}")


def _profile_values(values: ArrayLike, nodes_s: FloatArray, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    try:
        array = np.broadcast_to(array, nodes_s.shape)
    except ValueError as error:
        raise ValueError(f"{name} must return values broadcastable to s") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} returned non-finite values")
    return np.asarray(array)
