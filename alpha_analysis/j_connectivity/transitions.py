"""Pre-cut transition mapping on marginal-maximum curves.

This module implements DESIGN.md §§5.3--5.4 and 10.2--10.4. Each sampled
``GAMMA_MAX`` point is traced backward and forward on that *same lifted field
line* to obtain the companion point ``a``, outgoing point ``d``, and the three
limiting half-bounce actions. Ports are index-aligned by the common critical
curve parameter ``u``; no Euclidean nearest-neighbor association is used.

Logical coordinates are dimensionless ``(x=sqrt(s) cos(theta),
y=sqrt(s) sin(theta), zeta)`` with angles in radians. Actions have the length
units of ``G`` and ``I``. Nongeneric equal-height contacts and all numerical
failures remain explicit and receive ``NaN`` actions (DESIGN.md §21.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace
from typing import Literal

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .critical_curves import CriticalCurveStatus, CriticalCurves, CriticalKind
from .field import BoozerFieldLike
from .types import FloatArray, IntArray, TransitionStatus
from .well_trace import _mode_frequency


@dataclass(frozen=True)
class TransitionMappingConfig:
    """Numerical controls for DESIGN.md §10.2 transition tracing.

    Field and derivative tolerances use the field's units and radians.
    ``action_quadrature_order`` is the baseline Gauss--Legendre order for each
    child-action refinement check. Order doubles until the configured action
    tolerance or ``max_action_quadrature_order``; when that optional cap is
    omitted, at least order 512 and one refinement are tried. The reported
    value is the finest result and the last difference is its error estimate. Every
    detected internal extremum is a quadrature breakpoint so high-mode wells
    cannot alias to a plausible action. ``A_W`` is independently evaluated by
    adaptive quadrature over ``[a,d]`` with the same extrema and tangent point
    supplied as interior breakpoints. These controls are the transition-action
    refinement dimension required by §§21.3 and 23.
    ``max_curve_samples`` optionally selects a deterministic uniform subset of
    the critical polyline's existing cumulative-arc-length samples; ``None``
    retains every vertex. It is an explicit validation/performance control,
    not geometric coarsening: ``total_u_length`` and source vertex IDs remain
    those of the authoritative critical curve.
    ``additivity_atol`` has action-length units and ``additivity_rtol`` is
    dimensionless.
    """

    samples_per_field_period: int = 64
    samples_per_wavelength: int = 24
    max_field_periods: int = 128
    root_atol_B: float = 1.0e-10
    root_atol_zeta: float = 1.0e-12
    root_rtol: float = 1.0e-12
    tangent_atol_B: float = 1.0e-9
    tangent_slope_tolerance: float = 1.0e-8
    D2_tolerance: float = 1.0e-10
    action_quadrature_order: int = 32
    action_quadrature_atol: float = 1.0e-11
    action_quadrature_rtol: float = 1.0e-11
    additivity_atol: float = 1.0e-8
    additivity_rtol: float = 1.0e-7
    field_identity_tolerance: float = 1.0e-8
    max_curve_samples: int | None = None
    max_action_quadrature_order: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "samples_per_field_period",
            "samples_per_wavelength",
            "max_field_periods",
            "action_quadrature_order",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_curve_samples is not None and self.max_curve_samples < 2:
            raise ValueError("max_curve_samples must be at least two when supplied")
        if (
            self.max_action_quadrature_order is not None
            and self.max_action_quadrature_order < 1
        ):
            raise ValueError("max_action_quadrature_order must be positive")
        if (
            self.max_action_quadrature_order is not None
            and self.max_action_quadrature_order <= self.action_quadrature_order
        ):
            raise ValueError(
                "max_action_quadrature_order must exceed action_quadrature_order"
            )
        for name in (
            "root_atol_B",
            "root_atol_zeta",
            "root_rtol",
            "tangent_atol_B",
            "tangent_slope_tolerance",
            "D2_tolerance",
            "action_quadrature_atol",
            "action_quadrature_rtol",
            "additivity_atol",
            "additivity_rtol",
            "field_identity_tolerance",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class TransitionPort:
    """One limiting-well port sampled at the transition's common ``u``.

    ``points`` are reduced logical ``(x,y,zeta)`` points. The parent and
    child-1 ports both lie on the not-yet-inserted companion curve ``T``;
    child-3 lies on ``GAMMA_MAX``. ``zeta_unwrapped`` preserves the lifted
    location used by the trace. ``source_vertex_ids`` is ``-1`` for the
    pre-cut ``T`` ports and the critical-curve vertex ID for child-3.
    """

    role: Literal["parent", "child_1", "child_3", "generic"]
    points: FloatArray
    zeta_unwrapped: FloatArray
    action_values: FloatArray
    quadrature_error: FloatArray
    source_vertex_ids: IntArray
    sheet_id: int = -1

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        zeta = np.asarray(self.zeta_unwrapped, dtype=np.float64)
        action = np.asarray(self.action_values, dtype=np.float64)
        error = np.asarray(self.quadrature_error, dtype=np.float64)
        vertex_ids = np.asarray(self.source_vertex_ids, dtype=np.int64)
        if self.role not in {"parent", "child_1", "child_3", "generic"}:
            raise ValueError("unknown transition port role")
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("transition port points must have shape (n, 3)")
        n_samples = len(points)
        if any(
            values.shape != (n_samples,) for values in (zeta, action, error, vertex_ids)
        ):
            raise ValueError("transition port arrays must share one sample axis")
        finite_rows = np.all(np.isfinite(points), axis=1)
        nan_rows = np.all(np.isnan(points), axis=1)
        if np.any(~(finite_rows | nan_rows)):
            raise ValueError("each transition port point must be finite or all-NaN")
        for name, values in (
            ("points", points),
            ("zeta_unwrapped", zeta),
            ("action_values", action),
            ("quadrature_error", error),
            ("source_vertex_ids", vertex_ids),
        ):
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class TransitionCurve:
    """One matched pre-cut transition correspondence (DESIGN.md §10.4).

    ``field_line_identity[:,0]`` is normalized flux ``s`` and column 1 is the
    lifted straight-field-line label ``alpha=theta-iota*zeta`` in radians.
    ``event_zeta_unwrapped`` stores ``(zeta_a,zeta_m,zeta_d)`` for diagnostics.
    Every port uses exactly this sample ordering, making a mismatched
    nearest-neighbor association structurally impossible. ``sample_status``
    keeps nongeneric endpoints and numerical failures explicit without
    discarding valid samples elsewhere on the same critical polyline;
    ``sample_failure_reason`` identifies the failed stage for convergence
    diagnostics.

    ``interior_maximum_count`` is the number of other maxima the root scan
    detected inside each sample's parent well -- a resolution-dependent count,
    §21.3 dimension 5 -- and is ``-1`` where no well was traced;
    ``barrier_margin`` is ``b`` minus the *highest* of them in field units, is
    ``inf`` when the well has none, and is ``NaN`` where no well was traced;
    it therefore discriminates a fold from
    an equal-height contact only when the bracket's count change belongs to
    that highest barrier, which is not so in a well holding many of them. ``contact_sample_pairs`` holds the
    adjacent sample indices that bracket a nongeneric event the sampling
    stepped over (DESIGN.md §5.4) -- a barrier crossing ``b``, which makes the
    port actions discontinuous in ``u``, or a fold in which a barrier
    annihilates with its minimum, which does not; a small ``barrier_margin``
    at the bracket distinguishes the first. Such a curve is ``MULTIWAY`` even
    when every sample is regular, but never in place of a sample-level
    failure: a capped or failed sample keeps its own curve status and the
    bracket is still recorded. Rows are in sample order and are not sorted: a
    closed curve's wraparound row is ``(n_samples - 1, 0)``, whose ``u``
    values decrease.
    """

    transition_id: int
    b: float
    u: FloatArray
    total_u_length: float
    ports: tuple[TransitionPort, ...]
    marginal_points: FloatArray
    field_line_identity: FloatArray
    event_zeta_unwrapped: FloatArray
    additivity_residual: FloatArray
    status: TransitionStatus
    source_critical_status: CriticalCurveStatus
    controls: TransitionMappingConfig
    sample_status: tuple[TransitionStatus, ...] = dataclass_field(default_factory=tuple)
    sample_failure_reason: tuple[str, ...] = dataclass_field(default_factory=tuple)
    interior_maximum_count: IntArray | None = None
    barrier_margin: FloatArray | None = None
    contact_sample_pairs: IntArray | None = None

    def __post_init__(self) -> None:
        u = np.asarray(self.u, dtype=np.float64)
        marginal = np.asarray(self.marginal_points, dtype=np.float64)
        identity = np.asarray(self.field_line_identity, dtype=np.float64)
        event_zeta = np.asarray(self.event_zeta_unwrapped, dtype=np.float64)
        residual = np.asarray(self.additivity_residual, dtype=np.float64)
        sample_status = tuple(self.sample_status)
        sample_failure_reason = tuple(self.sample_failure_reason)
        n_samples = len(u)
        interior_maximum_count = (
            np.full(n_samples, -1, dtype=np.int64)
            if self.interior_maximum_count is None
            else np.asarray(self.interior_maximum_count, dtype=np.int64)
        )
        barrier_margin = (
            np.full(n_samples, np.nan)
            if self.barrier_margin is None
            else np.asarray(self.barrier_margin, dtype=np.float64)
        )
        contact_sample_pairs = (
            np.empty((0, 2), dtype=np.int64)
            if self.contact_sample_pairs is None
            else np.asarray(self.contact_sample_pairs, dtype=np.int64).reshape(-1, 2)
        )
        if not sample_status:
            sample_status = (self.status,) * n_samples
        if not sample_failure_reason:
            reason = (
                "regular"
                if self.status is TransitionStatus.REGULAR
                else self.status.name.lower()
            )
            sample_failure_reason = (reason,) * n_samples
        if not np.isfinite(self.b) or self.b <= 0.0:
            raise ValueError("transition b must be finite and positive")
        if u.ndim != 1 or (len(u) > 1 and np.any(np.diff(u) <= 0.0)):
            raise ValueError("transition u must be a strictly increasing vector")
        if marginal.shape != (n_samples, 3):
            raise ValueError("marginal_points must have shape (n_samples, 3)")
        if identity.shape != (n_samples, 2):
            raise ValueError("field_line_identity must have shape (n_samples, 2)")
        if event_zeta.shape != (n_samples, 3):
            raise ValueError("event_zeta_unwrapped must have shape (n_samples, 3)")
        if residual.shape != (n_samples,):
            raise ValueError("additivity_residual must have one value per sample")
        if len(sample_status) != n_samples or any(
            not isinstance(status, TransitionStatus) for status in sample_status
        ):
            raise ValueError(
                "sample_status must contain one TransitionStatus per sample"
            )
        if len(sample_failure_reason) != n_samples or any(
            not isinstance(reason, str) or not reason
            for reason in sample_failure_reason
        ):
            raise ValueError(
                "sample_failure_reason must contain one nonempty string per sample"
            )
        if len(self.ports) < 3:
            raise ValueError("a transition hyperedge needs at least three ports")
        if any(len(port.points) != n_samples for port in self.ports):
            raise ValueError("all ports must use the common transition sample axis")
        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(marginal)):
            raise ValueError("transition curve coordinates must be finite")
        if not np.isfinite(self.total_u_length) or self.total_u_length < 0.0:
            raise ValueError("total_u_length must be finite and nonnegative")
        if not isinstance(self.source_critical_status, CriticalCurveStatus):
            raise ValueError("source_critical_status must be a CriticalCurveStatus")
        if not isinstance(self.controls, TransitionMappingConfig):
            raise ValueError("controls must be a TransitionMappingConfig")
        if interior_maximum_count.shape != (n_samples,):
            raise ValueError("interior_maximum_count must have one value per sample")
        if barrier_margin.shape != (n_samples,):
            raise ValueError("barrier_margin must have one value per sample")
        if len(contact_sample_pairs) and (
            contact_sample_pairs.min() < 0 or contact_sample_pairs.max() >= n_samples
        ):
            raise ValueError("contact_sample_pairs must index the sample axis")
        object.__setattr__(self, "interior_maximum_count", interior_maximum_count)
        object.__setattr__(self, "barrier_margin", barrier_margin)
        object.__setattr__(self, "contact_sample_pairs", contact_sample_pairs)
        object.__setattr__(self, "sample_status", sample_status)
        object.__setattr__(self, "sample_failure_reason", sample_failure_reason)
        for name, values in (
            ("u", u),
            ("marginal_points", marginal),
            ("field_line_identity", identity),
            ("event_zeta_unwrapped", event_zeta),
            ("additivity_residual", residual),
        ):
            object.__setattr__(self, name, values)

    def to_pyvista(self):
        """Return optional PolyData blocks with named port arrays (§17.1)."""
        from . import optional_import

        pyvista = optional_import("pyvista", extra="connectivity")
        blocks = pyvista.MultiBlock()
        for port in self.ports:
            finite = np.all(np.isfinite(port.points), axis=1)
            view = pyvista.PolyData(port.points[finite])
            if np.count_nonzero(finite) > 1:
                ids = np.arange(np.count_nonzero(finite), dtype=np.int64)
                view.lines = np.concatenate(([len(ids)], ids))
            view.point_data["u [logical arc length]"] = self.u[finite]
            view.point_data["action A [length]"] = port.action_values[finite]
            view.point_data["action quadrature error [length]"] = port.quadrature_error[
                finite
            ]
            view.point_data["lifted zeta [rad]"] = port.zeta_unwrapped[finite]
            view.point_data["source vertex id [integer]"] = port.source_vertex_ids[
                finite
            ]
            view.point_data["transition sample status [enum]"] = np.asarray(
                [status.value for status in self.sample_status], dtype=np.int64
            )[finite]
            blocks[port.role] = view
        return blocks


@dataclass(frozen=True)
class _DirectionalTrace:
    status: TransitionStatus
    distance: float
    zeta: float
    extrema_distances: FloatArray
    extrema_curvatures: FloatArray
    extrema_B_minus_b: FloatArray


def _scalar(value) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError("field methods must return one value for scalar coordinates")
    return float(array.reshape(()))


def _logical_point(s: float, theta: float, zeta: float, period: float) -> FloatArray:
    radius = np.sqrt(s)
    return np.array(
        [radius * np.cos(theta), radius * np.sin(theta), np.mod(zeta, period)],
        dtype=float,
    )


def _scan_step(
    field: BoozerFieldLike,
    iota: float,
    period: float,
    config: TransitionMappingConfig,
) -> float:
    step = period / config.samples_per_field_period
    frequency = _mode_frequency(field, iota)
    if frequency > 0.0:
        step = min(
            step,
            2.0 * np.pi / (config.samples_per_wavelength * frequency),
        )
    return float(step)


def _directional_crossing(
    field: BoozerFieldLike,
    *,
    b: float,
    s: float,
    theta_m: float,
    zeta_m: float,
    direction: float,
    iota: float,
    period: float,
    config: TransitionMappingConfig,
) -> _DirectionalTrace:
    """Find the first regular crossing away from a tangent maximum."""

    def coordinates(distance: float) -> tuple[float, float]:
        zeta = zeta_m + direction * distance
        theta = theta_m + iota * (zeta - zeta_m)
        return theta, zeta

    def value(distance: float) -> float:
        theta, zeta = coordinates(distance)
        return _scalar(field.B(s, theta, zeta)) - b

    def derivative(distance: float) -> float:
        theta, zeta = coordinates(distance)
        return direction * _scalar(field.D_B(s, theta, zeta))

    step = _scan_step(field, iota, period, config)
    cell_count = int(np.ceil(config.max_field_periods * period / step))
    extrema_distances: list[float] = []
    extrema_curvatures: list[float] = []
    extrema_values: list[float] = []

    def result(
        status: TransitionStatus, distance: float = np.nan, zeta: float = np.nan
    ) -> _DirectionalTrace:
        return _DirectionalTrace(
            status,
            distance,
            zeta,
            np.asarray(extrema_distances, dtype=np.float64),
            np.asarray(extrema_curvatures, dtype=np.float64),
            np.asarray(extrema_values, dtype=np.float64),
        )

    left = 0.0
    for candidate in step * np.geomspace(1.0e-10, 1.0, 33):
        if value(float(candidate)) < -config.root_atol_B:
            left = float(candidate)
            break
    f_left = value(left)
    d_left = derivative(left)
    for cell_index in range(cell_count):
        right = min((cell_index + 1) * step, config.max_field_periods * period)
        f_right = value(right)
        d_right = derivative(right)
        if not np.all(np.isfinite([f_left, d_left, f_right, d_right])):
            return result(TransitionStatus.UNRESOLVED)

        subdivision = [left]
        if right > config.root_atol_zeta and abs(f_right) <= config.tangent_atol_B:
            theta_right, zeta_right = coordinates(right)
            curvature = _scalar(field.D2_B(s, theta_right, zeta_right))
            if (
                abs(d_right) <= config.tangent_slope_tolerance
                and curvature < -config.D2_tolerance
            ):
                return result(TransitionStatus.MULTIWAY, right, zeta_right)
        if d_left * d_right < 0.0:
            try:
                extremum = brentq(
                    derivative,
                    left,
                    right,
                    xtol=config.root_atol_zeta,
                    rtol=max(config.root_rtol, 4.0 * np.finfo(float).eps),
                )
            except ValueError:
                return result(TransitionStatus.MATCH_FAILURE)
            if extremum > config.root_atol_zeta:
                theta_extremum, zeta_extremum = coordinates(extremum)
                extremum_value = value(extremum)
                curvature = _scalar(field.D2_B(s, theta_extremum, zeta_extremum))
                if (
                    abs(extremum_value) <= config.tangent_atol_B
                    and curvature < -config.D2_tolerance
                ):
                    return result(TransitionStatus.MULTIWAY, extremum, zeta_extremum)
                subdivision.append(extremum)
                extrema_distances.append(float(extremum))
                extrema_curvatures.append(float(curvature))
                extrema_values.append(float(extremum_value))
        subdivision.append(right)

        for first, second in zip(subdivision[:-1], subdivision[1:]):
            first_value = value(first)
            second_value = value(second)
            if first_value < -config.root_atol_B and second_value >= 0.0:
                try:
                    root = brentq(
                        value,
                        first,
                        second,
                        xtol=config.root_atol_zeta,
                        rtol=max(config.root_rtol, 4.0 * np.finfo(float).eps),
                    )
                except ValueError:
                    return result(TransitionStatus.MATCH_FAILURE)
                slope = derivative(root)
                if slope <= config.tangent_slope_tolerance:
                    theta_root, zeta_root = coordinates(root)
                    curvature = _scalar(field.D2_B(s, theta_root, zeta_root))
                    if curvature < -config.D2_tolerance:
                        return result(TransitionStatus.MULTIWAY, root, zeta_root)
                    return result(TransitionStatus.MATCH_FAILURE)
                _, zeta = coordinates(root)
                return result(TransitionStatus.REGULAR, root, zeta)
        left, f_left, d_left = right, f_right, d_right
    return result(TransitionStatus.MAX_PERIODS)


def _action(
    field: BoozerFieldLike,
    *,
    b: float,
    s: float,
    alpha: float,
    sigma: float,
    start_zeta: float,
    distance: float,
    breakpoints: FloatArray,
    order: int,
    root_tolerance: float,
) -> float:
    """Evaluate §4.2 action by fixed Gauss--Legendre rules between extrema."""
    if distance <= 0.0 or not np.isfinite(distance):
        raise ValueError("action interval must be finite and positive")
    breakpoints = np.asarray(breakpoints, dtype=float)
    if breakpoints.ndim != 1 or np.any(~np.isfinite(breakpoints)):
        raise ValueError("action breakpoints must be a finite vector")
    if len(breakpoints) and (
        breakpoints[0] <= 0.0
        or breakpoints[-1] >= distance
        or np.any(np.diff(breakpoints) <= 0.0)
    ):
        raise ValueError("action breakpoints must increase inside the interval")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    angle = 0.25 * np.pi * (nodes + 1.0)
    unit_nodes = 0.5 * (nodes + 1.0)
    iota = _scalar(field.iota(s))
    C = abs(_scalar(field.G(s)) + iota * _scalar(field.I(s)))
    if not np.isfinite(C) or C == 0.0:
        raise ValueError("field-line current factor must be finite and nonzero")
    boundaries = np.concatenate(([0.0], breakpoints, [distance]))
    total = 0.0
    n_intervals = len(boundaries) - 1
    for interval_index, (left, right) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        width = right - left
        if n_intervals == 1:
            path_distance = left + width * np.sin(angle) ** 2
            jacobian = 0.5 * np.pi * width * np.sin(angle) * np.cos(angle)
        elif interval_index == 0:
            path_distance = left + width * unit_nodes**2
            jacobian = width * unit_nodes
        elif interval_index == n_intervals - 1:
            path_distance = right - width * (1.0 - unit_nodes) ** 2
            jacobian = width * (1.0 - unit_nodes)
        else:
            path_distance = left + width * unit_nodes
            jacobian = np.full_like(nodes, 0.5 * width)
        zeta = start_zeta + sigma * path_distance
        theta = alpha + iota * zeta
        B = np.asarray(field.B(s, theta, zeta), dtype=float)
        field_difference = b - B
        if np.any(~np.isfinite(B)) or np.any(B <= 0.0):
            raise ValueError("field values on a transition action interval are invalid")
        if np.min(field_difference) < -root_tolerance:
            raise ValueError("transition action interval left the B<=b well")
        field_difference = np.where(field_difference < 0.0, 0.0, field_difference)
        integrand = C / B * np.sqrt(field_difference / b) * jacobian
        total += float(np.dot(weights, integrand))
    return total


def _clean_breakpoints(
    values: FloatArray, distance: float, tolerance: float
) -> FloatArray:
    """Return strictly increasing interior extrema after scan-cell merging."""
    values = np.sort(np.asarray(values, dtype=float))
    values = values[(values > tolerance) & (values < distance - tolerance)]
    if not len(values):
        return np.empty(0, dtype=np.float64)
    keep = np.concatenate(([True], np.diff(values) > tolerance))
    return values[keep]


def _refined_action(
    field: BoozerFieldLike,
    *,
    b: float,
    s: float,
    alpha: float,
    sigma: float,
    start_zeta: float,
    distance: float,
    breakpoints: FloatArray,
    config: TransitionMappingConfig,
) -> tuple[float, float]:
    """Increase child-action order until tolerance or the explicit cap."""
    order = config.action_quadrature_order
    max_order = (
        max(512, 2 * order)
        if config.max_action_quadrature_order is None
        else config.max_action_quadrature_order
    )
    previous = _action(
        field,
        b=b,
        s=s,
        alpha=alpha,
        sigma=sigma,
        start_zeta=start_zeta,
        distance=distance,
        breakpoints=breakpoints,
        order=order,
        root_tolerance=config.root_atol_B,
    )
    error = np.inf
    while order < max_order:
        order = min(2 * order, max_order)
        value = _action(
            field,
            b=b,
            s=s,
            alpha=alpha,
            sigma=sigma,
            start_zeta=start_zeta,
            distance=distance,
            breakpoints=breakpoints,
            order=order,
            root_tolerance=config.root_atol_B,
        )
        error = abs(value - previous)
        if (
            error
            <= config.action_quadrature_atol
            + config.action_quadrature_rtol * abs(value)
        ):
            return value, error
        previous = value
    return previous, float(error)


def _adaptive_parent_action(
    field: BoozerFieldLike,
    *,
    b: float,
    s: float,
    alpha: float,
    sigma: float,
    start_zeta: float,
    distance: float,
    tangent_distance: float,
    breakpoints: FloatArray,
    config: TransitionMappingConfig,
) -> tuple[float, float]:
    """Independently evaluate ``A_W=A[a,d]`` (DESIGN.md §10.2 step 4)."""
    if not 0.0 < tangent_distance < distance:
        raise ValueError("the marginal point must be interior to the parent well")
    breakpoints = np.asarray(breakpoints, dtype=float)
    if breakpoints.ndim != 1 or np.any(~np.isfinite(breakpoints)):
        raise ValueError("parent action breakpoints must be a finite vector")
    points = _clean_breakpoints(
        np.concatenate((breakpoints, [tangent_distance])),
        distance,
        config.root_atol_zeta,
    )
    if points[0] <= 0.0 or points[-1] >= distance:
        raise ValueError("parent action breakpoints must lie inside the well")
    iota = _scalar(field.iota(s))
    C = abs(_scalar(field.G(s)) + iota * _scalar(field.I(s)))

    def path_integrand(path_distance: float) -> float:
        zeta = start_zeta + sigma * path_distance
        theta = alpha + iota * zeta
        B = _scalar(field.B(s, theta, zeta))
        if not np.isfinite(B) or B <= 0.0 or not np.isfinite(C) or C == 0.0:
            raise ValueError("field values on the parent action interval are invalid")
        difference = b - B
        if difference < -config.root_atol_B:
            raise ValueError("parent action interval left the B<=b well")
        difference = max(difference, 0.0)
        return C / B * np.sqrt(difference / b)

    boundaries = np.concatenate(([0.0], points, [distance]))
    singular = np.array([0.0, tangent_distance, distance])
    n_intervals = len(boundaries) - 1
    value = 0.0
    error = 0.0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        width = right - left
        left_singular = np.min(np.abs(singular - left)) <= config.root_atol_zeta
        right_singular = np.min(np.abs(singular - right)) <= config.root_atol_zeta

        def transformed(unit: float) -> float:
            if left_singular and right_singular:
                angle = 0.5 * np.pi * unit
                path_distance = left + width * np.sin(angle) ** 2
                jacobian = np.pi * width * np.sin(angle) * np.cos(angle)
            elif left_singular:
                path_distance = left + width * unit**2
                jacobian = 2.0 * width * unit
            elif right_singular:
                path_distance = right - width * (1.0 - unit) ** 2
                jacobian = 2.0 * width * (1.0 - unit)
            else:
                path_distance = left + width * unit
                jacobian = width
            return path_integrand(path_distance) * jacobian

        result = quad(
            transformed,
            0.0,
            1.0,
            epsabs=config.action_quadrature_atol / n_intervals,
            epsrel=config.action_quadrature_rtol,
            limit=100,
            full_output=1,
        )
        interval_value, interval_error = result[:2]
        if not np.all(np.isfinite([interval_value, interval_error])):
            raise ValueError("parent action quadrature returned nonfinite data")
        value += float(interval_value)
        error += float(interval_error)
    tolerance = config.action_quadrature_atol + config.action_quadrature_rtol * abs(
        value
    )
    if error > tolerance:
        raise ValueError("parent action quadrature tolerance was not achieved")
    return float(value), float(error)


def _failed_ports(
    marginal: FloatArray,
    source_ids: IntArray,
    n_samples: int,
) -> tuple[TransitionPort, ...]:
    nan = np.full(n_samples, np.nan)
    unknown = np.full(n_samples, -1, dtype=np.int64)
    unavailable = np.full_like(marginal, np.nan)
    return (
        TransitionPort("parent", unavailable, nan, nan, nan, unknown),
        TransitionPort("child_1", unavailable, nan, nan, nan, unknown),
        TransitionPort("child_3", marginal, nan, nan, nan, source_ids),
    )


def _aggregate_status(
    sample_status: tuple[TransitionStatus, ...],
) -> TransitionStatus:
    """Return one conservative curve status without erasing sample outcomes."""
    if all(status is TransitionStatus.REGULAR for status in sample_status):
        return TransitionStatus.REGULAR
    for status in (
        TransitionStatus.MULTIWAY,
        TransitionStatus.UNRESOLVED,
        TransitionStatus.MATCH_FAILURE,
        TransitionStatus.MAX_PERIODS,
        TransitionStatus.TANGENT,
    ):
        if status in sample_status:
            return status
    return TransitionStatus.UNRESOLVED


def _interior_maximum_data(
    trace: _DirectionalTrace, config: TransitionMappingConfig
) -> tuple[int, float]:
    """Count the scan's interior maxima and return their margin to ``b``.

    An extremum recorded by the scan is a barrier when its curvature is
    negative; ``b - B`` at the highest such barrier is the margin, in field
    units, by which that second maximum stays below the marginal height
    (the scan stores ``B - b`` at each extremum, so no ``b`` is needed here).
    Extrema at or beyond the crossing are not interior to the well.

    The count is of the maxima ``_directional_crossing`` *detected*: it takes
    one extremum per scan cell that brackets a sign change of ``dB/dl``, so a
    maximum and minimum inside one cell are both missed. The count therefore
    depends on ``samples_per_field_period`` and ``samples_per_wavelength``
    (§21.3 dimension 5) as well as on the geometry, and a missed barrier that
    is not the highest one changes the count without changing the margin.
    """
    distances = np.asarray(trace.extrema_distances, dtype=float)
    curvatures = np.asarray(trace.extrema_curvatures, dtype=float)
    values = np.asarray(trace.extrema_B_minus_b, dtype=float)
    interior = distances < trace.distance - config.root_atol_zeta
    maxima = interior & (curvatures < -config.D2_tolerance)
    if not np.any(maxima):
        return 0, np.inf
    return int(np.count_nonzero(maxima)), float(np.min(-values[maxima]))


def _between_sample_contacts(
    interior_maximum_count: IntArray,
    sample_status: list[TransitionStatus],
    closed: bool,
) -> IntArray:
    """Bracket nongeneric events that fall between adjacent samples.

    Moving along ``Gamma_max``, the interior-maximum count of the parent well
    changes in two ways, and DESIGN.md §5.4 requires reporting both. A barrier
    rising through ``b`` is a second maximum of the marginal height, and the
    port actions jump across it; a barrier annihilating with its neighboring
    minimum in a fold (``D_parallel^2 B -> 0``) changes the count at a height
    that can be far below ``b``, with no jump in ``A_W``. The recorded
    ``barrier_margin`` separates the two: it approaches zero at an
    equal-height contact and stays finite at a fold.

    Adjacent regular samples whose counts differ therefore bracket one of
    these events, and returning the bracket keeps it explicit instead of
    letting the port actions jump inside a nominally regular hyperedge. A
    count change straddling a non-regular sample is not bracketed: that
    sample's own status already carries the failure. A curve later demoted
    whole -- a duplicate companion component -- keeps no brackets, so every
    row a caller sees still names two regular samples.

    Rows follow the sample order and are not sorted; a closed curve's
    wraparound row is ``(n_samples - 1, 0)``, whose ``u`` values decrease.
    """
    n_samples = len(interior_maximum_count)
    pairs = [(index, index + 1) for index in range(n_samples - 1)]
    if closed and n_samples > 2:
        pairs.append((n_samples - 1, 0))
    brackets = [
        pair
        for pair in pairs
        if sample_status[pair[0]] is TransitionStatus.REGULAR
        and sample_status[pair[1]] is TransitionStatus.REGULAR
        and interior_maximum_count[pair[0]] != interior_maximum_count[pair[1]]
    ]
    return np.asarray(brackets, dtype=np.int64).reshape(-1, 2)


def _curve_status(
    sample_status: tuple[TransitionStatus, ...], n_contacts: int
) -> TransitionStatus:
    """Lift an otherwise regular curve to ``MULTIWAY`` for a stepped-over event.

    A bracket found between two regular samples never outranks a sample-level
    failure: a capped scan stays ``MAX_PERIODS`` and a failed action stays
    ``UNRESOLVED``, because those name why the curve is unusable, while the
    bracket is recorded in ``contact_sample_pairs`` either way.
    """
    status = _aggregate_status(sample_status)
    if status is TransitionStatus.REGULAR and n_contacts:
        return TransitionStatus.MULTIWAY
    return status


def _curve_sample_indices(polyline, max_samples: int | None) -> IntArray:
    """Select deterministic, ordered samples without changing the PL curve."""
    count = len(polyline.vertex_ids)
    if max_samples is None or count <= max_samples:
        return np.arange(count, dtype=np.int64)
    if polyline.closed:
        return np.floor(np.arange(max_samples) * count / max_samples).astype(np.int64)
    return np.rint(np.linspace(0, count - 1, max_samples)).astype(np.int64)


def _map_polyline(
    field: BoozerFieldLike,
    critical: CriticalCurves,
    polyline,
    transition_id: int,
    config: TransitionMappingConfig,
) -> TransitionCurve:
    sample_indices = _curve_sample_indices(polyline, config.max_curve_samples)
    source_ids = np.asarray(polyline.vertex_ids, dtype=np.int64)[sample_indices]
    marginal = np.asarray(critical.points[source_ids], dtype=float)
    u = np.asarray(polyline.u, dtype=float)[sample_indices]
    n_samples = len(source_ids)
    s_values = np.sum(marginal[:, :2] ** 2, axis=1)
    theta_m = np.unwrap(np.arctan2(marginal[:, 1], marginal[:, 0]))
    zeta_m = np.unwrap(marginal[:, 2], period=critical.period)
    iota_values = np.asarray(field.iota(s_values), dtype=float)
    if iota_values.shape == ():
        iota_values = np.full(n_samples, float(iota_values))
    identity = np.column_stack((s_values, theta_m - iota_values * zeta_m))
    event_zeta = np.full((n_samples, 3), np.nan)

    point_kinds = critical.point_kind[source_ids]
    segment_kinds = critical.segment_kind[np.asarray(polyline.segment_ids, dtype=int)]
    if not np.all(segment_kinds == CriticalKind.GAMMA_MAX.value):
        sample_status = (TransitionStatus.UNRESOLVED,) * n_samples
        sample_failure_reason = ("segment_classification",) * n_samples
        return TransitionCurve(
            transition_id,
            critical.b,
            u,
            float(polyline.total_length),
            _failed_ports(marginal, source_ids, n_samples),
            marginal,
            identity,
            event_zeta,
            np.full(n_samples, np.nan),
            TransitionStatus.UNRESOLVED,
            critical.status,
            config,
            sample_status,
            sample_failure_reason,
        )

    a_points = np.full_like(marginal, np.nan)
    parent_action = np.full(n_samples, np.nan)
    child_1_action = np.full(n_samples, np.nan)
    child_3_action = np.full(n_samples, np.nan)
    parent_error = np.full(n_samples, np.nan)
    child_1_error = np.full(n_samples, np.nan)
    child_3_error = np.full(n_samples, np.nan)
    sample_status = [TransitionStatus.UNRESOLVED] * n_samples
    sample_failure_reason = ["source_classification"] * n_samples
    interior_maximum_count = np.full(n_samples, -1, dtype=np.int64)
    barrier_margin = np.full(n_samples, np.nan)
    event_zeta[:, 1] = zeta_m
    for index in range(n_samples):
        if point_kinds[index] != CriticalKind.GAMMA_MAX.value:
            continue
        s = float(s_values[index])
        theta = float(theta_m[index])
        zeta = float(zeta_m[index])
        iota = float(iota_values[index])
        C = _scalar(field.G(s)) + iota * _scalar(field.I(s))
        B_m = _scalar(field.B(s, theta, zeta))
        D_m = _scalar(field.D_B(s, theta, zeta))
        D2_m = _scalar(field.D2_B(s, theta, zeta))
        if (
            not np.all(np.isfinite([C, B_m, D_m, D2_m]))
            or C == 0.0
            or abs(B_m - critical.b) > config.root_atol_B
            or abs(D_m) > config.tangent_atol_B
            or D2_m >= -config.D2_tolerance
        ):
            sample_failure_reason[index] = "source_residual"
            continue
        sigma = float(np.sign(C))
        backward = _directional_crossing(
            field,
            b=critical.b,
            s=s,
            theta_m=theta,
            zeta_m=zeta,
            direction=-sigma,
            iota=iota,
            period=critical.period,
            config=config,
        )
        forward = _directional_crossing(
            field,
            b=critical.b,
            s=s,
            theta_m=theta,
            zeta_m=zeta,
            direction=sigma,
            iota=iota,
            period=critical.period,
            config=config,
        )
        if TransitionStatus.MULTIWAY in (backward.status, forward.status):
            sample_status[index] = TransitionStatus.MULTIWAY
            sample_failure_reason[index] = "multiway_crossing"
            continue
        if backward.status is not TransitionStatus.REGULAR:
            sample_status[index] = backward.status
            sample_failure_reason[index] = f"backward_{backward.status.name.lower()}"
            continue
        if forward.status is not TransitionStatus.REGULAR:
            sample_status[index] = forward.status
            sample_failure_reason[index] = f"forward_{forward.status.name.lower()}"
            continue

        backward_maxima, backward_margin = _interior_maximum_data(backward, config)
        forward_maxima, forward_margin = _interior_maximum_data(forward, config)
        interior_maximum_count[index] = backward_maxima + forward_maxima
        barrier_margin[index] = min(backward_margin, forward_margin)

        zeta_a = float(backward.zeta)
        zeta_d = float(forward.zeta)
        alpha = float(identity[index, 1])
        theta_a = alpha + iota * zeta_a
        a_points[index] = _logical_point(s, theta_a, zeta_a, critical.period)
        event_zeta[index] = (zeta_a, zeta, zeta_d)
        child_1_breakpoints = _clean_breakpoints(
            backward.distance - backward.extrema_distances[::-1],
            backward.distance,
            config.root_atol_zeta,
        )
        child_3_breakpoints = _clean_breakpoints(
            forward.extrema_distances,
            forward.distance,
            config.root_atol_zeta,
        )
        parent_breakpoints = _clean_breakpoints(
            np.concatenate(
                (
                    child_1_breakpoints,
                    [backward.distance],
                    backward.distance + child_3_breakpoints,
                )
            ),
            backward.distance + forward.distance,
            config.root_atol_zeta,
        )
        try:
            child_1_action[index], child_1_error[index] = _refined_action(
                field,
                b=critical.b,
                s=s,
                alpha=alpha,
                sigma=sigma,
                start_zeta=zeta_a,
                distance=backward.distance,
                breakpoints=child_1_breakpoints,
                config=config,
            )
            child_3_action[index], child_3_error[index] = _refined_action(
                field,
                b=critical.b,
                s=s,
                alpha=alpha,
                sigma=sigma,
                start_zeta=zeta,
                distance=forward.distance,
                breakpoints=child_3_breakpoints,
                config=config,
            )
            parent_action[index], parent_error[index] = _adaptive_parent_action(
                field,
                b=critical.b,
                s=s,
                alpha=alpha,
                sigma=sigma,
                start_zeta=zeta_a,
                distance=backward.distance + forward.distance,
                tangent_distance=backward.distance,
                breakpoints=parent_breakpoints,
                config=config,
            )
        except (ValueError, FloatingPointError) as error:
            sample_failure_reason[index] = f"action: {error}"
            continue
        sample_status[index] = TransitionStatus.REGULAR
        sample_failure_reason[index] = "regular"

    residual = parent_action - child_1_action - child_3_action
    quadrature_failure = (
        (
            parent_error
            > config.action_quadrature_atol
            + config.action_quadrature_rtol * np.abs(parent_action)
        )
        | (
            child_1_error
            > config.action_quadrature_atol
            + config.action_quadrature_rtol * np.abs(child_1_action)
        )
        | (
            child_3_error
            > config.action_quadrature_atol
            + config.action_quadrature_rtol * np.abs(child_3_action)
        )
    )
    for index in np.flatnonzero(quadrature_failure):
        sample_status[index] = TransitionStatus.UNRESOLVED
        sample_failure_reason[index] = "action_quadrature_error"
    tolerance = config.additivity_atol + config.additivity_rtol * np.abs(parent_action)
    additivity_failure = (np.abs(residual) > tolerance) & ~quadrature_failure
    for index in np.flatnonzero(additivity_failure):
        sample_status[index] = TransitionStatus.UNRESOLVED
        sample_failure_reason[index] = "additivity"
    contact_sample_pairs = _between_sample_contacts(
        interior_maximum_count, sample_status, bool(polyline.closed)
    )
    sample_status_tuple = tuple(sample_status)
    sample_failure_reason_tuple = tuple(sample_failure_reason)
    status = _curve_status(sample_status_tuple, len(contact_sample_pairs))
    unknown = np.full(n_samples, -1, dtype=np.int64)
    ports = (
        TransitionPort(
            "parent",
            a_points,
            event_zeta[:, 0],
            parent_action,
            parent_error,
            unknown,
        ),
        TransitionPort(
            "child_1",
            a_points,
            event_zeta[:, 0],
            child_1_action,
            child_1_error,
            unknown,
        ),
        TransitionPort(
            "child_3",
            marginal,
            event_zeta[:, 1],
            child_3_action,
            child_3_error,
            source_ids,
        ),
    )
    return TransitionCurve(
        transition_id,
        critical.b,
        u,
        float(polyline.total_length),
        ports,
        marginal,
        identity,
        event_zeta,
        residual,
        status,
        critical.status,
        config,
        sample_status_tuple,
        sample_failure_reason_tuple,
        interior_maximum_count,
        barrier_margin,
        contact_sample_pairs,
    )


def map_transitions(
    field: BoozerFieldLike,
    critical: CriticalCurves,
    config: TransitionMappingConfig | None = None,
) -> tuple[TransitionCurve, ...]:
    """Construct companion curves and matched ports (DESIGN.md §10.2).

    One result is returned for every ``GAMMA_MAX`` polyline. Backward and
    forward searches originate from each marginal point itself, so all three
    ports share the same lifted field-line identity and common ``u`` index.
    A second equal-height maximum encountered before a regular crossing is a
    ``MULTIWAY`` event, never an arbitrarily decomposed binary transition.
    A nongeneric event that falls *between* two adjacent samples is detected
    from the change in each parent well's interior-maximum count and reported
    the same way (§5.4), with the bracketing samples recorded in
    ``contact_sample_pairs`` and the height of the barrier involved in
    ``barrier_margin``.
    """
    cfg = TransitionMappingConfig() if config is None else config
    maxima = [
        polyline
        for polyline in critical.polylines
        if polyline.kind is CriticalKind.GAMMA_MAX
    ]
    transitions = [
        _map_polyline(field, critical, polyline, index, cfg)
        for index, polyline in enumerate(maxima)
    ]
    duplicate_components: set[int] = set()
    for first_index, first in enumerate(transitions):
        first_regular = np.array(
            [status is TransitionStatus.REGULAR for status in first.sample_status]
        )
        if np.count_nonzero(first_regular) < 2:
            continue
        first_T = next(port for port in first.ports if port.role == "parent")
        for second_index in range(first_index + 1, len(transitions)):
            second = transitions[second_index]
            second_regular = np.array(
                [status is TransitionStatus.REGULAR for status in second.sample_status]
            )
            if np.count_nonzero(second_regular) < 2:
                continue
            second_T = next(port for port in second.ports if port.role == "parent")
            ds = np.abs(
                first.field_line_identity[:, np.newaxis, 0]
                - second.field_line_identity[np.newaxis, :, 0]
            )
            dalpha = (
                first.field_line_identity[:, np.newaxis, 1]
                - second.field_line_identity[np.newaxis, :, 1]
            )
            dalpha -= 2.0 * np.pi * np.round(dalpha / (2.0 * np.pi))
            dzeta = (
                first_T.zeta_unwrapped[:, np.newaxis]
                - second_T.zeta_unwrapped[np.newaxis, :]
            )
            matches = (
                (ds <= cfg.field_identity_tolerance)
                & (np.abs(dalpha) <= cfg.field_identity_tolerance)
                & (np.abs(dzeta) <= cfg.root_atol_zeta)
                & first_regular[:, np.newaxis]
                & second_regular[np.newaxis, :]
            )
            if np.count_nonzero(np.any(matches, axis=1)) >= 2:
                duplicate_components.update((first_index, second_index))
    for index in duplicate_components:
        transition = transitions[index]
        source_ids = next(
            port.source_vertex_ids
            for port in transition.ports
            if port.role == "child_3"
        )
        transitions[index] = replace(
            transition,
            ports=_failed_ports(
                transition.marginal_points, source_ids, len(transition.u)
            ),
            additivity_residual=np.full(len(transition.u), np.nan),
            status=TransitionStatus.MULTIWAY,
            sample_status=(TransitionStatus.MULTIWAY,) * len(transition.u),
            sample_failure_reason=("duplicate_companion",) * len(transition.u),
            # No sample is regular any more, so no bracket between two of them
            # survives: this curve is a duplicate companion, not a curve with
            # subdivision points for milestone 10.
            contact_sample_pairs=np.empty((0, 2), dtype=np.int64),
            barrier_margin=np.full(len(transition.u), np.nan),
            interior_maximum_count=np.full(len(transition.u), -1, dtype=np.int64),
        )
    return tuple(transitions)
