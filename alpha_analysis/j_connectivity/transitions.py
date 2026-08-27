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

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq

from .critical_curves import CriticalCurveStatus, CriticalCurves, CriticalKind
from .field import BoozerFieldLike
from .types import FloatArray, IntArray, TransitionStatus
from .well_trace import _mode_frequency


@dataclass(frozen=True)
class TransitionMappingConfig:
    """Numerical controls for DESIGN.md §10.2 transition tracing.

    Field and derivative tolerances use the field's units and radians.
    ``action_quadrature_order`` is the Gauss--Legendre order for each of the
    direct ``A_W``, ``A_1``, and ``A_3`` evaluations. Increasing it is the
    transition-action refinement dimension required by §§21.3 and 23.
    ``additivity_atol`` has action-length units and ``additivity_rtol`` is
    dimensionless.
    """

    samples_per_field_period: int = 64
    samples_per_wavelength: int = 24
    max_field_periods: int = 16
    root_atol_B: float = 1.0e-10
    root_atol_zeta: float = 1.0e-12
    root_rtol: float = 1.0e-12
    tangent_atol_B: float = 1.0e-9
    tangent_slope_tolerance: float = 1.0e-8
    D2_tolerance: float = 1.0e-10
    action_quadrature_order: int = 32
    additivity_atol: float = 1.0e-8
    additivity_rtol: float = 1.0e-7

    def __post_init__(self) -> None:
        for name in (
            "samples_per_field_period",
            "samples_per_wavelength",
            "max_field_periods",
            "action_quadrature_order",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in (
            "root_atol_B",
            "root_atol_zeta",
            "root_rtol",
            "tangent_atol_B",
            "tangent_slope_tolerance",
            "D2_tolerance",
            "additivity_atol",
            "additivity_rtol",
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
    source_vertex_ids: IntArray
    sheet_id: int = -1

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        zeta = np.asarray(self.zeta_unwrapped, dtype=np.float64)
        action = np.asarray(self.action_values, dtype=np.float64)
        vertex_ids = np.asarray(self.source_vertex_ids, dtype=np.int64)
        if self.role not in {"parent", "child_1", "child_3", "generic"}:
            raise ValueError("unknown transition port role")
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("transition port points must have shape (n, 3)")
        n_samples = len(points)
        if any(values.shape != (n_samples,) for values in (zeta, action, vertex_ids)):
            raise ValueError("transition port arrays must share one sample axis")
        finite_rows = np.all(np.isfinite(points), axis=1)
        nan_rows = np.all(np.isnan(points), axis=1)
        if np.any(~(finite_rows | nan_rows)):
            raise ValueError("each transition port point must be finite or all-NaN")
        for name, values in (
            ("points", points),
            ("zeta_unwrapped", zeta),
            ("action_values", action),
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
    nearest-neighbor association structurally impossible.
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

    def __post_init__(self) -> None:
        u = np.asarray(self.u, dtype=np.float64)
        marginal = np.asarray(self.marginal_points, dtype=np.float64)
        identity = np.asarray(self.field_line_identity, dtype=np.float64)
        event_zeta = np.asarray(self.event_zeta_unwrapped, dtype=np.float64)
        residual = np.asarray(self.additivity_residual, dtype=np.float64)
        n_samples = len(u)
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
        if len(self.ports) < 3:
            raise ValueError("a transition hyperedge needs at least three ports")
        if any(len(port.points) != n_samples for port in self.ports):
            raise ValueError("all ports must use the common transition sample axis")
        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(marginal)):
            raise ValueError("transition curve coordinates must be finite")
        if not np.isfinite(self.total_u_length) or self.total_u_length < 0.0:
            raise ValueError("total_u_length must be finite and nonnegative")
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
            view.point_data["lifted zeta [rad]"] = port.zeta_unwrapped[finite]
            view.point_data["source vertex id [integer]"] = port.source_vertex_ids[
                finite
            ]
            blocks[port.role] = view
        return blocks


@dataclass(frozen=True)
class _DirectionalTrace:
    status: TransitionStatus
    distance: float
    zeta: float


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
    left = 0.0
    f_left = value(left)
    d_left = derivative(left)
    for cell_index in range(cell_count):
        right = min((cell_index + 1) * step, config.max_field_periods * period)
        f_right = value(right)
        d_right = derivative(right)
        if not np.all(np.isfinite([f_left, d_left, f_right, d_right])):
            return _DirectionalTrace(TransitionStatus.UNRESOLVED, np.nan, np.nan)

        subdivision = [left]
        if right > config.root_atol_zeta and abs(f_right) <= config.tangent_atol_B:
            theta_right, zeta_right = coordinates(right)
            curvature = _scalar(field.D2_B(s, theta_right, zeta_right))
            if (
                abs(d_right) <= config.tangent_slope_tolerance
                and curvature < -config.D2_tolerance
            ):
                return _DirectionalTrace(TransitionStatus.MULTIWAY, right, zeta_right)
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
                return _DirectionalTrace(TransitionStatus.MATCH_FAILURE, np.nan, np.nan)
            if extremum > config.root_atol_zeta:
                theta_extremum, zeta_extremum = coordinates(extremum)
                extremum_value = value(extremum)
                curvature = _scalar(field.D2_B(s, theta_extremum, zeta_extremum))
                if (
                    abs(extremum_value) <= config.tangent_atol_B
                    and curvature < -config.D2_tolerance
                ):
                    return _DirectionalTrace(
                        TransitionStatus.MULTIWAY, extremum, zeta_extremum
                    )
                subdivision.append(extremum)
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
                    return _DirectionalTrace(
                        TransitionStatus.MATCH_FAILURE, np.nan, np.nan
                    )
                slope = derivative(root)
                if slope <= config.tangent_slope_tolerance:
                    theta_root, zeta_root = coordinates(root)
                    curvature = _scalar(field.D2_B(s, theta_root, zeta_root))
                    if curvature < -config.D2_tolerance:
                        return _DirectionalTrace(
                            TransitionStatus.MULTIWAY, root, zeta_root
                        )
                    return _DirectionalTrace(
                        TransitionStatus.MATCH_FAILURE, np.nan, np.nan
                    )
                _, zeta = coordinates(root)
                return _DirectionalTrace(TransitionStatus.REGULAR, root, zeta)
        left, f_left, d_left = right, f_right, d_right
    return _DirectionalTrace(TransitionStatus.MATCH_FAILURE, np.nan, np.nan)


def _action(
    field: BoozerFieldLike,
    *,
    b: float,
    s: float,
    alpha: float,
    sigma: float,
    start_zeta: float,
    distance: float,
    order: int,
    root_tolerance: float,
) -> float:
    """Evaluate the §4.2 half-bounce action by fixed Gauss--Legendre rule."""
    if distance <= 0.0 or not np.isfinite(distance):
        raise ValueError("action interval must be finite and positive")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    angle = 0.25 * np.pi * (nodes + 1.0)
    path_distance = distance * np.sin(angle) ** 2
    jacobian = 0.5 * np.pi * distance * np.sin(angle) * np.cos(angle)
    zeta = start_zeta + sigma * path_distance
    iota = _scalar(field.iota(s))
    theta = alpha + iota * zeta
    B = np.asarray(field.B(s, theta, zeta), dtype=float)
    C = abs(_scalar(field.G(s)) + iota * _scalar(field.I(s)))
    field_difference = b - B
    if np.any(~np.isfinite(B)) or np.any(B <= 0.0) or not np.isfinite(C) or C == 0.0:
        raise ValueError("field values on a transition action interval are invalid")
    if np.min(field_difference) < -root_tolerance:
        raise ValueError("transition action interval left the B<=b well")
    field_difference = np.where(field_difference < 0.0, 0.0, field_difference)
    integrand = C / B * np.sqrt(field_difference / b) * jacobian
    return float(np.dot(weights, integrand))


def _failed_ports(
    marginal: FloatArray,
    source_ids: IntArray,
    n_samples: int,
) -> tuple[TransitionPort, ...]:
    nan = np.full(n_samples, np.nan)
    unknown = np.full(n_samples, -1, dtype=np.int64)
    unavailable = np.full_like(marginal, np.nan)
    return (
        TransitionPort("parent", unavailable, nan, nan, unknown),
        TransitionPort("child_1", unavailable, nan, nan, unknown),
        TransitionPort("child_3", marginal, nan, nan, source_ids),
    )


def _map_polyline(
    field: BoozerFieldLike,
    critical: CriticalCurves,
    polyline,
    transition_id: int,
    config: TransitionMappingConfig,
) -> TransitionCurve:
    source_ids = np.asarray(polyline.vertex_ids, dtype=np.int64)
    marginal = np.asarray(critical.points[source_ids], dtype=float)
    u = np.asarray(polyline.u, dtype=float)
    n_samples = len(source_ids)
    s_values = np.sum(marginal[:, :2] ** 2, axis=1)
    theta_m = np.unwrap(np.arctan2(marginal[:, 1], marginal[:, 0]))
    zeta_m = np.unwrap(marginal[:, 2], period=critical.period)
    iota_values = np.asarray(field.iota(s_values), dtype=float)
    if iota_values.shape == ():
        iota_values = np.full(n_samples, float(iota_values))
    identity = np.column_stack((s_values, theta_m - iota_values * zeta_m))
    event_zeta = np.full((n_samples, 3), np.nan)

    if critical.status is not CriticalCurveStatus.REGULAR:
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
        )

    a_points = np.empty_like(marginal)
    parent_action = np.full(n_samples, np.nan)
    child_1_action = np.full(n_samples, np.nan)
    child_3_action = np.full(n_samples, np.nan)
    status = TransitionStatus.REGULAR
    for index in range(n_samples):
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
            status = TransitionStatus.UNRESOLVED
            break
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
            status = TransitionStatus.MULTIWAY
            break
        if backward.status is not TransitionStatus.REGULAR:
            status = backward.status
            break
        if forward.status is not TransitionStatus.REGULAR:
            status = forward.status
            break

        zeta_a = float(backward.zeta)
        zeta_d = float(forward.zeta)
        alpha = float(identity[index, 1])
        theta_a = alpha + iota * zeta_a
        a_points[index] = _logical_point(s, theta_a, zeta_a, critical.period)
        event_zeta[index] = (zeta_a, zeta, zeta_d)
        try:
            child_1_action[index] = _action(
                field,
                b=critical.b,
                s=s,
                alpha=alpha,
                sigma=sigma,
                start_zeta=zeta_a,
                distance=backward.distance,
                order=config.action_quadrature_order,
                root_tolerance=config.root_atol_B,
            )
            child_3_action[index] = _action(
                field,
                b=critical.b,
                s=s,
                alpha=alpha,
                sigma=sigma,
                start_zeta=zeta,
                distance=forward.distance,
                order=config.action_quadrature_order,
                root_tolerance=config.root_atol_B,
            )
            parent_action[index] = _action(
                field,
                b=critical.b,
                s=s,
                alpha=alpha,
                sigma=sigma,
                start_zeta=zeta_a,
                distance=backward.distance,
                order=2 * config.action_quadrature_order,
                root_tolerance=config.root_atol_B,
            )
            parent_action[index] += _action(
                field,
                b=critical.b,
                s=s,
                alpha=alpha,
                sigma=sigma,
                start_zeta=zeta,
                distance=forward.distance,
                order=2 * config.action_quadrature_order,
                root_tolerance=config.root_atol_B,
            )
        except (ValueError, FloatingPointError):
            status = TransitionStatus.UNRESOLVED
            break

    if status is not TransitionStatus.REGULAR:
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
            status,
        )

    residual = parent_action - child_1_action - child_3_action
    tolerance = config.additivity_atol + config.additivity_rtol * np.abs(parent_action)
    if np.any(np.abs(residual) > tolerance):
        status = TransitionStatus.UNRESOLVED
    unknown = np.full(n_samples, -1, dtype=np.int64)
    ports = (
        TransitionPort("parent", a_points, event_zeta[:, 0], parent_action, unknown),
        TransitionPort("child_1", a_points, event_zeta[:, 0], child_1_action, unknown),
        TransitionPort(
            "child_3", marginal, event_zeta[:, 1], child_3_action, source_ids
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
    """
    cfg = TransitionMappingConfig() if config is None else config
    maxima = [
        polyline
        for polyline in critical.polylines
        if polyline.kind is CriticalKind.GAMMA_MAX
    ]
    return tuple(
        _map_polyline(field, critical, polyline, index, cfg)
        for index, polyline in enumerate(maxima)
    )
