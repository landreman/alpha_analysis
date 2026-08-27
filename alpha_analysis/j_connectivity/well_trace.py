"""First-return well tracing and bounce quadrature (DESIGN.md §9)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, cumulative_trapezoid, quad
from scipy.optimize import brentq

from .field import BoozerFieldLike
from .types import FloatArray, IntArray, TraceStatus, WellTrace


@dataclass(frozen=True)
class WellTraceConfig:
    """Numerical controls for a regular well trace (DESIGN.md §§9.2–9.4).

    ``root_atol_B`` and ``tangent_atol_B`` are absolute magnetic-field
    tolerances. ``root_atol_zeta`` is in radians and ``root_rtol`` is
    dimensionless. The scan takes at least ``samples_per_field_period``
    samples per field period and, when the
    field exposes Fourier mode numbers, at least
    ``samples_per_wavelength`` samples for the fastest retained mode along the
    field line. ``extrema_tolerance`` is in field units per radian and
    ``second_derivative_tolerance`` is in field units per radian squared.
    Quadrature absolute errors have the units of ``A`` and ``K`` respectively;
    ``quadrature_rtol`` is dimensionless.
    """

    samples_per_field_period: int = 64
    max_field_periods: int = 128
    root_rtol: float = 1.0e-12
    root_atol_B: float = 1.0e-10
    extrema_tolerance: float = 1.0e-10
    second_derivative_tolerance: float = 1.0e-10
    quadrature_rtol: float = 1.0e-10
    quadrature_atol: float = 1.0e-10
    samples_per_wavelength: int = 24
    root_atol_zeta: float = 1.0e-12
    tangent_atol_B: float = 1.0e-9
    itinerary_quantization: float = 1.0e-8

    def __post_init__(self) -> None:
        positive = (
            "root_rtol",
            "root_atol_B",
            "extrema_tolerance",
            "second_derivative_tolerance",
            "quadrature_rtol",
            "quadrature_atol",
            "root_atol_zeta",
            "tangent_atol_B",
            "itinerary_quantization",
        )
        for name in positive:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.samples_per_field_period < 4:
            raise ValueError("samples_per_field_period must be at least four")
        if self.samples_per_wavelength < 4:
            raise ValueError("samples_per_wavelength must be at least four")
        if self.max_field_periods < 1:
            raise ValueError("max_field_periods must be at least one")


@dataclass(frozen=True)
class WellProfile:
    """Sampled §17.4 diagnostic data for one regular well.

    Angles are radians. ``action_integrand`` and ``bounce_time_integrand``
    are per radian of unwrapped ``|zeta|``. Their cumulative counterparts use
    the endpoint-regularized coordinate and have the length units of ``A``
    and ``K`` from DESIGN.md §4.2. ``theta_unwrapped`` starts from the
    canonical reduced incoming angle, so its path is defined up to one global
    integer multiple of ``2*pi``; relative winding is authoritative.
    """

    zeta_unwrapped: FloatArray
    theta_unwrapped: FloatArray
    B: FloatArray
    action_integrand: FloatArray
    bounce_time_integrand: FloatArray
    cumulative_A: FloatArray
    cumulative_K: FloatArray


class _QuadratureDomainError(RuntimeError):
    pass


def _scalar(value) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError("field profiles must return one value for scalar coordinates")
    return float(array.reshape(-1)[0])


def _reduced(value: float, period: float, tolerance: float) -> float:
    result = float(np.mod(value, period))
    if min(result, period - result) <= 10.0 * tolerance:
        return 0.0
    return result


def _mode_frequency(field: BoozerFieldLike, iota: float) -> float:
    m = getattr(field, "m", None)
    n = getattr(field, "n", None)
    if m is None or n is None:
        m = getattr(field, "xm", None)
        n = getattr(field, "xn", None)
    if m is None or n is None:
        return 0.0
    m_array = np.asarray(m, dtype=float)
    n_array = np.asarray(n, dtype=float)
    if m_array.size == 0 or n_array.shape != m_array.shape:
        return 0.0
    return float(np.max(np.abs(m_array * iota - n_array)))


def _failed_trace(
    status: TraceStatus,
    *,
    b: float,
    q_in: FloatArray,
    B_residual_in: float,
    field_period_count: int = 0,
    extrema_zeta: list[float] | None = None,
    extrema_B: list[float] | None = None,
    extrema_kind: list[int] | None = None,
    tangent_zeta: list[float] | None = None,
    tangent_B: list[float] | None = None,
) -> WellTrace:
    zeta = np.asarray(extrema_zeta if extrema_zeta is not None else [], dtype=float)
    values = np.asarray(extrema_B if extrema_B is not None else [], dtype=float)
    kinds = np.asarray(extrema_kind if extrema_kind is not None else [], dtype=np.int64)
    tangent_positions = np.asarray(
        tangent_zeta if tangent_zeta is not None else [], dtype=float
    )
    tangent_values = np.asarray(tangent_B if tangent_B is not None else [], dtype=float)
    return WellTrace(
        status=status,
        b=float(b),
        q_in=np.asarray(q_in, dtype=float),
        q_out_reduced=np.full(3, np.nan),
        zeta_out_unwrapped=np.nan,
        field_period_count=int(field_period_count),
        action_length=np.nan,
        bounce_time_length=np.nan,
        extrema_zeta_unwrapped=zeta,
        extrema_B=values,
        extrema_kind=kinds,
        tangent_zeta_unwrapped=tangent_positions,
        tangent_B=tangent_values,
        n_internal_maxima=int(np.count_nonzero(kinds == -1)),
        itinerary_hash=np.uint64(0),
        B_residual_in=float(B_residual_in),
        B_residual_out=np.nan,
        quadrature_error_A=np.nan,
        quadrature_error_K=np.nan,
    )


def _itinerary_hash(
    *,
    q_out: FloatArray,
    u_out: float,
    period: float,
    extrema_zeta: FloatArray,
    extrema_B: FloatArray,
    extrema_kind: IntArray,
    zeta_in: float,
    sigma: float,
    b: float,
    quantization: float,
) -> np.uint64:
    """Return a stable quick-comparison hash; unquantized arrays remain primary."""
    relative_extrema = sigma * (extrema_zeta - zeta_in) / period
    floating = [u_out / period, q_out[1] / (2.0 * np.pi), q_out[2] / period]
    for relative, value in zip(relative_extrema, extrema_B):
        floating.extend((relative, np.mod(relative, 1.0), (value - b) / abs(b)))
    quantized = np.rint(np.asarray(floating) / quantization).astype("<i8")
    kinds = np.asarray(extrema_kind, dtype="<i8")
    payload = quantized.tobytes() + kinds.tobytes()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"well-it").digest()
    return np.uint64(int.from_bytes(digest, byteorder="little", signed=False))


def _regularized_integrands(
    field: BoozerFieldLike,
    *,
    b: float,
    s: float,
    theta_in: float,
    zeta_in: float,
    iota: float,
    sigma: float,
    C_abs: float,
    u_out: float,
    slope_in: float,
    slope_out: float,
    root_tolerance: float,
):
    """Build the sine-squared endpoint map for the §4.2 ``A`` and ``K``."""

    limit_in = np.pi * C_abs * np.sqrt(u_out / (b * -slope_in))
    limit_out = np.pi * C_abs * np.sqrt(u_out / (b * slope_out))

    def pair(x: float) -> tuple[float, float]:
        if x <= 0.0:
            return 0.0, float(limit_in)
        if x >= 1.0:
            return 0.0, float(limit_out)
        sine = np.sin(0.5 * np.pi * x)
        u = u_out * sine * sine
        jacobian = 0.5 * np.pi * u_out * np.sin(np.pi * x)
        zeta = zeta_in + sigma * u
        theta = theta_in + iota * (zeta - zeta_in)
        B_value = _scalar(field.B(s, theta, zeta))
        if not np.isfinite(B_value) or B_value <= 0.0:
            raise _QuadratureDomainError("B must remain finite and positive")
        radicand = 1.0 - B_value / b
        if radicand <= 0.0:
            if radicand < -root_tolerance / abs(b):
                raise _QuadratureDomainError("trace left the B < b well")
            if x < 0.5:
                endpoint_distance = u
                slope = -slope_in
            else:
                endpoint_distance = u_out - u
                slope = slope_out
            endpoint_tolerance = max(
                100.0 * np.finfo(float).eps * u_out,
                root_tolerance / slope,
            )
            if endpoint_distance > endpoint_tolerance:
                raise _QuadratureDomainError(
                    "nonpositive radicand away from an endpoint"
                )
            radicand = (slope / b) * endpoint_distance
        if radicand <= 0.0 or not np.isfinite(radicand):
            raise _QuadratureDomainError("invalid endpoint radicand")
        root = np.sqrt(radicand)
        common = C_abs * jacobian / B_value
        return float(common * root), float(common / root)

    return pair


def trace_regular_well(
    field: BoozerFieldLike,
    b: float,
    q_in,
    config: WellTraceConfig | None = None,
) -> WellTrace:
    """Trace the first outgoing well return along ``+B`` (DESIGN.md §9).

    ``q_in=(s, theta, zeta)`` uses normalized toroidal flux and radians and
    must be a regular physical incoming point: ``B=b`` and
    ``(B/(G+iota*I)) D_parallel B < 0``. The scan is carried in an unwrapped
    nonnegative distance ``u`` with ``zeta=zeta_in+sign(G+iota I) u``. Thus
    every returned ordinary root is the first physical outgoing crossing,
    even when ``G+iota I`` is negative. ``action_length`` and
    ``bounce_time_length`` are the half-bounce lengths ``A`` and ``K`` of
    DESIGN.md §4.2. All capped, tangent, root, and quadrature failures keep
    an explicit non-regular status and never receive zero action.
    """
    cfg = WellTraceConfig() if config is None else config
    q = np.asarray(q_in, dtype=float)
    if q.shape != (3,) or not np.all(np.isfinite(q)):
        raise ValueError("q_in must be a finite (s, theta, zeta) array")
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError("b must be finite and positive")
    s, theta_in, zeta_in = map(float, q)
    period = 2.0 * np.pi / int(field.nfp)
    iota = _scalar(field.iota(s))
    C = _scalar(field.G(s)) + iota * _scalar(field.I(s))
    q_reduced = np.array(
        [
            s,
            _reduced(theta_in, 2.0 * np.pi, cfg.root_atol_zeta),
            _reduced(zeta_in, period, cfg.root_atol_zeta),
        ]
    )
    B_in = _scalar(field.B(s, theta_in, zeta_in))
    residual_in = B_in - b
    if not np.isfinite(C) or abs(C) <= np.finfo(float).tiny:
        return _failed_trace(
            TraceStatus.DEGENERATE,
            b=b,
            q_in=q_reduced,
            B_residual_in=residual_in,
        )
    sigma = float(np.sign(C))

    def coordinates(u):
        u_array = np.asarray(u, dtype=float)
        zeta = zeta_in + sigma * u_array
        theta = theta_in + iota * (zeta - zeta_in)
        return theta, zeta

    def F(u: float) -> float:
        theta, zeta = coordinates(u)
        return _scalar(field.B(s, theta, zeta)) - b

    def derivative(u: float) -> float:
        theta, zeta = coordinates(u)
        return sigma * _scalar(field.D_B(s, theta, zeta))

    if not np.isfinite(residual_in) or abs(residual_in) > cfg.root_atol_B:
        return _failed_trace(
            TraceStatus.ROOT_FAILURE,
            b=b,
            q_in=q_reduced,
            B_residual_in=residual_in,
        )
    slope_in = derivative(0.0)
    if not np.isfinite(slope_in):
        return _failed_trace(
            TraceStatus.DEGENERATE,
            b=b,
            q_in=q_reduced,
            B_residual_in=residual_in,
        )
    if abs(slope_in) <= cfg.extrema_tolerance:
        return _failed_trace(
            TraceStatus.TANGENT_OR_TRANSITION,
            b=b,
            q_in=q_reduced,
            B_residual_in=residual_in,
            tangent_zeta=[zeta_in],
            tangent_B=[B_in],
        )
    if slope_in > 0.0:
        return _failed_trace(
            TraceStatus.NO_WELL,
            b=b,
            q_in=q_reduced,
            B_residual_in=residual_in,
        )

    maximum_frequency = _mode_frequency(field, iota)
    step = period / cfg.samples_per_field_period
    if maximum_frequency > 0.0:
        step = min(
            step,
            2.0 * np.pi / (cfg.samples_per_wavelength * maximum_frequency),
        )
    steps_per_period = int(np.ceil(period / step))

    extrema_zeta: list[float] = []
    extrema_values: list[float] = []
    extrema_kinds: list[int] = []
    u_out = None
    slope_out = np.nan
    completed_periods = 0

    for period_index in range(cfg.max_field_periods):
        start = period_index * period
        u_grid = np.linspace(start, start + period, steps_per_period + 1)
        theta_grid, zeta_grid = coordinates(u_grid)
        B_grid = np.asarray(field.B(s, theta_grid, zeta_grid), dtype=float)
        D_grid = sigma * np.asarray(field.D_B(s, theta_grid, zeta_grid), dtype=float)
        if B_grid.shape != u_grid.shape or D_grid.shape != u_grid.shape:
            raise ValueError("field B and D_B must preserve broadcast coordinate shape")
        if not np.all(np.isfinite(B_grid)) or not np.all(np.isfinite(D_grid)):
            return _failed_trace(
                TraceStatus.ROOT_FAILURE,
                b=b,
                q_in=q_reduced,
                B_residual_in=residual_in,
                field_period_count=period_index,
                extrema_zeta=extrema_zeta,
                extrema_B=extrema_values,
                extrema_kind=extrema_kinds,
            )
        F_grid = B_grid - b
        for index in range(steps_per_period):
            left = float(u_grid[index])
            right = float(u_grid[index + 1])
            f_left = float(F_grid[index])
            f_right = float(F_grid[index + 1])
            d_left = float(D_grid[index])
            d_right = float(D_grid[index + 1])

            crossing = None
            if f_left < 0.0 <= f_right:
                try:
                    crossing = brentq(
                        F,
                        left,
                        right,
                        xtol=cfg.root_atol_zeta,
                        rtol=max(cfg.root_rtol, 4.0 * np.finfo(float).eps),
                    )
                except ValueError:
                    return _failed_trace(
                        TraceStatus.ROOT_FAILURE,
                        b=b,
                        q_in=q_reduced,
                        B_residual_in=residual_in,
                        field_period_count=period_index,
                        extrema_zeta=extrema_zeta,
                        extrema_B=extrema_values,
                        extrema_kind=extrema_kinds,
                    )

            extremum = None
            if d_left == 0.0 and left > cfg.root_atol_zeta:
                extremum = left
            elif d_left * d_right < 0.0:
                try:
                    extremum = brentq(
                        derivative,
                        left,
                        right,
                        xtol=cfg.root_atol_zeta,
                        rtol=max(cfg.root_rtol, 4.0 * np.finfo(float).eps),
                    )
                except ValueError:
                    return _failed_trace(
                        TraceStatus.ROOT_FAILURE,
                        b=b,
                        q_in=q_reduced,
                        B_residual_in=residual_in,
                        field_period_count=period_index,
                        extrema_zeta=extrema_zeta,
                        extrema_B=extrema_values,
                        extrema_kind=extrema_kinds,
                    )
            elif d_right == 0.0:
                extremum = right

            if extremum is not None and (
                crossing is None or extremum < crossing - cfg.root_atol_zeta
            ):
                if (
                    not extrema_zeta
                    or abs(sigma * (extrema_zeta[-1] - zeta_in) - extremum)
                    > 10.0 * cfg.root_atol_zeta
                ):
                    theta_ext, zeta_ext = coordinates(extremum)
                    B_ext = _scalar(field.B(s, theta_ext, zeta_ext))
                    second = _scalar(field.D2_B(s, theta_ext, zeta_ext))
                    if abs(second) <= cfg.second_derivative_tolerance:
                        return _failed_trace(
                            TraceStatus.DEGENERATE,
                            b=b,
                            q_in=q_reduced,
                            B_residual_in=residual_in,
                            field_period_count=period_index,
                            extrema_zeta=extrema_zeta,
                            extrema_B=extrema_values,
                            extrema_kind=extrema_kinds,
                        )
                    kind = 1 if second > 0.0 else -1
                    if abs(B_ext - b) <= cfg.tangent_atol_B:
                        return _failed_trace(
                            TraceStatus.TANGENT_OR_TRANSITION,
                            b=b,
                            q_in=q_reduced,
                            B_residual_in=residual_in,
                            field_period_count=period_index,
                            extrema_zeta=extrema_zeta + [float(zeta_ext)],
                            extrema_B=extrema_values + [B_ext],
                            extrema_kind=extrema_kinds + [kind],
                            tangent_zeta=[float(zeta_ext)],
                            tangent_B=[B_ext],
                        )
                    if B_ext > b + cfg.root_atol_B:
                        return _failed_trace(
                            TraceStatus.ROOT_FAILURE,
                            b=b,
                            q_in=q_reduced,
                            B_residual_in=residual_in,
                            field_period_count=period_index,
                            extrema_zeta=extrema_zeta,
                            extrema_B=extrema_values,
                            extrema_kind=extrema_kinds,
                        )
                    extrema_zeta.append(float(zeta_ext))
                    extrema_values.append(B_ext)
                    extrema_kinds.append(kind)

            if crossing is not None:
                slope_candidate = derivative(crossing)
                if slope_candidate <= cfg.extrema_tolerance:
                    return _failed_trace(
                        TraceStatus.TANGENT_OR_TRANSITION,
                        b=b,
                        q_in=q_reduced,
                        B_residual_in=residual_in,
                        field_period_count=period_index,
                        extrema_zeta=extrema_zeta,
                        extrema_B=extrema_values,
                        extrema_kind=extrema_kinds,
                        tangent_zeta=[float(coordinates(crossing)[1])],
                        tangent_B=[b + F(crossing)],
                    )
                u_out = float(crossing)
                slope_out = float(slope_candidate)
                break
        if u_out is not None:
            break
        completed_periods = period_index + 1

    if u_out is None:
        return _failed_trace(
            TraceStatus.MAX_PERIODS,
            b=b,
            q_in=q_reduced,
            B_residual_in=residual_in,
            field_period_count=completed_periods,
            extrema_zeta=extrema_zeta,
            extrema_B=extrema_values,
            extrema_kind=extrema_kinds,
        )

    theta_out, zeta_out = coordinates(u_out)
    residual_out = _scalar(field.B(s, theta_out, zeta_out)) - b
    if abs(residual_out) > cfg.root_atol_B:
        return _failed_trace(
            TraceStatus.ROOT_FAILURE,
            b=b,
            q_in=q_reduced,
            B_residual_in=residual_in,
            field_period_count=int(np.floor(u_out / period)),
            extrema_zeta=extrema_zeta,
            extrema_B=extrema_values,
            extrema_kind=extrema_kinds,
        )

    pair = _regularized_integrands(
        field,
        b=b,
        s=s,
        theta_in=theta_in,
        zeta_in=zeta_in,
        iota=iota,
        sigma=sigma,
        C_abs=abs(C),
        u_out=u_out,
        slope_in=slope_in,
        slope_out=slope_out,
        root_tolerance=cfg.root_atol_B,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            action, error_A = quad(
                lambda x: pair(x)[0],
                0.0,
                1.0,
                epsabs=cfg.quadrature_atol,
                epsrel=cfg.quadrature_rtol,
                limit=200,
            )
            bounce_time, error_K = quad(
                lambda x: pair(x)[1],
                0.0,
                1.0,
                epsabs=cfg.quadrature_atol,
                epsrel=cfg.quadrature_rtol,
                limit=200,
            )
    except (IntegrationWarning, _QuadratureDomainError, ValueError, FloatingPointError):
        return _failed_trace(
            TraceStatus.QUADRATURE_FAILURE,
            b=b,
            q_in=q_reduced,
            B_residual_in=residual_in,
            field_period_count=int(np.floor(u_out / period)),
            extrema_zeta=extrema_zeta,
            extrema_B=extrema_values,
            extrema_kind=extrema_kinds,
        )
    if not np.all(np.isfinite([action, bounce_time, error_A, error_K])):
        return _failed_trace(
            TraceStatus.QUADRATURE_FAILURE,
            b=b,
            q_in=q_reduced,
            B_residual_in=residual_in,
            field_period_count=int(np.floor(u_out / period)),
            extrema_zeta=extrema_zeta,
            extrema_B=extrema_values,
            extrema_kind=extrema_kinds,
        )

    q_out = np.array(
        [
            s,
            _reduced(float(theta_out), 2.0 * np.pi, cfg.root_atol_zeta),
            _reduced(float(zeta_out), period, cfg.root_atol_zeta),
        ]
    )
    extrema_zeta_array = np.asarray(extrema_zeta, dtype=float)
    extrema_B_array = np.asarray(extrema_values, dtype=float)
    extrema_kind_array = np.asarray(extrema_kinds, dtype=np.int64)
    period_count = int(
        np.floor(u_out / period + max(10.0 * cfg.root_atol_zeta / period, 1.0e-11))
    )
    itinerary = _itinerary_hash(
        q_out=q_out,
        u_out=u_out,
        period=period,
        extrema_zeta=extrema_zeta_array,
        extrema_B=extrema_B_array,
        extrema_kind=extrema_kind_array,
        zeta_in=zeta_in,
        sigma=sigma,
        b=b,
        quantization=cfg.itinerary_quantization,
    )
    return WellTrace(
        status=TraceStatus.REGULAR,
        b=float(b),
        q_in=q_reduced,
        q_out_reduced=q_out,
        zeta_out_unwrapped=float(zeta_out),
        field_period_count=period_count,
        action_length=float(action),
        bounce_time_length=float(bounce_time),
        extrema_zeta_unwrapped=extrema_zeta_array,
        extrema_B=extrema_B_array,
        extrema_kind=extrema_kind_array,
        tangent_zeta_unwrapped=np.empty(0, dtype=float),
        tangent_B=np.empty(0, dtype=float),
        n_internal_maxima=int(np.count_nonzero(extrema_kind_array == -1)),
        itinerary_hash=itinerary,
        B_residual_in=float(residual_in),
        B_residual_out=float(residual_out),
        quadrature_error_A=float(error_A),
        quadrature_error_K=float(error_K),
    )


def sample_well_profile(
    field: BoozerFieldLike,
    trace: WellTrace,
    *,
    n_samples: int = 513,
) -> WellProfile:
    """Sample a regular trace for the diagnostic required by §17.4.

    The cumulative curves are diagnostic trapezoidal approximations in the
    endpoint-regularized coordinate. They are not substituted for the
    authoritative adaptive values stored on ``trace``.
    """
    if trace.status is not TraceStatus.REGULAR:
        raise ValueError("a well profile requires TraceStatus.REGULAR")
    if n_samples < 3:
        raise ValueError("n_samples must be at least three")
    s, theta_in, zeta_in = map(float, trace.q_in)
    period = 2.0 * np.pi / int(field.nfp)
    zeta_out = float(trace.zeta_out_unwrapped)
    C = _scalar(field.G(s)) + _scalar(field.iota(s)) * _scalar(field.I(s))
    sigma = float(np.sign(C))
    if sigma > 0.0:
        fractional_span = np.mod(trace.q_out_reduced[2] - zeta_in, period)
    else:
        fractional_span = np.mod(zeta_in - trace.q_out_reduced[2], period)
    if min(fractional_span, period - fractional_span) <= 1.0e-11:
        fractional_span = 0.0
    u_out = trace.field_period_count * period + fractional_span
    zeta_in = zeta_out - sigma * u_out
    iota = _scalar(field.iota(s))
    slope_in = sigma * _scalar(field.D_B(s, theta_in, zeta_in))
    theta_out = theta_in + iota * (zeta_out - zeta_in)
    slope_out = sigma * _scalar(field.D_B(s, theta_out, zeta_out))
    pair = _regularized_integrands(
        field,
        b=trace.b,
        s=s,
        theta_in=theta_in,
        zeta_in=zeta_in,
        iota=iota,
        sigma=sigma,
        C_abs=abs(C),
        u_out=u_out,
        slope_in=slope_in,
        slope_out=slope_out,
        root_tolerance=max(abs(trace.B_residual_in), abs(trace.B_residual_out), 1e-12),
    )
    x = np.linspace(0.0, 1.0, n_samples)
    sine = np.sin(0.5 * np.pi * x)
    u = u_out * sine * sine
    zeta = zeta_in + sigma * u
    theta = theta_in + iota * (zeta - zeta_in)
    B_values = np.asarray(field.B(s, theta, zeta), dtype=float)
    radicand = np.maximum(0.0, 1.0 - B_values / trace.b)
    action_integrand = abs(C) / B_values * np.sqrt(radicand)
    with np.errstate(divide="ignore"):
        bounce_integrand = abs(C) / (B_values * np.sqrt(radicand))
    regularized = np.asarray([pair(float(value)) for value in x], dtype=float)
    cumulative_A = np.concatenate(([0.0], cumulative_trapezoid(regularized[:, 0], x)))
    cumulative_K = np.concatenate(([0.0], cumulative_trapezoid(regularized[:, 1], x)))
    return WellProfile(
        zeta_unwrapped=zeta,
        theta_unwrapped=theta,
        B=B_values,
        action_integrand=action_integrand,
        bounce_time_integrand=bounce_integrand,
        cumulative_A=cumulative_A,
        cumulative_K=cumulative_K,
    )
