"""Reusable field-line scans and ordinary bounce integrals (DESIGN.md §9).

The catalogue is a local, pitch-independent scan. Its root certificate applies
only to the recorded lifted window; an open window is never a passing proof.
Fourier coefficient envelopes bound every possible hidden B=b barrier inside a
scan cell. No field-representation error is enclosed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .field import BoozerFieldLike
from .population import LinewiseTrappingMasks
from .types import TraceStatus
from .well_trace import _regularized_integrands, _scalar, _scan_step


@dataclass(frozen=True)
class ForwardScanConfig:
    """R1 scan controls; angles are radians and B tolerances use field units."""

    samples_per_period: int = 64
    samples_per_wavelength: int = 24
    max_periods: int = 128
    max_cell_subdivisions: int = 12
    root_atol_B: float = 1e-10
    root_atol_zeta: float = 1e-12
    tangent_atol_B: float = 1e-9
    second_derivative_tolerance: float = 1e-10
    quadrature_atol: float = 1e-9
    quadrature_rtol: float = 1e-7
    max_quadrature_order: int = 256

    def __post_init__(self) -> None:
        if self.samples_per_period < 4 or self.samples_per_wavelength < 4:
            raise ValueError("scan sampling must be at least four")
        if self.max_periods < 1 or self.max_cell_subdivisions < 0:
            raise ValueError(
                "scan limits must be nonnegative, with at least one period"
            )
        if self.max_quadrature_order < 32:
            raise ValueError("max_quadrature_order must be at least 32")
        for name in (
            "root_atol_B",
            "root_atol_zeta",
            "tangent_atol_B",
            "second_derivative_tolerance",
            "quadrature_atol",
            "quadrature_rtol",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ScanExtremum:
    """A derivative-bracketed extremum on the physical lifted line (§9.1)."""

    u: float
    zeta: float
    B: float
    kind: int  # -1 maximum, +1 minimum, 0 degenerate/unknown


@dataclass(frozen=True)
class OrdinaryWell:
    """Certified first-entry/first-exit root pair within a scanned window.

    ``u`` increases along +B; lifted ``zeta`` may decrease when C<0. B and b
    use field units. The endpoints define A and K of DESIGN.md §4.2.
    """

    b: float
    u_in: float
    u_out: float
    zeta_in: float
    zeta_out: float
    q_in: np.ndarray
    q_out: np.ndarray
    extrema_u: np.ndarray


@dataclass(frozen=True)
class PitchQuery:
    """Ordinary wells and explicit limitations for one b within scan coverage."""

    b: float
    wells: tuple[OrdinaryWell, ...]
    root_complete: bool
    open_left: bool
    open_right: bool
    passing_certified: bool
    unknown_cells: tuple[tuple[float, float], ...]
    reason: str | None
    scanned_periods: int
    status: TraceStatus
    tangent_candidates: tuple[float, ...] = ()


@dataclass(frozen=True)
class BounceIntegral:
    """Half-bounce A/K lengths and numerical *estimates*, never field bounds."""

    status: TraceStatus
    A: float = np.nan
    K: float = np.nan
    error_A: float = np.nan
    error_K: float = np.nan
    method: str = "unresolved"
    reason: str | None = None
    error_scope: str = "estimate"


def _fourier_coefficients(field: BoozerFieldLike, s: float):
    """Return line Fourier coefficients, or None when no envelope is available."""
    if hasattr(field, "cosine_coefficients") and hasattr(field, "m"):
        cosine = np.polynomial.polynomial.polyval(
            s, np.asarray(field.cosine_coefficients, dtype=float).T
        )
        sine = np.polynomial.polynomial.polyval(
            s, np.asarray(field.sine_coefficients, dtype=float).T
        )
        m, n = np.asarray(field.m, dtype=float), np.asarray(field.n, dtype=float)
    elif hasattr(field, "bmnc") and hasattr(field, "xm"):
        cosine = np.asarray(field.bmnc(s), dtype=float).reshape(-1)
        sine = np.asarray(field.bmns(s), dtype=float).reshape(-1)
        m, n = np.asarray(field.xm, dtype=float), np.asarray(field.xn, dtype=float)
    else:
        return None
    if not (
        cosine.shape == sine.shape == m.shape == n.shape
        and np.all(np.isfinite(cosine))
        and np.all(np.isfinite(sine))
    ):
        return None
    return cosine, sine, m, n


class ForwardLineCatalogue:
    """Resume a lifted +B scan and query many conserved bounce fields (§9.1).

    The cache identity is this object plus its field, s, lifted starting angles,
    direction and controls. Reuse the object only for the same field instance;
    ``extend_to`` appends windows and never reuses data for a changed field.
    """

    def __init__(
        self,
        field: BoozerFieldLike,
        s: float,
        theta0: float,
        zeta0: float,
        config: ForwardScanConfig | None = None,
    ) -> None:
        self.field = field
        self.s, self.theta0, self.zeta0 = float(s), float(theta0), float(zeta0)
        self.config = ForwardScanConfig() if config is None else config
        if not np.all(np.isfinite([self.s, self.theta0, self.zeta0])):
            raise ValueError("line coordinates must be finite")
        self.iota = _scalar(field.iota(self.s))
        self.C = _scalar(field.G(self.s)) + self.iota * _scalar(field.I(self.s))
        if not np.isfinite(self.C) or self.C == 0:
            raise ValueError("G+iota*I must be finite and nonzero")
        self.sigma = float(np.sign(self.C))
        self.period = 2 * np.pi / field.nfp
        step = _scan_step(
            field,
            self.iota,
            self.period,
            self.config.samples_per_period,
            self.config.samples_per_wavelength,
        )
        self.steps_per_period = int(np.ceil(self.period / step))
        self.scanned_periods = 0
        self.u = np.empty(0)
        self.B_samples = np.empty(0)
        self.D_samples = np.empty(0)
        self.extrema: tuple[ScanExtremum, ...] = ()
        self.extrema_unverified_cells: tuple[tuple[float, float], ...] = ()
        coefficients = _fourier_coefficients(field, self.s)
        self._curvature_bound = None
        self._third_bound = None
        self._line_upper = None
        if coefficients is not None:
            cosine, sine, m, n = coefficients
            frequency = m * self.iota - n
            amplitude = np.hypot(cosine, sine)
            # Allow for final floating-point contraction of the Fourier sum.
            inflation = 1 + 64 * np.finfo(float).eps * len(amplitude)
            self._curvature_bound = float(np.sum(amplitude * frequency**2) * inflation)
            self._third_bound = float(
                np.sum(amplitude * np.abs(frequency) ** 3) * inflation
            )
            phase = m * self.theta0 - n * self.zeta0
            # Near-rational frequencies still vary on a sufficiently long lift.
            constant = frequency == 0.0
            center = np.sum(
                cosine[constant] * np.cos(phase[constant])
                + sine[constant] * np.sin(phase[constant])
            )
            self._line_upper = float(center + np.sum(amplitude[~constant]) * inflation)

    @property
    def sample_count(self) -> int:
        return len(self.u)

    @property
    def cache_key(self) -> tuple:
        """Runtime key including field identity, lift, controls and coverage (§9.1)."""
        return (
            id(self.field),
            self.s,
            self.theta0,
            self.zeta0,
            self.sigma,
            self.config,
            self.scanned_periods,
        )

    def coordinates(self, u):
        u_array = np.asarray(u, dtype=float)
        zeta = self.zeta0 + self.sigma * u_array
        theta = self.theta0 + self.iota * (zeta - self.zeta0)
        return theta, zeta

    def _B(self, u):
        theta, zeta = self.coordinates(u)
        return np.asarray(self.field.B(self.s, theta, zeta), dtype=float)

    def _D(self, u):
        theta, zeta = self.coordinates(u)
        return self.sigma * np.asarray(self.field.D_B(self.s, theta, zeta), dtype=float)

    def extend_to(self, periods: int) -> None:
        """Append field-period windows up to the requested total coverage."""
        if periods < self.scanned_periods or periods > self.config.max_periods:
            raise ValueError("requested coverage is outside the configured scan cap")
        if periods == self.scanned_periods:
            return
        new_u = np.linspace(
            self.scanned_periods * self.period,
            periods * self.period,
            (periods - self.scanned_periods) * self.steps_per_period + 1,
        )
        new_B = self._B(new_u)
        new_D = self._D(new_u)
        if new_B.shape != new_u.shape or new_D.shape != new_u.shape:
            raise ValueError("field B and D_B must preserve coordinate shape")
        if not np.all(np.isfinite(new_B)) or not np.all(np.isfinite(new_D)):
            raise ValueError("nonfinite field scan cannot be cached")
        if self.scanned_periods:
            new_u, new_B, new_D = new_u[1:], new_B[1:], new_D[1:]
        prior_count = len(self.u)
        self.u = np.concatenate((self.u, new_u))
        self.B_samples = np.concatenate((self.B_samples, new_B))
        self.D_samples = np.concatenate((self.D_samples, new_D))
        self.scanned_periods = periods
        self._scan_extrema(max(0, prior_count - 1))

    def _scan_extrema(self, first_cell: int) -> None:
        """Append bracketed extrema and expose cells lacking a completeness proof."""
        extrema = list(self.extrema)
        unverified = list(self.extrema_unverified_cells)
        if self._third_bound is not None:
            theta, zeta = self.coordinates(self.u[first_cell:])
            second_grid = np.asarray(self.field.D2_B(self.s, theta, zeta), dtype=float)
        else:
            second_grid = None
        for i in range(first_cell, len(self.u) - 1):
            dl, dr = self.D_samples[i : i + 2]
            h = self.u[i + 1] - self.u[i]
            if self._third_bound is None or second_grid is None:
                unverified.append((float(self.u[i]), float(self.u[i + 1])))
            else:
                M3 = self._third_bound
                allowance = M3 * h * h / 8
                no_root = min(dl, dr) - allowance > 0 or max(dl, dr) + allowance < 0
                sl, sr = second_grid[i - first_cell : i - first_cell + 2]
                monotone = min(sl, sr) - M3 * h > 0 or max(sl, sr) + M3 * h < 0
                if not no_root and not (dl * dr < 0 and monotone):
                    unverified.append((float(self.u[i]), float(self.u[i + 1])))
            if dl * dr < 0:
                root = brentq(
                    lambda x: float(self._D(x)),
                    self.u[i],
                    self.u[i + 1],
                    xtol=self.config.root_atol_zeta,
                )
                second = self.sigma**2 * _scalar(
                    self.field.D2_B(self.s, *self.coordinates(root))
                )
                degenerate = (
                    not np.isfinite(second)
                    or abs(second) <= self.config.second_derivative_tolerance
                )
                if degenerate:
                    unverified.append((float(self.u[i]), float(self.u[i + 1])))
                extrema.append(
                    ScanExtremum(
                        float(root),
                        float(self.coordinates(root)[1]),
                        float(self._B(root)),
                        0 if degenerate else 1 if second > 0 else -1,
                    )
                )
        for j in range(max(1, first_cell), len(self.u) - 1):
            if (
                self.D_samples[j] == 0
                and self.D_samples[j - 1] * self.D_samples[j + 1] < 0
            ):
                root = float(self.u[j])
                second = _scalar(self.field.D2_B(self.s, *self.coordinates(root)))
                degenerate = (
                    not np.isfinite(second)
                    or abs(second) <= self.config.second_derivative_tolerance
                )
                if degenerate:
                    unverified.append((float(self.u[j - 1]), float(self.u[j + 1])))
                extrema.append(
                    ScanExtremum(
                        root,
                        float(self.coordinates(root)[1]),
                        float(self._B(root)),
                        0 if degenerate else 1 if second > 0 else -1,
                    )
                )
        extrema.sort(key=lambda item: item.u)
        self.extrema = tuple(
            item
            for index, item in enumerate(extrema)
            if index == 0
            or abs(item.u - extrema[index - 1].u) > 4 * self.config.root_atol_zeta
        )
        self.extrema_unverified_cells = tuple(unverified)

    def _cell_roots(self, left, right, Bl, Br, Dl, Dr, depth, roots, unknown):
        """Certify all B=b roots in a cell using a Fourier curvature envelope."""
        b = self._query_b
        M = self._curvature_bound
        if M is None:
            unknown.append((left, right))
            return
        h = right - left
        allowance = M * h * h / 8 + 32 * np.finfo(float).eps * max(
            abs(Bl), abs(Br), abs(b)
        )
        if max(Bl, Br) + allowance < b:
            return
        if min(Bl, Br) - allowance > b:
            return
        fl, fr = Bl - b, Br - b
        monotone_up = min(Dl, Dr) - M * h > 0
        monotone_down = max(Dl, Dr) + M * h < 0
        rounding = 32 * np.finfo(float).eps * max(abs(Bl), abs(Br), abs(b), 1.0)
        if (monotone_up or monotone_down) and (
            fl * fr < 0 or abs(fl) <= rounding or abs(fr) <= rounding
        ):
            if fl * fr < 0:
                root = brentq(
                    lambda x: float(self._B(x)) - b,
                    left,
                    right,
                    xtol=self.config.root_atol_zeta,
                )
            elif abs(fl) <= rounding:
                root = left
            elif abs(fr) <= rounding:
                root = right
            if abs(float(self._B(root)) - b) > self.config.root_atol_B:
                unknown.append((left, right))
            else:
                roots.append((float(root), -1 if monotone_down else 1))
            return
        if depth >= self.config.max_cell_subdivisions:
            unknown.append((left, right))
            return
        mid = (left + right) / 2
        Bm, Dm = float(self._B(mid)), float(self._D(mid))
        if not np.all(np.isfinite([Bm, Dm])):
            unknown.append((left, right))
            return
        self._cell_roots(left, mid, Bl, Bm, Dl, Dm, depth + 1, roots, unknown)
        self._cell_roots(mid, right, Bm, Br, Dm, Dr, depth + 1, roots, unknown)

    def query(self, b: float) -> PitchQuery:
        """Find maximal ordinary wells over the stored lifted window (§9.2).

        ``root_complete`` is scoped to this finite window. It does not assert
        global line coverage or prove that no later well exists.
        """
        if not np.isfinite(b) or b <= 0:
            raise ValueError("b must be finite and positive")
        if not self.scanned_periods:
            raise ValueError("extend the catalogue before querying")
        if (
            self._line_upper is not None
            and self._line_upper < b - self.config.root_atol_B
        ):
            return PitchQuery(
                b,
                (),
                True,
                False,
                False,
                True,
                (),
                None,
                self.scanned_periods,
                TraceStatus.NO_WELL,
            )
        self._query_b = float(b)
        roots: list[tuple[float, int]] = []
        unknown: list[tuple[float, float]] = []
        for i in range(len(self.u) - 1):
            self._cell_roots(
                self.u[i],
                self.u[i + 1],
                self.B_samples[i],
                self.B_samples[i + 1],
                self.D_samples[i],
                self.D_samples[i + 1],
                0,
                roots,
                unknown,
            )
        tangent_candidates = tuple(
            item.u
            for item in self.extrema
            if abs(item.B - b) <= self.config.tangent_atol_B
        )
        roots.sort()
        unique_roots = []
        for root in roots:
            if (
                unique_roots
                and abs(root[0] - unique_roots[-1][0]) <= 4 * self.config.root_atol_zeta
            ):
                if root[1] != unique_roots[-1][1]:
                    unknown.append((root[0], root[0]))
                continue
            unique_roots.append(root)
        events = [(u, "unknown", 0) for span in unknown for u in [span[0]]]
        events += [(u, "root", sign) for u, sign in unique_roots]
        events.sort(key=lambda value: value[0])
        start_B = self.B_samples[0]
        start_D = self.D_samples[0]
        start_rounding = 32 * np.finfo(float).eps * max(abs(start_B), abs(b), 1.0)
        open_left = bool(start_B < b - start_rounding)
        entry = None
        if abs(start_B - b) <= start_rounding and start_D < 0:
            entry = 0.0
        elif open_left:
            entry = None  # left endpoint is outside the scan, not a root
        wells = []
        censored = open_left
        for u, kind, sign in events:
            if kind == "unknown":
                entry = None
                censored = True
            elif sign < 0:
                entry = u
                censored = False
            elif entry is not None:
                if u > entry + self.config.root_atol_zeta and not any(
                    lo < u and hi > entry for lo, hi in unknown
                ):
                    theta_in, zeta_in = self.coordinates(entry)
                    theta_out, zeta_out = self.coordinates(u)
                    extrema_u = np.array(
                        [item.u for item in self.extrema if entry < item.u < u],
                        dtype=float,
                    )
                    wells.append(
                        OrdinaryWell(
                            float(b),
                            float(entry),
                            float(u),
                            float(zeta_in),
                            float(zeta_out),
                            np.array([self.s, theta_in, zeta_in]),
                            np.array([self.s, theta_out, zeta_out]),
                            extrema_u,
                        )
                    )
                entry = None
                censored = False
            else:
                censored = True
        end_B = self.B_samples[-1]
        end_rounding = 32 * np.finfo(float).eps * max(abs(end_B), abs(b), 1.0)
        open_right = bool(end_B < b - end_rounding or entry is not None)
        reason = None
        if tangent_candidates:
            reason = "B=b tangent or transition limit in scanned window"
        elif unknown:
            reason = "unresolved possible B=b barrier in scan cells"
        elif open_left or open_right or censored:
            reason = "well crosses a finite scan-window boundary"
        return PitchQuery(
            float(b),
            tuple(wells),
            not (unknown or open_left or open_right or censored),
            open_left,
            open_right,
            False,
            tuple(unknown),
            reason,
            self.scanned_periods,
            (
                TraceStatus.TANGENT_OR_TRANSITION
                if tangent_candidates
                else (
                    TraceStatus.ROOT_FAILURE
                    if unknown
                    else (
                        TraceStatus.MAX_PERIODS
                        if open_left or open_right or censored
                        else TraceStatus.REGULAR
                    )
                )
            ),
            tangent_candidates,
        )


class CatalogueLinewisePredicate:
    """Source-independent trapped masks from centered finite scans (§12.1).

    A cell is definitely trapped only when its sampled point lies inside a
    certified incoming/first-outgoing root pair. Every unresolved allowed point
    remains possibly trapped with a reason. Cached lines reuse scans over b.
    """

    def __init__(self, field: BoozerFieldLike, config: ForwardScanConfig | None = None):
        self.field = field
        self.config = ForwardScanConfig() if config is None else config
        self._lines: dict[tuple[float, float, float], ForwardLineCatalogue] = {}

    def __call__(self, s, theta, zeta, b: float, B=None) -> LinewiseTrappingMasks:
        if not np.isfinite(b) or b <= 0:
            raise ValueError("b must be finite and positive")
        s_array, theta_array, zeta_array = np.broadcast_arrays(
            np.asarray(s, dtype=float),
            np.asarray(theta, dtype=float),
            np.asarray(zeta, dtype=float),
        )
        B_array = (
            np.asarray(self.field.B(s_array, theta_array, zeta_array), dtype=float)
            if B is None
            else np.broadcast_to(np.asarray(B, dtype=float), s_array.shape)
        )
        if not np.all(np.isfinite(B_array)):
            raise ValueError("nonfinite population-node B cannot be classified")
        definite = np.zeros(s_array.shape, dtype=bool)
        possible = B_array < b
        reasons: set[str] = set()
        for index in np.ndindex(s_array.shape):
            if not possible[index]:
                continue
            key = (
                float(s_array[index]),
                float(theta_array[index]),
                float(zeta_array[index]),
            )
            try:
                catalogue = self._lines.get(key)
                if catalogue is None:
                    iota = _scalar(self.field.iota(key[0]))
                    C = _scalar(self.field.G(key[0])) + iota * _scalar(
                        self.field.I(key[0])
                    )
                    sigma = float(np.sign(C))
                    period = 2 * np.pi / self.field.nfp
                    zeta0 = key[2] - sigma * self.config.max_periods * period / 2
                    theta0 = key[1] + iota * (zeta0 - key[2])
                    catalogue = ForwardLineCatalogue(
                        self.field, key[0], theta0, zeta0, self.config
                    )
                    catalogue.extend_to(self.config.max_periods)
                    self._lines[key] = catalogue
                result = catalogue.query(b)
            except (ValueError, FloatingPointError, RuntimeError) as exc:
                reasons.add(f"line scan failed: {type(exc).__name__}: {exc}")
                continue
            point_u = self.config.max_periods * catalogue.period / 2
            if any(w.u_in < point_u < w.u_out for w in result.wells):
                definite[index] = True
            elif result.passing_certified:
                possible[index] = False
            else:
                reasons.add(result.reason or "finite scan lacks a two-sided root pair")
        return LinewiseTrappingMasks(
            definite,
            possible,
            tuple(sorted(reasons)),
        )


def _batch_order(catalogue: ForwardLineCatalogue, wells, segments, order: int):
    """Evaluate one composite Gauss order across every well/segment together."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    owner = np.array([segment[0] for segment in segments], dtype=int)
    left = np.array([segment[1] for segment in segments])
    right = np.array([segment[2] for segment in segments])
    x = (left[:, None] + right[:, None]) / 2 + (right - left)[:, None] * nodes / 2
    x_weight = (right - left)[:, None] * weights / 2
    U = np.array([w.u_out - w.u_in for w in wells])[owner, None]
    entry = np.array([w.u_in for w in wells])[owner, None]
    b = np.array([w.b for w in wells])[owner, None]
    sine = np.sin(np.pi * x / 2)
    u = entry + U * sine**2
    jac = U * np.pi * np.sin(np.pi * x) / 2
    theta, zeta = catalogue.coordinates(u)
    B = np.asarray(catalogue.field.B(catalogue.s, theta, zeta), dtype=float)
    if B.shape != x.shape or not np.all(np.isfinite(B)) or np.any(B <= 0):
        raise ValueError("nonfinite or nonpositive field in bounce quadrature")
    difference = b - B
    distance_in = u - entry
    distance_out = entry + U - u
    near_in = (x < 0.5) & (distance_in < catalogue.period / catalogue.steps_per_period)
    near_out = (x >= 0.5) & (
        distance_out < catalogue.period / catalogue.steps_per_period
    )
    cancellation = abs(difference) < np.cbrt(np.finfo(float).eps) * np.maximum(1.0, b)
    for mask, from_exit in (
        (near_in & cancellation, False),
        (near_out & cancellation, True),
    ):
        if not np.any(mask):
            continue
        start = u[mask] if from_exit else entry.repeat(order, axis=1)[mask]
        end = (entry + U).repeat(order, axis=1)[mask] if from_exit else u[mask]
        gl_x, gl_w = np.polynomial.legendre.leggauss(16)
        points = (start[:, None] + end[:, None]) / 2 + (end - start)[:, None] * gl_x / 2
        point_theta, point_zeta = catalogue.coordinates(points)
        derivative = catalogue.sigma * np.asarray(
            catalogue.field.D_B(catalogue.s, point_theta, point_zeta), dtype=float
        )
        if derivative.shape != points.shape or not np.all(np.isfinite(derivative)):
            raise ValueError("nonfinite derivative in endpoint correction")
        integral = (end - start) * (derivative @ gl_w) / 2
        endpoint_u = (
            (entry + U).repeat(order, axis=1)[mask]
            if from_exit
            else entry.repeat(order, axis=1)[mask]
        )
        endpoint_theta, endpoint_zeta = catalogue.coordinates(endpoint_u)
        residual = (
            np.asarray(
                catalogue.field.B(catalogue.s, endpoint_theta, endpoint_zeta),
                dtype=float,
            )
            - b.repeat(order, axis=1)[mask]
        )
        difference[mask] = -residual + integral if from_exit else -residual - integral
    radicand = difference / b
    if np.any(radicand <= 0) or not np.all(np.isfinite(radicand)):
        raise ValueError("nonpositive bounce radicand")
    common = abs(catalogue.C) * jac / B
    A = common * np.sqrt(radicand)
    K = common / np.sqrt(radicand)
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(K)):
        raise ValueError("nonfinite bounce quadrature")
    segment_A = np.sum(A * x_weight, axis=1)
    segment_K = np.sum(K * x_weight, axis=1)
    return (
        np.bincount(owner, weights=segment_A, minlength=len(wells)),
        np.bincount(owner, weights=segment_K, minlength=len(wells)),
        segment_A,
        segment_K,
    )


def adaptive_bounce_integral(
    catalogue: ForwardLineCatalogue, well: OrdinaryWell
) -> BounceIntegral:
    """Independent adaptive QUADPACK reference for A,K (§§4.2, 9.3)."""
    cfg = catalogue.config
    entry, exit = well.u_in, well.u_out
    slope_in = float(catalogue._D(entry))
    slope_out = float(catalogue._D(exit))
    if slope_in >= 0 or slope_out <= 0:
        return BounceIntegral(
            TraceStatus.TANGENT_OR_TRANSITION, reason="nonordinary root slope"
        )
    pair = _regularized_integrands(
        catalogue.field,
        b=well.b,
        s=catalogue.s,
        theta_in=float(well.q_in[1]),
        zeta_in=well.zeta_in,
        iota=catalogue.iota,
        sigma=catalogue.sigma,
        C_abs=abs(catalogue.C),
        u_out=exit - entry,
        slope_in=slope_in,
        slope_out=slope_out,
        residual_in=float(catalogue._B(entry)) - well.b,
        residual_out=float(catalogue._B(exit)) - well.b,
        endpoint_window=catalogue.period / catalogue.steps_per_period,
        root_tolerance=cfg.root_atol_B,
    )
    breakpoints = [0.0]
    for u in well.extrema_u:
        breakpoints.append(2 / np.pi * np.arcsin(np.sqrt((u - entry) / (exit - entry))))
    breakpoints.append(1.0)
    values = []
    errors = []
    try:
        for component in (0, 1):
            total = 0.0
            error = 0.0
            for left, right in zip(breakpoints[:-1], breakpoints[1:]):
                result = quad(
                    lambda x: pair(x)[component],
                    left,
                    right,
                    epsabs=cfg.quadrature_atol / (2 * (len(breakpoints) - 1)),
                    epsrel=cfg.quadrature_rtol / 2,
                    limit=100,
                    full_output=1,
                )
                if len(result) > 3:
                    raise ValueError("adaptive reference did not converge")
                total += result[0]
                error += result[1]
            if error > max(cfg.quadrature_atol, cfg.quadrature_rtol * abs(total)):
                raise ValueError("adaptive reference exceeded global error estimate")
            values.append(total)
            errors.append(error)
    except (ValueError, FloatingPointError, RuntimeError) as exc:
        return BounceIntegral(TraceStatus.QUADRATURE_FAILURE, reason=str(exc))
    return BounceIntegral(
        TraceStatus.REGULAR, values[0], values[1], errors[0], errors[1], "adaptive"
    )


def batched_bounce_integrals(
    catalogue: ForwardLineCatalogue,
    wells: tuple[OrdinaryWell, ...] | list[OrdinaryWell],
) -> tuple[BounceIntegral, ...]:
    """Composite batched GL A,K with order-difference estimates (§9.3).

    Failed or near-separatrix integrals use the independent adaptive fallback.
    Neither n-versus-2n nor QUADPACK's estimate is called a field enclosure.
    """
    wells = tuple(wells)
    if not wells:
        return ()
    segments = []
    for index, well in enumerate(wells):
        if well.u_out <= well.u_in or well.b <= 0:
            raise ValueError("well endpoints must be ordered in physical u")
        edges = [0.0]
        for u in well.extrema_u:
            edges.append(
                2
                / np.pi
                * np.arcsin(np.sqrt((u - well.u_in) / (well.u_out - well.u_in)))
            )
        edges.append(1.0)
        segments.extend(
            (index, left, right) for left, right in zip(edges[:-1], edges[1:])
        )
    owner = np.array([segment[0] for segment in segments], dtype=int)
    cfg = catalogue.config
    previous = None
    order = 16
    try:
        while order <= cfg.max_quadrature_order:
            current = _batch_order(catalogue, wells, segments, order)
            if previous is not None:
                # Sum local estimates before checking the global A/K budget;
                # opposite-signed segment errors must not cancel (§9.3).
                error = (
                    np.bincount(
                        owner,
                        weights=abs(current[2] - previous[2]),
                        minlength=len(wells),
                    ),
                    np.bincount(
                        owner,
                        weights=abs(current[3] - previous[3]),
                        minlength=len(wells),
                    ),
                )
                okay = (
                    error[0]
                    <= np.maximum(
                        cfg.quadrature_atol, cfg.quadrature_rtol * abs(current[0])
                    )
                ) & (
                    error[1]
                    <= np.maximum(
                        cfg.quadrature_atol, cfg.quadrature_rtol * abs(current[1])
                    )
                )
                if np.all(okay):
                    return tuple(
                        BounceIntegral(
                            TraceStatus.REGULAR,
                            float(current[0][i]),
                            float(current[1][i]),
                            float(error[0][i]),
                            float(error[1][i]),
                            "batched_gl",
                        )
                        for i in range(len(wells))
                    )
            previous = current
            order *= 2
    except (ValueError, FloatingPointError):
        pass
    return tuple(adaptive_bounce_integral(catalogue, well) for well in wells)
