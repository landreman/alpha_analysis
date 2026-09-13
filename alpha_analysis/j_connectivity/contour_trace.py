"""Independent continuous-field action-contour queries (DESIGN.md §11.1).

The oracle does not use an atlas mesh or a reachability graph. It follows
continued physical bounce roots in lifted (s, alpha) coordinates. Positive
answers include a field-checked witness; negative answers require a closed
contour and a finite-window root-pattern certificate along every segment.
Unproved events, numerical failure, and exhausted work remain unknown. All
actions are half-bounce lengths A in the units of |C|/B times radians (§4.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq, newton

from .branch_atlas import (
    AtlasPort,
    AtlasTransition,
    _certified_roots,
    height_at,
    transition_at,
)
from .field import BoozerFieldLike
from .forward_catalogue import ForwardLineCatalogue, ForwardScanConfig


class ContourStatus(Enum):
    """Field-query outcome; UNKNOWN is never a negative classification (§11.1)."""

    ACCESSIBLE = auto()
    INACCESSIBLE = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class ContourConfig:
    """Finite contour controls; s and alpha/(2π) share the step metric (§11.1)."""

    step: float = 0.025
    min_step: float = 0.001
    max_steps: int = 300
    max_branches: int = 16
    max_certificate_boxes: int = 64
    max_certificate_depth: int = 10
    corrector_steps: int = 6
    action_atol: float = 2e-6
    closure_tol: float = 0.015
    root_shift_periods: float = 0.3
    gradient_floor: float = 2e-5
    event_tol: float = 0.015
    event_search_radius_s: float = 0.02
    event_height_fraction: float = 0.02
    max_event_curve_samples: int = 9
    scan_periods: int = 4

    def __post_init__(self) -> None:
        if not (0 < self.min_step <= self.step < 0.2):
            raise ValueError("require 0 < min_step <= step < 0.2")
        if (
            self.max_steps < 1
            or self.max_branches < 1
            or self.corrector_steps < 1
            or self.max_certificate_boxes < 1
            or self.max_certificate_depth < 1
            or self.max_event_curve_samples < 3
        ):
            raise ValueError("work budgets must be positive")
        if self.scan_periods < 2:
            raise ValueError("scan_periods must be at least two")
        for value in (
            self.action_atol,
            self.closure_tol,
            self.root_shift_periods,
            self.gradient_floor,
            self.event_tol,
            self.event_search_radius_s,
            self.event_height_fraction,
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("tolerances must be finite and positive")


@dataclass(frozen=True)
class ContourPoint:
    """One ordinary root-labelled state; A and its estimated error have length units (§8.1)."""

    s: float
    alpha: float
    zeta_in: float
    zeta_out: float
    action_length: float
    error_A: float = np.nan


@dataclass(frozen=True)
class ContourPath:
    """Numerical constant-A path with explicit physical-root lifts (§11.1)."""

    points: tuple[ContourPoint, ...]
    closed: bool
    edge_reached: bool
    reason: str | None = None


@dataclass(frozen=True)
class ContourResult:
    """Accessibility query and its witness/limitations, not an f enclosure."""

    status: ContourStatus
    paths: tuple[ContourPath, ...]
    reason: str
    event_ports: tuple[AtlasPort, ...] = ()
    port_outcomes: tuple[tuple[str, ContourStatus, str], ...] = ()
    bound_scope: str = (
        "represented-field numerical query; estimated action quadrature error"
    )
    event_discovery: EventDiscovery | None = None

    @property
    def witness(self) -> ContourPath | None:
        if self.status is not ContourStatus.ACCESSIBLE:
            return None
        return next((p for p in self.paths if p.edge_reached), None)


class _Unresolved(RuntimeError):
    pass


@dataclass(frozen=True)
class EventDiscovery:
    """Bounded local marginal/action solve and one-sided numerical checks (§10.2)."""

    status: str
    reason: str
    attempted_s: tuple[float, ...]
    event: AtlasTransition | None = None
    incoming_role: str | None = None
    marginal_residual_B: float = np.nan
    action_residual: float = np.nan
    one_sided_action_residuals: tuple[tuple[str, float], ...] = ()


class DirectContourOracle:
    """Predict-correct A levels using direct root continuation and field gradients.

    The traced coordinate is x=(s,alpha/2π); root zeta lifts are not reduced at
    a field-period seam. A root pair is followed by both endpoints, so equal-A
    but disconnected wells cannot be memoized into one state. Events supplied
    to ``query`` or ``query_event`` must have a common physical parameter and
    certified pointwise ports (§§5.3, 10.2). Missing event geometry can leave
    a query unknown; it never establishes nonreachability.
    """

    def __init__(
        self, field: BoozerFieldLike, b: float, config: ContourConfig | None = None
    ) -> None:
        if not np.isfinite(b) or b <= 0:
            raise ValueError("bounce field b must be positive and finite")
        self.field, self.b = field, float(b)
        self.config = ContourConfig() if config is None else config
        self.period = 2 * np.pi / field.nfp
        self._nodes, self._weights = np.polynomial.legendre.leggauss(64)

    def _coordinates(self, s: float, alpha: float, zeta):
        return alpha + float(self.field.iota(s)) * zeta

    def _root(self, s: float, alpha: float, hint: float) -> float:
        def F(z):
            return float(self.field.B(s, self._coordinates(s, alpha, z), z)) - self.b

        def D(z):
            return float(self.field.D_B(s, self._coordinates(s, alpha, z), z))

        try:
            z = float(newton(F, hint, fprime=D, tol=2e-12, maxiter=15))
        except (ValueError, RuntimeError, OverflowError, ZeroDivisionError) as exc:
            raise _Unresolved(f"bounce-root continuation failed: {exc}") from exc
        if (
            not np.isfinite(z)
            or abs(z - hint) > self.config.root_shift_periods * self.period
        ):
            raise _Unresolved("bounce root moved outside the continuation window")
        if abs(F(z)) > 1e-8 * max(self.b, 1.0):
            raise _Unresolved("continued bounce root has a field residual")
        return z

    def _action(
        self, s: float, alpha: float, zin: float, zout: float
    ) -> tuple[float, float]:
        """Evaluate §4.2 A with an adaptive numerical error estimate, not a field bound."""
        # A vanishes at ordinary endpoints. The cosine map removes their
        # square-root behavior; adaptive integration resolves interior structure.
        mid, half = (zin + zout) / 2, abs(zout - zin) / 2
        C = abs(float(self.field.C(s)))
        iota = float(self.field.iota(s))

        def integrand(t: float) -> float:
            z = mid + half * np.cos(np.pi * t)
            B = float(self.field.B(s, alpha + iota * z, z))
            if not np.isfinite(B) or B <= 0 or B > self.b + 1e-10 * max(self.b, 1.0):
                raise _Unresolved("invalid B inside continued trapped well")
            return (
                half
                * np.pi
                * np.sin(np.pi * t)
                * C
                / B
                * np.sqrt(max(0.0, 1 - B / self.b))
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", IntegrationWarning)
                value, error = quad(
                    integrand,
                    0.0,
                    1.0,
                    epsabs=self.config.action_atol / 4,
                    epsrel=1e-10,
                    limit=200,
                )
        except (ValueError, IntegrationWarning) as exc:
            raise _Unresolved(f"action quadrature failed: {exc}") from exc
        if (
            not np.isfinite(value)
            or not np.isfinite(error)
            or error > self.config.action_atol / 2
        ):
            raise _Unresolved("action quadrature error exceeds tolerance")
        return float(value), float(error)

    def seed(
        self, s: float, alpha: float, zeta_in_hint: float | None = None
    ) -> ContourPoint:
        """Find one complete ordinary well on a lifted field line (§§3.4, 9.2).

        With no hint, choose the entry closest to zero; supply a lifted entry
        hint to distinguish multiple wells on the same transverse line.
        """
        if not (0 <= s <= 1) or not np.isfinite(alpha):
            raise ValueError("seed requires 0<=s<=1 and finite alpha")
        sigma = np.sign(float(self.field.C(s)))
        if sigma == 0:
            raise _Unresolved("zero physical field-line orientation")
        z0 = -sigma * self.period
        scan = ForwardLineCatalogue(
            self.field,
            s,
            self._coordinates(s, alpha, z0),
            z0,
            ForwardScanConfig(max_periods=self.config.scan_periods),
        )
        scan.extend_to(self.config.scan_periods)
        query = scan.query(self.b)
        if not query.wells:
            raise _Unresolved(
                query.reason or "no complete ordinary well in scan window"
            )
        target = 0.0 if zeta_in_hint is None else zeta_in_hint
        well = min(query.wells, key=lambda w: abs(w.zeta_in - target))
        if zeta_in_hint is not None and abs(well.zeta_in - target) > self.period / 2:
            raise _Unresolved("requested incoming-root lift not found")
        return self.sample(
            float(s),
            float(alpha),
            ContourPoint(s, alpha, well.zeta_in, well.zeta_out, np.nan),
        )

    def sample(self, s: float, alpha: float, reference: ContourPoint) -> ContourPoint:
        """Continue both ordinary roots and evaluate A of §4.2 at (s,alpha)."""
        if not (0 <= s <= 1) or not np.isfinite(alpha):
            raise _Unresolved("contour left the physical (s,alpha) domain")
        zin = self._root(s, alpha, reference.zeta_in)
        zout = self._root(s, alpha, reference.zeta_out)
        sigma = np.sign(float(self.field.C(s)))
        if sigma * (zout - zin) <= 0:
            raise _Unresolved("continued roots reversed physical first-return order")
        d_in = sigma * float(self.field.D_B(s, self._coordinates(s, alpha, zin), zin))
        d_out = sigma * float(
            self.field.D_B(s, self._coordinates(s, alpha, zout), zout)
        )
        if d_in >= -1e-9 or d_out <= 1e-9:
            raise _Unresolved("marginal or incorrectly oriented bounce root")
        A, error_A = self._action(s, alpha, zin, zout)
        if not np.isfinite(A) or A <= 0:
            raise _Unresolved("ordinary well has nonpositive/unknown action")
        return ContourPoint(float(s), float(alpha), zin, zout, A, error_A)

    def action_gradient(self, point: ContourPoint) -> np.ndarray:
        """Return (∂s A, ∂alpha A), including C' and iota' z B_theta (§11.1).

        Endpoint contributions vanish for A, but differentiation of the
        integrand retains the radial variation of |G+iota I|/B and the field
        line's shear. The endpoint-singular derivative uses a cosine map.
        This is a numerical gradient of the represented field, not an enclosure.
        """
        s, alpha = point.s, point.alpha
        t = (self._nodes + 1) / 2
        mid = (point.zeta_in + point.zeta_out) / 2
        half = abs(point.zeta_out - point.zeta_in) / 2
        z = mid + half * np.cos(np.pi * t)
        jac = half * np.pi * np.sin(np.pi * t)
        theta = self._coordinates(s, alpha, z)
        B = np.asarray(self.field.B(s, theta, z), dtype=float)
        if np.any(B <= 0) or np.any(B >= self.b):
            raise _Unresolved("action gradient has a nonordinary quadrature node")
        B_theta = np.asarray(self.field.dB_dtheta(s, theta, z), dtype=float)
        diota = getattr(self.field, "diota_ds", None)
        if diota is None:
            eps = min(1e-5, (1 - s) / 2, s / 2)
            if eps <= 0:
                eps = 1e-5
                slope_iota = (
                    float(self.field.iota(s + eps) - self.field.iota(s)) / eps
                    if s == 0
                    else float(self.field.iota(s) - self.field.iota(s - eps)) / eps
                )
            else:
                slope_iota = float(
                    self.field.iota(s + eps) - self.field.iota(s - eps)
                ) / (2 * eps)
        else:
            slope_iota = float(diota(s))
        B_s = (
            np.asarray(self.field.dB_ds(s, theta, z), dtype=float)
            + slope_iota * z * B_theta
        )
        eps = min(1e-5, (1 - s) / 2, s / 2)
        if eps <= 0:
            eps = 1e-5
            Cprime = (
                float(self.field.C(s + eps) - self.field.C(s)) / eps
                if s == 0
                else float(self.field.C(s) - self.field.C(s - eps)) / eps
            )
        else:
            Cprime = float(self.field.C(s + eps) - self.field.C(s - eps)) / (2 * eps)
        C = float(self.field.C(s))
        if C == 0:
            raise _Unresolved("zero field-line orientation in action gradient")
        C_abs, C_abs_prime = abs(C), np.sign(C) * Cprime
        root = np.sqrt(1 - B / self.b)
        B_factor = -C_abs * (root / B**2 + 1 / (2 * self.b * B * root))
        weight = self._weights / 2 * jac
        gradient = np.array(
            [
                np.dot(weight, C_abs_prime * root / B + B_factor * B_s),
                np.dot(weight, B_factor * B_theta),
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(gradient)):
            raise _Unresolved("nonfinite action gradient")
        return gradient

    def _equivalent(self, first: ContourPoint, last: ContourPoint) -> bool:
        """Check closure under the explicit radius-dependent lifted seam (§8.2)."""
        if abs(first.s - last.s) > self.config.closure_tol:
            return False
        shift = round((first.zeta_in - last.zeta_in) / self.period)
        if (
            abs(last.zeta_in - (first.zeta_in - shift * self.period))
            > 0.08 * self.period
        ):
            return False
        if (
            abs(last.zeta_out - (first.zeta_out - shift * self.period))
            > 0.08 * self.period
        ):
            return False
        expected = first.alpha + shift * float(self.field.iota(first.s)) * self.period
        angular = np.angle(np.exp(1j * (last.alpha - expected)))
        return abs(angular) / (2 * np.pi) <= self.config.closure_tol

    def _correct(
        self, predicted: np.ndarray, previous: ContourPoint, target: float
    ) -> ContourPoint:
        x = predicted.copy()
        at_edge = x[0] >= 1
        if at_edge:
            x[0] = 1.0
        for _ in range(self.config.corrector_steps):
            candidate = self.sample(float(x[0]), float(2 * np.pi * x[1]), previous)
            error = candidate.action_length - target
            if abs(error) <= self.config.action_atol:
                return candidate
            grad = self.action_gradient(candidate)
            scaled = np.array([grad[0], 2 * np.pi * grad[1]])
            if at_edge:
                if abs(scaled[1]) <= self.config.gradient_floor:
                    raise _Unresolved("edge-level corrector is tangent or singular")
                x[1] -= error / scaled[1]
            else:
                norm2 = float(np.dot(scaled, scaled))
                if norm2 <= self.config.gradient_floor**2:
                    raise _Unresolved("saddle/flat action gradient in corrector")
                x -= error * scaled / norm2
            if not (0 <= x[0] <= 1) or not np.all(np.isfinite(x)):
                raise _Unresolved("constant-action correction left physical domain")
        raise _Unresolved("constant-action corrector did not converge")

    def _trace_one(self, seed: ContourPoint, direction: int) -> ContourPath:
        cfg = self.config
        target = seed.action_length
        points = [seed]
        traveled = 0.0
        for step_index in range(cfg.max_steps):
            current = points[-1]
            grad = self.action_gradient(current)
            scaled = np.array([grad[0], 2 * np.pi * grad[1]])
            norm = float(np.linalg.norm(scaled))
            if norm <= cfg.gradient_floor:
                return ContourPath(
                    tuple(points), False, False, "saddle/flat action gradient"
                )
            tangent = direction * np.array([-scaled[1], scaled[0]]) / norm
            x0 = np.array([current.s, current.alpha / (2 * np.pi)])
            local_step = cfg.step
            failure = "predictor-corrector failed"
            while local_step >= cfg.min_step * (1 - 1e-12):
                try:
                    predicted = x0 + local_step * tangent
                    if predicted[0] < 0:
                        raise _Unresolved(
                            "contour reached the unresolved magnetic axis"
                        )
                    next_point = self._correct(predicted, current, target)
                    if (
                        np.linalg.norm(
                            np.array([next_point.s, next_point.alpha / (2 * np.pi)])
                            - x0
                        )
                        > 2 * local_step
                    ):
                        raise _Unresolved(
                            "corrector jumped outside local predictor step"
                        )
                    break
                except _Unresolved as exc:
                    failure = str(exc)
                    local_step /= 2
            else:
                return ContourPath(tuple(points), False, False, failure)
            traveled += float(
                np.linalg.norm(
                    np.array([next_point.s, next_point.alpha / (2 * np.pi)]) - x0
                )
            )
            points.append(next_point)
            if next_point.s == 1.0:
                return ContourPath(tuple(points), False, True)
            if (
                step_index >= 7
                and traveled > max(8 * cfg.step, 6 * cfg.closure_tol)
                and self._equivalent(seed, next_point)
            ):
                initial_gradient = self.action_gradient(seed)
                final_gradient = self.action_gradient(next_point)
                initial_scaled = np.array(
                    [initial_gradient[0], 2 * np.pi * initial_gradient[1]]
                )
                final_scaled = np.array(
                    [final_gradient[0], 2 * np.pi * final_gradient[1]]
                )
                alignment = float(np.dot(initial_scaled, final_scaled)) / (
                    np.linalg.norm(initial_scaled) * np.linalg.norm(final_scaled)
                )
                if not np.isfinite(alignment) or alignment < 0.8:
                    return ContourPath(
                        tuple(points), False, False, "closure tangent mismatch"
                    )
                return ContourPath(tuple(points), True, False)
        return ContourPath(tuple(points), False, False, "contour step budget exhausted")

    def _root_pattern_certified(self, path: ContourPath) -> bool:
        """Certify regular first-return topology in segment boxes (§§8.4, 11.1).

        This uses field/scan enclosures, never atlas triangles or a connectivity
        result. A failed certificate makes negative closure unknown.
        """
        checked_boxes = 0
        for left, right in zip(path.points[:-1], path.points[1:]):
            pending = [(left, right, 0)]
            while pending:
                checked_boxes += 1
                if checked_boxes > self.config.max_certificate_boxes:
                    return False
                first, last, depth = pending.pop()
                midpoint = None
                try:
                    midpoint = self.sample(
                        (first.s + last.s) / 2,
                        (first.alpha + last.alpha) / 2,
                        first,
                    )
                    sigma = np.sign(float(self.field.C(midpoint.s)))
                    length = sigma * (midpoint.zeta_out - midpoint.zeta_in)
                    pad = 1e-8
                    bounds = (
                        max(0.0, min(first.s, last.s) - pad),
                        min(1.0, max(first.s, last.s) + pad),
                        min(first.alpha, last.alpha) - pad,
                        max(first.alpha, last.alpha) + pad,
                    )
                    scan_config = ForwardScanConfig(
                        max_periods=self.config.scan_periods
                    )
                    probe_z0 = midpoint.zeta_in - sigma * 0.35 * self.period
                    probe = ForwardLineCatalogue(
                        self.field,
                        midpoint.s,
                        self._coordinates(midpoint.s, midpoint.alpha, probe_z0),
                        probe_z0,
                        scan_config,
                    )
                    node_step = self.period / probe.steps_per_period
                    guard = max(
                        4 * node_step,
                        min(0.15 * length, 0.04 * self.period),
                    )
                    span = length + 2 * guard
                    if span > self.config.scan_periods * self.period:
                        return False
                    z0 = midpoint.zeta_in - sigma * guard
                    scan = ForwardLineCatalogue(
                        self.field,
                        midpoint.s,
                        self._coordinates(midpoint.s, midpoint.alpha, z0),
                        z0,
                        scan_config,
                    ).local_root_window(0.0, span)
                    roots, _ = _certified_roots(scan, self.b, bounds, 12)
                    if roots is None:
                        raise _Unresolved("selected root guards did not certify")
                    u_in, u_out = guard, guard + length
                    left_index = next(
                        (
                            i
                            for i, (a, b, sign) in enumerate(roots)
                            if sign < 0 and a - 1e-8 <= u_in <= b + 1e-8
                        ),
                        None,
                    )
                    right_index = next(
                        (
                            i
                            for i, (a, b, sign) in enumerate(roots)
                            if sign > 0 and a - 1e-8 <= u_out <= b + 1e-8
                        ),
                        None,
                    )
                    if left_index is None or right_index != left_index + 1:
                        raise _Unresolved("selected roots are not a first-return pair")
                    continue
                except (_Unresolved, ValueError, ArithmeticError):
                    pass
                if depth >= self.config.max_certificate_depth or midpoint is None:
                    return False
                pending.extend(
                    ((first, midpoint, depth + 1), (midpoint, last, depth + 1))
                )
        return True

    def _event_on_height_curve(
        self, s: float, alpha_guess: float, zeta_guess: float
    ) -> AtlasTransition:
        """Solve H(s,alpha)=b on one continued nondegenerate max (§10.1)."""
        alpha, zeta = float(alpha_guess), float(zeta_guess)
        for _ in range(12):
            height = self._nearby_maximum(s, alpha, zeta)
            if height.curvature >= 0:
                raise _Unresolved("candidate marginal extremum is not a maximum")
            zeta = height.zeta
            residual = height.value - self.b
            if abs(residual) <= 1e-10 * max(1.0, self.b):
                event = transition_at(
                    self.field,
                    self.b,
                    s,
                    alpha,
                    periods=2,
                    tolerance_B=1e-8 * max(1.0, self.b),
                )
                if event.status != "generic":
                    raise _Unresolved(event.reason or "marginal event is not generic")
                return event
            if abs(height.gradient_alpha) < 1e-10:
                raise _Unresolved("height curve has unresolved alpha tangent")
            step = np.clip(residual / height.gradient_alpha, -0.1, 0.1)
            alpha -= float(step)
        raise _Unresolved("height-curve solve budget exhausted")

    def _nearby_maximum(self, s: float, alpha: float, zeta: float):
        """Select one lifted nondegenerate max with bounded bracket retries."""
        for fraction in (0.02, 0.05, 0.1, 0.2):
            try:
                height = height_at(
                    self.field,
                    s,
                    alpha,
                    zeta,
                    bracket=fraction * self.period,
                )
            except (ValueError, ArithmeticError):
                continue
            if height.curvature < 0:
                return height
        raise _Unresolved("no nearby nondegenerate marginal maximum")

    def _one_sided_action_checks(
        self, event: AtlasTransition
    ) -> tuple[tuple[str, float], ...]:
        """Compare independent ordinary A limits with all pointwise ports.

        These are convergence estimates in length units, not rigorous action
        enclosures. The parent is approached from H<b, children from H>b.
        """
        height = height_at(
            self.field,
            event.parameter[0],
            event.parameter[1],
            event.marginal_zeta[0],
            bracket=0.2 * self.period,
        )
        normal = np.array([height.gradient_s, 2 * np.pi * height.gradient_alpha])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-10:
            raise _Unresolved("event height has no transverse normal")
        normal /= norm
        center = np.array([event.parameter[0], event.parameter[1] / (2 * np.pi)])
        checks = []
        for port in event.ports:
            away = 0.02 * self.period * np.sign(float(self.field.C(center[0])))
            reference = ContourPoint(
                event.parameter[0],
                event.parameter[1],
                port.zeta_in + (away if port.role == "child_3" else 0.0),
                port.zeta_out - (away if port.role == "child_1" else 0.0),
                port.action_length,
            )
            side = -1 if port.role == "parent" else 1
            values = []
            for radius in (1.25e-4, 6.25e-5, 3.125e-5):
                x = center + side * radius * normal
                if not 0 < x[0] < 1:
                    raise _Unresolved("one-sided event probe leaves radial support")
                ordinary = self.sample(float(x[0]), float(2 * np.pi * x[1]), reference)
                values.append(ordinary.action_length)
            if not abs(values[-1] - port.action_length) < abs(
                values[0] - port.action_length
            ):
                raise _Unresolved(f"{port.role} one-sided action does not converge")
            extrapolated = 2 * values[-1] - values[-2]
            residual = abs(extrapolated - port.action_length)
            if residual > max(20 * self.config.action_atol, 5e-4):
                raise _Unresolved(f"{port.role} one-sided action limit disagrees")
            checks.append((port.role, float(residual)))
        return tuple(checks)

    def discover_generic_event(self, point: ContourPoint) -> EventDiscovery:
        """Discover a nearby common-parameter generic event at adopted A (§23 R3.5).

        A candidate must be a continued marginal maximum near a selected
        endpoint. Its height curve is sampled within a fixed radial budget;
        a sign-changing port-action residual is solved, with all incident
        one-sided actions checked independently. Failure is explicitly unknown.
        """
        attempts: list[float] = []
        candidates = []
        for zeta in (point.zeta_in, point.zeta_out):
            try:
                height = self._nearby_maximum(point.s, point.alpha, zeta)
            except (ValueError, RuntimeError, ArithmeticError):
                continue
            if height.curvature < 0 and abs(height.value - self.b) <= (
                self.config.event_height_fraction * self.b
            ):
                candidates.append((abs(height.value - self.b), height.zeta))
        if not candidates:
            return EventDiscovery("not_near", "no nearby marginal maximum", ())
        _, zeta = min(candidates)
        try:
            center = self._event_on_height_curve(point.s, point.alpha, zeta)
            nearby = []
            for port in center.ports:
                for shift in range(
                    -self.config.scan_periods, self.config.scan_periods + 1
                ):
                    expected_alpha = (
                        center.parameter[1]
                        + shift
                        * float(self.field.iota(center.parameter[0]))
                        * self.period
                    )
                    angular = np.angle(np.exp(1j * (point.alpha - expected_alpha)))
                    distance = np.hypot(
                        point.s - center.parameter[0], angular / (2 * np.pi)
                    )
                    root_distance = max(
                        abs(point.zeta_in + shift * self.period - port.zeta_in),
                        abs(point.zeta_out + shift * self.period - port.zeta_out),
                    )
                    if (
                        distance <= self.config.event_tol
                        and root_distance
                        <= self.config.root_shift_periods * self.period
                    ):
                        nearby.append((root_distance, port.role))
            if not nearby:
                raise _Unresolved("candidate event lacks incident root-labelled port")
            role = min(nearby)[1]
            center_port = next(port for port in center.ports if port.role == role)
            radius = self.config.event_search_radius_s
            s_values = np.linspace(
                max(1e-8, point.s - radius),
                min(1 - 1e-8, point.s + radius),
                self.config.max_event_curve_samples,
            )
            curve = []
            for s in s_values:
                attempts.append(float(s))
                try:
                    event = self._event_on_height_curve(
                        float(s), center.parameter[1], center.marginal_zeta[0]
                    )
                    port = next(p for p in event.ports if p.role == role)
                    if (
                        max(
                            abs(port.zeta_in - center_port.zeta_in),
                            abs(port.zeta_out - center_port.zeta_out),
                        )
                        > self.config.root_shift_periods * self.period
                    ):
                        continue
                    curve.append((float(s), port.action_length - point.action_length))
                except (ValueError, RuntimeError, ArithmeticError, StopIteration):
                    curve.append((float(s), np.nan))
            bracket = next(
                (
                    (left[0], right[0])
                    for left, right in zip(curve[:-1], curve[1:])
                    if np.isfinite(left[1])
                    and np.isfinite(right[1])
                    and left[1] * right[1] <= 0
                ),
                None,
            )
            if bracket is None:
                raise _Unresolved("event action has no certified local sign bracket")

            def residual_at(s):
                event = self._event_on_height_curve(
                    float(s), center.parameter[1], center.marginal_zeta[0]
                )
                port = next(p for p in event.ports if p.role == role)
                return port.action_length - point.action_length

            root_s = brentq(residual_at, *bracket, xtol=1e-11)
            event = self._event_on_height_curve(
                root_s, center.parameter[1], center.marginal_zeta[0]
            )
            port = next(p for p in event.ports if p.role == role)
            action_residual = abs(port.action_length - point.action_length)
            if action_residual > self.config.action_atol + port.error_estimate:
                raise _Unresolved("located event misses the adopted action")
            one_sided = self._one_sided_action_checks(event)
            marginal = abs(
                height_at(
                    self.field,
                    event.parameter[0],
                    event.parameter[1],
                    event.marginal_zeta[0],
                ).value
                - self.b
            )
            return EventDiscovery(
                "verified",
                "generic marginal event at adopted action with one-sided port checks",
                tuple(attempts),
                event,
                role,
                float(marginal),
                float(action_residual),
                one_sided,
            )
        except (ValueError, RuntimeError, ArithmeticError, StopIteration) as error:
            return EventDiscovery("unresolved", str(error), tuple(attempts))

    def query(
        self,
        seed: ContourPoint,
        events: tuple[AtlasTransition, ...] = (),
        _discover_events: bool = True,
    ) -> ContourResult:
        """Follow one root-labelled contour to edge, certified closure, or unknown.

        Both orientations are explored before a negative result. Event geometry
        that is met but not fully resolved is unknown; a nearby port cannot be
        interpreted by action overlap alone (§10.2).
        """
        try:
            seed = self.sample(seed.s, seed.alpha, seed)
        except _Unresolved as exc:
            return ContourResult(ContourStatus.UNKNOWN, (), str(exc))
        paths = []
        for direction in (1, -1):
            path = self._trace_one(seed, direction)
            if path.edge_reached:
                if self._root_pattern_certified(path):
                    if _discover_events and not events:
                        discovery = self.discover_generic_event(seed)
                        if discovery.status == "verified":
                            branched = self.query_event(
                                discovery.event, discovery.incoming_role
                            )
                            return ContourResult(
                                ContourStatus.ACCESSIBLE,
                                tuple(paths) + (path,) + branched.paths,
                                "edge witness and discovered local event",
                                branched.event_ports,
                                branched.port_outcomes,
                                event_discovery=discovery,
                            )
                    return ContourResult(
                        ContourStatus.ACCESSIBLE, tuple(paths) + (path,), "edge witness"
                    )
                path = ContourPath(
                    path.points,
                    path.closed,
                    path.edge_reached,
                    "edge path has uncertified root topology",
                )
            paths.append(path)
        if events:
            # Pointwise R2 ports are not a complete event-curve catalogue. A
            # matched supplied event is expanded by query_event; otherwise an
            # unlocated/uncertified possible event cannot make a negative proof.
            for event in events:
                if event.status != "generic":
                    return ContourResult(
                        ContourStatus.UNKNOWN, tuple(paths), "unresolved supplied event"
                    )
                for path in paths:
                    for p in path.points:
                        near_event = False
                        matches = []
                        for port in event.ports:
                            shift = round((port.zeta_in - p.zeta_in) / self.period)
                            expected_alpha = (
                                event.parameter[1]
                                + shift
                                * float(self.field.iota(event.parameter[0]))
                                * self.period
                            )
                            angular = np.angle(np.exp(1j * (p.alpha - expected_alpha)))
                            distance = np.hypot(
                                p.s - event.parameter[0], angular / (2 * np.pi)
                            )
                            if distance >= self.config.event_tol:
                                continue
                            near_event = True
                            root_distance = max(
                                abs(p.zeta_in + shift * self.period - port.zeta_in),
                                abs(p.zeta_out + shift * self.period - port.zeta_out),
                            )
                            if (
                                root_distance
                                <= self.config.root_shift_periods * self.period
                            ):
                                matches.append((root_distance, port, shift))
                        if near_event:
                            if not matches:
                                return ContourResult(
                                    ContourStatus.UNKNOWN,
                                    tuple(paths),
                                    "event encountered without a matched root-labelled port",
                                    event.ports,
                                )
                            _, matched_port, shift = min(
                                matches, key=lambda item: item[0]
                            )
                            try:
                                exact = self.sample(
                                    event.parameter[0],
                                    event.parameter[1],
                                    ContourPoint(
                                        p.s,
                                        event.parameter[1],
                                        p.zeta_in + shift * self.period,
                                        p.zeta_out + shift * self.period,
                                        p.action_length,
                                        p.error_A,
                                    ),
                                )
                                action_matches = (
                                    abs(exact.action_length - seed.action_length)
                                    <= self.config.action_atol
                                    and abs(
                                        exact.action_length - matched_port.action_length
                                    )
                                    <= self.config.action_atol
                                    + matched_port.error_estimate
                                )
                            except _Unresolved:
                                action_matches = False
                            if not action_matches or not self._root_pattern_certified(
                                path
                            ):
                                return ContourResult(
                                    ContourStatus.UNKNOWN,
                                    tuple(paths),
                                    "event contact lacks an exact-action path certificate",
                                    event.ports,
                                )
                            branched = self.query_event(event, matched_port.role)
                            return ContourResult(
                                branched.status,
                                tuple(paths) + branched.paths,
                                branched.reason,
                                event.ports,
                                branched.port_outcomes,
                            )
        if all(path.closed and self._root_pattern_certified(path) for path in paths):
            return ContourResult(
                ContourStatus.INACCESSIBLE,
                tuple(paths),
                "closed regular contour in both orientations",
            )
        if _discover_events and not events:
            # A local marginal may be approached at either endpoint or before
            # a numerical continuation failure. The finite candidate list is
            # explicit; a failed search never becomes negative closure.
            candidates = (seed,) + tuple(
                path.points[-1] for path in paths if path.points
            )
            for point in candidates:
                discovery = self.discover_generic_event(point)
                if discovery.status == "verified":
                    branched = self.query_event(
                        discovery.event, discovery.incoming_role
                    )
                    return ContourResult(
                        ContourStatus.UNKNOWN,
                        tuple(paths) + branched.paths,
                        "discovered local event; global branch incidence remains unresolved",
                        branched.event_ports,
                        branched.port_outcomes,
                        event_discovery=discovery,
                    )
                if discovery.status == "unresolved":
                    return ContourResult(
                        ContourStatus.UNKNOWN,
                        tuple(paths),
                        f"local event discovery unresolved: {discovery.reason}",
                        event_discovery=discovery,
                    )
        reason = "; ".join(
            p.reason or "uncertified closure" for p in paths if not p.closed
        )
        if not reason:
            reason = "closed trace lacks a root-pattern certificate"
        return ContourResult(ContourStatus.UNKNOWN, tuple(paths), reason)

    def query_event(self, event: AtlasTransition, incoming_role: str) -> ContourResult:
        """Expand all generic ports at exactly one physical parameter (§5.3).

        The R2 event supplies pointwise limiting roots/actions. Try regular
        continuations on *every* incident port; a failed one stays unknown.
        The event curve itself is not certified by pointwise ports, so all
        negative child results still leave the event query unknown.
        """
        if event.status != "generic" or len(event.ports) < 3:
            return ContourResult(
                ContourStatus.UNKNOWN, (), event.reason or "uncertified event"
            )
        if incoming_role not in {port.role for port in event.ports}:
            raise ValueError("incoming role is not an event port")
        if any(port.event_parameter != event.parameter for port in event.ports):
            raise ValueError("event ports do not share a physical parameter")
        if len(event.ports) > self.config.max_branches:
            return ContourResult(
                ContourStatus.UNKNOWN, (), "event branch budget exhausted", event.ports
            )
        paths = []
        outcomes = []
        accessible_role = None
        for port in event.ports:
            seed = self._regular_seed_near_port(event, port)
            if seed is None:
                outcomes.append(
                    (
                        port.role,
                        ContourStatus.UNKNOWN,
                        "no regular constant-action continuation",
                    )
                )
                continue
            branch = self.query(seed, _discover_events=False)
            paths.extend(branch.paths)
            outcomes.append((port.role, branch.status, branch.reason))
            if branch.status is ContourStatus.ACCESSIBLE:
                accessible_role = port.role
        if accessible_role is not None:
            return ContourResult(
                ContourStatus.UNKNOWN,
                tuple(paths),
                f"edge path through {accessible_role} remains unbound to the incident event branch",
                event.ports,
                tuple(outcomes),
            )
        return ContourResult(
            ContourStatus.UNKNOWN,
            tuple(paths),
            "all common-parameter ports explored; event-curve or branch uncertainty remains",
            event.ports,
            tuple(outcomes),
        )

    def _regular_seed_near_port(
        self, event: AtlasTransition, port: AtlasPort
    ) -> ContourPoint | None:
        """Search both event sides, preserving the limiting root-pair identity."""
        s, alpha = event.parameter
        sigma = np.sign(float(self.field.C(s)))
        # Child ports end/start at the marginal Γmax root. Newton cannot start
        # at that zero-derivative point; approach the child from its well side.
        away = 0.02 * self.period * sigma
        reference = ContourPoint(
            s,
            alpha,
            port.zeta_in + (away if port.role == "child_3" else 0.0),
            port.zeta_out - (away if port.role == "child_1" else 0.0),
            port.action_length,
        )
        for radius in (0.002, 0.006, 0.015):
            for ds, da in (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 0.1),
                (1, 0.2),
                (-1, -0.1),
                (-1, -0.2),
                (1, 0.4),
                (-1, -0.4),
                (1, 1),
                (-1, -1),
            ):
                trial_s = s + radius * ds
                trial_alpha = alpha + 2 * np.pi * radius * da
                if not 0 < trial_s < 1:
                    continue
                try:
                    regular = self.sample(trial_s, trial_alpha, reference)
                    gradient = self.action_gradient(regular)
                    scaled = np.array([gradient[0], 2 * np.pi * gradient[1]])
                    norm = float(np.linalg.norm(scaled))
                    if norm <= self.config.gradient_floor:
                        continue
                    tangent = np.array([-scaled[1], scaled[0]]) / norm
                    x = np.array([trial_s, trial_alpha / (2 * np.pi)])
                    for direction in (0, 1, -1):
                        candidate_x = x + direction * radius * tangent
                        if not 0 < candidate_x[0] < 1:
                            continue
                        candidate = self._correct(
                            candidate_x, regular, port.action_length
                        )
                        if (
                            abs(candidate.action_length - port.action_length)
                            <= self.config.action_atol
                        ):
                            return candidate
                except _Unresolved:
                    continue
        return None


def plot_contour_result(
    result: ContourResult,
    *,
    field_label: str,
    b: float,
    source_label: str = "h(rho)=1",
):
    """Plot root-labelled contour witnesses and unknowns in lifted q (§17.3).

    Coordinates are normalized toroidal flux s and unwrapped alpha radians;
    color indicates the half-bounce action length A. The plot visualizes a
    numerical query and does not promote it to a field enclosure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for index, path in enumerate(result.paths):
        alpha = np.array([p.alpha for p in path.points])
        s = np.array([p.s for p in path.points])
        label = (
            "edge witness"
            if path.edge_reached and result.status is ContourStatus.ACCESSIBLE
            else (
                "uncertified edge intersection"
                if path.edge_reached
                else "closed" if path.closed else "unresolved"
            )
        )
        ax.plot(alpha, s, lw=1.5, label=f"path {index + 1}: {label}")
        ax.scatter(alpha[0], s[0], marker="o", s=28)
        ax.scatter(alpha[-1], s[-1], marker="x", s=40)
    if result.event_ports:
        parameter = result.event_ports[0].event_parameter
        ax.scatter(
            parameter[1],
            parameter[0],
            marker="*",
            s=100,
            c="black",
            label="common event parameter",
        )
    ax.axhline(1, color="black", lw=0.8, ls=":", label="edge s=1")
    ax.set(xlabel="unwrapped alpha (rad)", ylabel="s", ylim=(0, 1.05))
    title_reason = result.reason
    if len(title_reason) > 65:
        title_reason = title_reason[:62] + "..."
    ax.set_title(
        f"{field_label}; b={b:.6g}; {source_label}\n"
        f"{result.status.name}: {title_reason}",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig
