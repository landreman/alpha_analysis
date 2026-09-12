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

import numpy as np
from scipy.optimize import newton

from .branch_atlas import AtlasPort, AtlasTransition, _certified_roots
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
    corrector_steps: int = 6
    action_atol: float = 2e-6
    closure_tol: float = 0.015
    root_shift_periods: float = 0.3
    gradient_floor: float = 2e-5
    event_tol: float = 0.015
    scan_periods: int = 4

    def __post_init__(self) -> None:
        if not (0 < self.min_step <= self.step < 0.2):
            raise ValueError("require 0 < min_step <= step < 0.2")
        if (
            self.max_steps < 1
            or self.max_branches < 1
            or self.corrector_steps < 1
            or self.max_certificate_boxes < 1
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
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("tolerances must be finite and positive")


@dataclass(frozen=True)
class ContourPoint:
    """One ordinary root-labelled state; angles are unwrapped radians (§8.1)."""

    s: float
    alpha: float
    zeta_in: float
    zeta_out: float
    action_length: float


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
    bound_scope: str = "represented-field numerical query"

    @property
    def witness(self) -> ContourPath | None:
        return next((p for p in self.paths if p.edge_reached), None)


class _Unresolved(RuntimeError):
    pass


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

    def _action(self, s: float, alpha: float, zin: float, zout: float) -> float:
        # A vanishes at ordinary endpoints; the cosine map still resolves the
        # square-root behavior and is reused for the singular derivative.
        t = (self._nodes + 1) / 2
        mid, half = (zin + zout) / 2, abs(zout - zin) / 2
        z = mid + half * np.cos(np.pi * t)
        jac = half * np.pi * np.sin(np.pi * t)
        B = np.asarray(self.field.B(s, self._coordinates(s, alpha, z), z), dtype=float)
        if np.any(~np.isfinite(B)) or np.any(B <= 0) or np.any(B > self.b):
            raise _Unresolved("invalid B inside continued trapped well")
        C = abs(float(self.field.C(s)))
        return float(np.dot(self._weights / 2, jac * C / B * np.sqrt(1 - B / self.b)))

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
        A = self._action(s, alpha, zin, zout)
        if not np.isfinite(A) or A <= 0:
            raise _Unresolved("ordinary well has nonpositive/unknown action")
        return ContourPoint(float(s), float(alpha), zin, zout, A)

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
                    # A scan node close to B=b can defeat an otherwise valid
                    # transverse interval test. Shift the sampling phase of
                    # the *same* lifted root window, and accept only a full
                    # certificate with the correctly ordered root pair.
                    probe_z0 = midpoint.zeta_in - sigma * 0.35 * self.period
                    probe = ForwardLineCatalogue(
                        self.field,
                        midpoint.s,
                        self._coordinates(midpoint.s, midpoint.alpha, probe_z0),
                        probe_z0,
                        scan_config,
                    )
                    for phase in (0.0, 0.5):
                        z0 = (
                            probe_z0
                            - sigma * phase * self.period / probe.steps_per_period
                        )
                        periods = max(
                            2,
                            int(
                                np.ceil(
                                    length / self.period
                                    + 0.7
                                    + phase / probe.steps_per_period
                                )
                            ),
                        )
                        if periods > self.config.scan_periods:
                            return False
                        scan = ForwardLineCatalogue(
                            self.field,
                            midpoint.s,
                            self._coordinates(midpoint.s, midpoint.alpha, z0),
                            z0,
                            scan_config,
                        )
                        scan.extend_to(periods)
                        roots, _ = _certified_roots(scan, self.b, bounds, 8)
                        if roots is None:
                            continue
                        u_in = sigma * (midpoint.zeta_in - z0)
                        u_out = sigma * (midpoint.zeta_out - z0)
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
                        if left_index is not None and right_index == left_index + 1:
                            break
                    else:
                        raise _Unresolved("scan phases did not certify root pattern")
                    continue
                except (_Unresolved, ValueError, ArithmeticError):
                    pass
                if depth >= 6 or midpoint is None:
                    return False
                pending.extend(
                    ((first, midpoint, depth + 1), (midpoint, last, depth + 1))
                )
        return True

    def query(
        self, seed: ContourPoint, events: tuple[AtlasTransition, ...] = ()
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
                eq = np.array(event.parameter) / np.array([1.0, 2 * np.pi])
                for path in paths:
                    for p in path.points:
                        x = np.array([p.s, p.alpha / (2 * np.pi)])
                        if np.linalg.norm(x - eq) < self.config.event_tol:
                            matches = sorted(
                                event.ports,
                                key=lambda port: max(
                                    abs(p.zeta_in - port.zeta_in),
                                    abs(p.zeta_out - port.zeta_out),
                                ),
                            )
                            if (
                                not matches
                                or max(
                                    abs(p.zeta_in - matches[0].zeta_in),
                                    abs(p.zeta_out - matches[0].zeta_out),
                                )
                                > self.config.root_shift_periods * self.period
                            ):
                                return ContourResult(
                                    ContourStatus.UNKNOWN,
                                    tuple(paths),
                                    "event encountered without a matched root-labelled port",
                                    event.ports,
                                )
                            try:
                                exact = self.sample(
                                    event.parameter[0], event.parameter[1], p
                                )
                                action_matches = (
                                    abs(exact.action_length - seed.action_length)
                                    <= self.config.action_atol
                                    and abs(
                                        exact.action_length - matches[0].action_length
                                    )
                                    <= self.config.action_atol
                                    + matches[0].error_estimate
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
                            branched = self.query_event(event, matches[0].role)
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
            branch = self.query(seed)
            paths.extend(branch.paths)
            outcomes.append((port.role, branch.status, branch.reason))
            if branch.status is ContourStatus.ACCESSIBLE:
                accessible_role = port.role
        if accessible_role is not None:
            return ContourResult(
                ContourStatus.ACCESSIBLE,
                tuple(paths),
                f"edge witness through {accessible_role} at common event parameter",
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
        reference = ContourPoint(
            s, alpha, port.zeta_in, port.zeta_out, port.action_length
        )
        for radius in (0.002, 0.006, 0.015):
            for ds, da in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)):
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
                    for direction in (1, -1):
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
            if path.edge_reached
            else "closed" if path.closed else "unresolved"
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
