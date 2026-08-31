"""Contact localization on ``B=b, D_parallel B=0`` (DESIGN.md §§5.4, 23).

Logical coordinates and the curve parameter are dimensionless; angles are in
radians. Localizing simultaneous marginal maxima does not decide their multiway
connectivity. Events retain that uncertainty independently of their geometry.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq, root

from .critical_curves import (
    CriticalCurveConfig,
    CriticalKind,
    _periodic_delta,
    _projected_midpoint,
)
from .transitions import (
    _directional_crossing,
    _map_polyline_samples,
    _scan_step,
    _refined_action,
    _adaptive_parent_action,
    _certify_interval,
    _interval_midpoint_index,
    _interval_priority,
)
from .types import TransitionStatus


@dataclass(frozen=True)
class ContactLocalizationConfig:
    """Bounded refinement controls for milestone 10.2.

    ``u_tolerance`` bounds the remaining logical curve-parameter interval.
    ``max_bisections`` counts additional transition traces per bracket.
    ``event_tolerance`` bounds logical displacement when recognizing the same
    independently solved event from two source-curve occurrences. None of these
    controls relaxes a field residual or accepts a failed transition trace.
    """

    u_tolerance: float = 1.0e-5
    max_bisections: int = 20
    event_tolerance: float = 1.0e-7
    scan_refinement_factor: int = 2

    def __post_init__(self):
        if self.max_bisections < 1:
            raise ValueError("max_bisections must be positive")
        if self.scan_refinement_factor < 2:
            raise ValueError("scan_refinement_factor must be at least two")
        for name in ("u_tolerance", "event_tolerance"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ContactBracket:
    """One source occurrence, including all new traces and its final bracket."""

    source_transition_id: int
    source_sample_pair: tuple[int, int]
    u_interval: np.ndarray
    samples: tuple
    localized: bool
    reason: str
    scan_samples: tuple = ()


@dataclass(frozen=True)
class TransitionEvent:
    """One physical nongeneric event, possibly encountered at several ``u``s.

    ``marginal_points`` contains the distinct simultaneous maxima on one lifted
    field line, reduced to the periodic logical domain. An unresolved bracket
    without an established common event keeps an empty point array instead of
    manufacturing an incidence. ``occurrences`` preserves source provenance.

    ``kind`` distinguishes the two §5.4 count-change certifications of ADR
    0003: an ``"equal_height"`` event has two or more marginal maxima at
    height ``b`` and discontinuous port actions, while a ``"fold"`` event has
    one marginal point and an interior maximum--minimum pair annihilating
    strictly below ``b``, with continuous port actions.  Both remain explicit
    hyperedge nodes; the kind never resolves multiway connectivity.
    """

    event_id: int
    marginal_points: np.ndarray
    occurrences: tuple[ContactBracket, ...]
    unresolved: bool = True
    field_line_identity: np.ndarray | None = None
    zeta_unwrapped: np.ndarray | None = None
    kind: str = "equal_height"


@dataclass(frozen=True)
class TransitionArc:
    """A continuous regular correspondence with explicit event endpoints.

    ``curve`` stores one-sided limiting values at an event, not a proposed
    resolution of that event. Event connectivity remains a separate hyperedge.
    ``endpoint_event_ids`` is in the arc's order, with -1 for an ordinary end.
    If endpoint construction fails, ``curve`` retains the source mapping for
    diagnosis, explicitly uncertified; ``source_interval`` is the requested
    unresolved span and no limiting event action is claimed for that payload.
    """

    curve: object
    source_transition_id: int
    endpoint_event_ids: tuple[int, int]
    unresolved_reason: str = ""
    source_u: np.ndarray | None = None
    additional_source_samples: int = 0
    unresolved_source_intervals: tuple[tuple[float, float], ...] = ()
    source_interval: tuple[float, float] | None = None


@dataclass(frozen=True)
class LocalizedTransitions:
    """Source mappings and explicit localized/unresolved nongeneric events."""

    transitions: tuple
    events: tuple[TransitionEvent, ...]
    arcs: tuple[TransitionArc, ...] = ()
    scan_artifacts: tuple[ContactBracket, ...] = ()


def _single_sample(transition, index, *, u=None):
    ports = tuple(
        replace(
            port,
            points=port.points[index : index + 1],
            zeta_unwrapped=port.zeta_unwrapped[index : index + 1],
            action_values=port.action_values[index : index + 1],
            quadrature_error=port.quadrature_error[index : index + 1],
            source_vertex_ids=port.source_vertex_ids[index : index + 1],
        )
        for port in transition.ports
    )
    return replace(
        transition,
        u=np.array([transition.u[index] if u is None else u]),
        ports=ports,
        marginal_points=transition.marginal_points[index : index + 1],
        field_line_identity=transition.field_line_identity[index : index + 1],
        event_zeta_unwrapped=transition.event_zeta_unwrapped[index : index + 1],
        additivity_residual=transition.additivity_residual[index : index + 1],
        status=transition.sample_status[index],
        sample_status=(transition.sample_status[index],),
        sample_failure_reason=(transition.sample_failure_reason[index],),
        interior_maximum_count=transition.interior_maximum_count[index : index + 1],
        barrier_margin=transition.barrier_margin[index : index + 1],
        contact_sample_pairs=None,
        sampling_samples_used=1,
        sampling_unresolved_intervals=None,
    )


def _map_point(field, critical, polyline, transition, point, u):
    s = np.sum(point[:2] ** 2)
    theta = np.arctan2(point[1], point[0])
    D2 = float(field.D2_B(s, theta, point[2]))
    temporary = replace(
        critical,
        points=np.asarray([point]),
        segments=np.empty((0, 2), dtype=np.int64),
        D2_B=np.array([D2]),
        point_kind=np.array([CriticalKind.GAMMA_MAX.value]),
        segment_kind=np.empty(0, dtype=np.int64),
        boundary_tags=np.array([0], dtype=np.int64),
    )
    singleton = replace(
        polyline,
        vertex_ids=np.array([0]),
        segment_ids=np.empty(0, dtype=np.int64),
        u=np.array([u]),
        closed=False,
    )
    mapped = _map_polyline_samples(
        field,
        temporary,
        singleton,
        transition.transition_id,
        transition.controls,
        np.array([0]),
    )
    # A new true-curve point is not an existing source mesh vertex.
    return replace(
        mapped,
        ports=tuple(
            replace(port, source_vertex_ids=np.array([-1])) for port in mapped.ports
        ),
    )


def _other_maxima(field, sample, period):
    s, alpha = sample.field_line_identity[0]
    zeta_m = sample.event_zeta_unwrapped[0, 1]
    iota = float(field.iota(s))
    theta = alpha + iota * zeta_m
    maxima = []
    for direction in (-1.0, 1.0):
        trace = _directional_crossing(
            field,
            b=sample.b,
            s=s,
            theta_m=theta,
            zeta_m=zeta_m,
            direction=direction,
            iota=iota,
            period=period,
            config=sample.controls,
        )
        for distance, curvature in zip(
            trace.extrema_distances, trace.extrema_curvatures
        ):
            if distance < trace.distance and curvature < -sample.controls.D2_tolerance:
                maxima.append(direction * distance)
    return maxima


def _solve_equal_height(field, left, right, period, config):
    """Solve both marginal equations on the same lifted field line.

    Count changes can also be root-scan artifacts or folds below ``b``. A
    count bracket alone therefore never establishes an equal-height event.
    All candidate interior maxima are considered, and exactly one local root
    with two strictly negative curvatures is required.
    """
    sample = max((left, right), key=lambda item: item.interior_maximum_count[0])
    s, alpha = sample.field_line_identity[0]
    zeta_m = sample.event_zeta_unwrapped[0, 1]
    theta_m = alpha + float(field.iota(s)) * zeta_m
    first = left.marginal_points[0]
    chord = _periodic_delta(first, right.marginal_points[0], period)
    length = np.linalg.norm(chord)
    candidates = []
    for tau in _other_maxima(field, sample, period):

        def residual(values):
            ss, theta, zeta, offset = values
            second_theta = theta + float(field.iota(ss)) * offset
            return np.array(
                [
                    float(field.B(ss, theta, zeta)) - sample.b,
                    float(field.D_B(ss, theta, zeta)),
                    float(field.B(ss, second_theta, zeta + offset)) - sample.b,
                    float(field.D_B(ss, second_theta, zeta + offset)),
                ]
            )

        # Unconstrained trial iterates can leave s>0 where the axis-regular
        # field continuation is defined. They are rejected, never accepted
        # from a NaN residual comparison.
        with np.errstate(invalid="ignore", over="ignore"):
            solved = root(residual, [s, theta_m, zeta_m, tau], tol=1.0e-11)
        ss, theta, zeta, offset = solved.x
        if not np.all(np.isfinite(solved.x)) or not 0 < ss <= 1:
            continue
        values = residual(solved.x)
        controls = sample.controls
        if (
            not np.all(np.isfinite(values))
            or np.max(np.abs(values[[0, 2]])) > controls.root_atol_B
            or np.max(np.abs(values[[1, 3]])) > controls.tangent_atol_B
        ):
            continue
        theta2 = theta + float(field.iota(ss)) * offset
        curvatures = np.array(
            [field.D2_B(ss, theta, zeta), field.D2_B(ss, theta2, zeta + offset)],
            dtype=float,
        )
        if (
            abs(offset) <= controls.root_atol_zeta
            or not np.all(np.isfinite(curvatures))
            or np.any(curvatures >= -controls.D2_tolerance)
        ):
            continue
        points = np.array(
            [
                [
                    np.sqrt(ss) * np.cos(theta),
                    np.sqrt(ss) * np.sin(theta),
                    zeta % period,
                ],
                [
                    np.sqrt(ss) * np.cos(theta2),
                    np.sqrt(ss) * np.sin(theta2),
                    (zeta + offset) % period,
                ],
            ]
        )
        delta = _periodic_delta(first, points[0], period)
        along = float(np.dot(delta, chord) / length**2) if length else np.inf
        if not -1.0e-3 <= along <= 1.0 + 1.0e-3:
            continue
        if np.linalg.norm(delta - along * chord) > max(length, config.event_tolerance):
            continue
        if not any(
            np.linalg.norm(_periodic_delta(item[0][1], points[1], period))
            <= config.event_tolerance
            for item in candidates
        ):
            candidates.append(
                (
                    points,
                    np.array([ss, theta - float(field.iota(ss)) * zeta]),
                    np.array([zeta, zeta + offset]),
                )
            )
    return candidates[0] if len(candidates) == 1 else None


def _sampled_event_geometry(field, sample, period, config):
    """Recover the marginal points of an exactly sampled tangent contact.

    The production scan's MULTIWAY stop retains the second maximum's lifted
    location even though it correctly leaves all limiting actions unknown.
    Both marginal residuals are checked before that location becomes an event.
    """
    s, alpha = sample.field_line_identity[0]
    zeta_m = sample.event_zeta_unwrapped[0, 1]
    if not np.all(np.isfinite([s, alpha, zeta_m])):
        return None
    iota = float(field.iota(s))
    maxima = [float(zeta_m)]
    points = [sample.marginal_points[0]]
    for direction in (-1.0, 1.0):
        trace = _directional_crossing(
            field,
            b=sample.b,
            s=s,
            theta_m=alpha + iota * zeta_m,
            zeta_m=zeta_m,
            direction=direction,
            iota=iota,
            period=period,
            config=sample.controls,
        )
        if trace.status is not TransitionStatus.MULTIWAY or not np.isfinite(trace.zeta):
            continue
        zeta = float(trace.zeta)
        theta = alpha + iota * zeta
        if (
            abs(float(field.B(s, theta, zeta)) - sample.b) > sample.controls.root_atol_B
            or abs(float(field.D_B(s, theta, zeta))) > sample.controls.tangent_atol_B
            or float(field.D2_B(s, theta, zeta)) >= -sample.controls.D2_tolerance
        ):
            continue
        point = np.array(
            [np.sqrt(s) * np.cos(theta), np.sqrt(s) * np.sin(theta), zeta % period]
        )
        if any(
            np.linalg.norm(_periodic_delta(point, old, period))
            <= config.event_tolerance
            for old in points
        ):
            continue
        points.append(point)
        maxima.append(zeta)
    if len(points) < 2:
        return None
    return np.asarray(points), np.array([s, alpha]), np.asarray(maxima)


def _register_event(events, geometry, occurrence, period, tolerance):
    """Match all marginal points and their lifted separations, not proximity alone."""
    if geometry is not None:
        points, identity, lift = geometry
        for index, event in enumerate(events):
            if (
                len(event.marginal_points) != len(points)
                or event.zeta_unwrapped is None
            ):
                continue
            distances = np.array(
                [
                    [
                        np.linalg.norm(_periodic_delta(a, b, period))
                        for b in event.marginal_points
                    ]
                    for a in points
                ]
            )
            order = np.argmin(distances, axis=1)
            if len(set(order)) != len(points) or np.any(
                distances[np.arange(len(points)), order] > tolerance
            ):
                continue
            # Identical reduced points after a different number of field-line
            # turns are not the same event. Only a common lift translation is
            # permitted for the entire matched set of maxima.
            shifts = event.zeta_unwrapped[order] - lift
            if (
                np.ptp(shifts) > tolerance
                or abs(event.field_line_identity[0] - identity[0]) > tolerance
            ):
                continue
            events[index] = replace(
                event, occurrences=event.occurrences + (occurrence,)
            )
            return
    events.append(
        TransitionEvent(
            len(events),
            np.empty((0, 3)) if geometry is None else geometry[0],
            (occurrence,),
            field_line_identity=None if geometry is None else geometry[1],
            zeta_unwrapped=None if geometry is None else geometry[2],
        )
    )


def _scan_artifact(field, critical, polyline, source, left, right, config, records):
    """Dissolve a scan alias only with unchanged wells and a refined midpoint.

    Two endpoint rescans must find the same extrema count while preserving
    both lifted ordinary crossings to root tolerance and all actions to their
    quadrature tolerance. A third, true-curve midpoint must independently pass
    the usual geometry/action interpolation tests. Real below-b folds are not
    dissolved merely because the highest barrier stays below b (ADR 0003).
    """
    controls = replace(
        source.controls,
        samples_per_field_period=source.controls.samples_per_field_period
        * config.scan_refinement_factor,
        samples_per_wavelength=source.controls.samples_per_wavelength
        * config.scan_refinement_factor,
    )
    refined_source = replace(source, controls=controls)
    endpoints = [
        _map_point(
            field,
            critical,
            polyline,
            refined_source,
            item.marginal_points[0],
            item.u[0],
        )
        for item in (left, right)
    ]
    records.extend(endpoints)
    if any(item.status is not TransitionStatus.REGULAR for item in endpoints):
        return None
    if endpoints[0].interior_maximum_count[0] != endpoints[1].interior_maximum_count[0]:
        return None
    for old, new in zip((left, right), endpoints):
        old_offsets = old.event_zeta_unwrapped[0] - old.event_zeta_unwrapped[0, 1]
        new_offsets = new.event_zeta_unwrapped[0] - new.event_zeta_unwrapped[0, 1]
        if np.any(
            np.abs(old_offsets - new_offsets)
            > 2
            * (
                controls.root_atol_zeta
                + controls.root_rtol
                * np.maximum(np.abs(old_offsets), np.abs(new_offsets))
            )
        ):
            return None
        for a, b in zip(old.ports, new.ports):
            if abs(a.action_values[0] - b.action_values[0]) > 2 * (
                controls.action_quadrature_atol
                + controls.action_quadrature_rtol
                * max(abs(a.action_values[0]), abs(b.action_values[0]))
            ):
                return None
    first, last = left.marginal_points[0], right.marginal_points[0]
    point = _projected_midpoint(
        first,
        last,
        first + 0.5 * _periodic_delta(first, last, critical.period),
        field,
        critical.b,
        critical.period,
        CriticalCurveConfig(
            B_tolerance=controls.root_atol_B, g_tolerance=controls.tangent_atol_B
        ),
    )
    if point is None:
        return None
    middle = _map_point(
        field, critical, polyline, refined_source, point, 0.5 * (left.u[0] + right.u[0])
    )
    records.append(middle)
    if (
        middle.status is not TransitionStatus.REGULAR
        or middle.interior_maximum_count[0] != endpoints[0].interior_maximum_count[0]
    ):
        return None
    span = float(right.u[0] - left.u[0])
    for a, m, b in zip(endpoints[0].ports, middle.ports, endpoints[1].ports):
        expected = a.points[0] + 0.5 * _periodic_delta(
            a.points[0], b.points[0], critical.period
        )
        if (
            np.linalg.norm(_periodic_delta(expected, m.points[0], critical.period))
            > controls.curve_geometry_atol + controls.curve_geometry_rtol * span
        ):
            return None
        expected_action = 0.5 * (a.action_values[0] + b.action_values[0])
        if abs(
            expected_action - m.action_values[0]
        ) > controls.curve_action_atol + controls.curve_action_rtol * max(
            abs(a.action_values[0]), abs(m.action_values[0]), abs(b.action_values[0])
        ):
            return None
    return endpoints[0], middle, endpoints[1]


def _localization_intervals(field, critical, polyline, source, left, right, config):
    """Split newly discovered count levels under one shared bracket budget."""
    pending = [(left, right)]
    finished = []
    samples = [left, right]
    used = 0
    while pending:
        index = max(
            range(len(pending)), key=lambda i: pending[i][1].u[0] - pending[i][0].u[0]
        )
        left, right = pending.pop(index)
        if right.u[0] - left.u[0] <= config.u_tolerance:
            finished.append((left, right, "localized count-change interval"))
            continue
        if used >= config.max_bisections:
            finished.append((left, right, "contact localization budget exhausted"))
            continue
        first, second = left.marginal_points[0], right.marginal_points[0]
        point = _projected_midpoint(
            first,
            second,
            first + 0.5 * _periodic_delta(first, second, critical.period),
            field,
            critical.b,
            critical.period,
            CriticalCurveConfig(
                B_tolerance=source.controls.root_atol_B,
                g_tolerance=source.controls.tangent_atol_B,
            ),
        )
        if point is None:
            finished.append((left, right, "critical-curve midpoint projection failed"))
            continue
        sample = _map_point(
            field, critical, polyline, source, point, 0.5 * (left.u[0] + right.u[0])
        )
        samples.append(sample)
        used += 1
        if sample.status is not TransitionStatus.REGULAR:
            if (
                sample.status is TransitionStatus.MULTIWAY
                and _sampled_event_geometry(field, sample, critical.period, config)
                is not None
            ):
                finished.append(
                    (sample, sample, "localized sampled equal-height contact")
                )
            else:
                finished.append(
                    (left, right, "midpoint trace: " + sample.sample_failure_reason[0])
                )
            continue
        count = sample.interior_maximum_count[0]
        if count != left.interior_maximum_count[0]:
            pending.append((left, sample))
        if count != right.interior_maximum_count[0]:
            pending.append((sample, right))
    return finished, tuple(samples)


def localize_transition_contacts(field, critical, transitions, config=None):
    """Bisect ADR 0003 brackets without choosing a multiway decomposition.

    Midpoints are corrected onto ``B=b, g=0`` before the production transition
    trace. A failed trace, exhausted budget, nonlocal solve, or unclassified
    count change remains an explicit event with its entire remaining interval.
    Equal-height event identities come from both marginal equations and the
    same lifted field line, never from companion-curve proximity alone.
    """
    config = config or ContactLocalizationConfig()
    maxima = [p for p in critical.polylines if p.kind is CriticalKind.GAMMA_MAX]
    events = []
    artifacts = []
    for transition in transitions:
        polyline = maxima[transition.transition_id]
        for pair in transition.contact_sample_pairs:
            first_index, second_index = map(int, pair)
            left = _single_sample(transition, first_index)
            right_u = float(transition.u[second_index])
            if second_index < first_index:
                right_u += transition.total_u_length
            right = _single_sample(transition, second_index, u=right_u)
            scan_samples = []
            refined = _scan_artifact(
                field, critical, polyline, transition, left, right, config, scan_samples
            )
            if refined is not None:
                artifacts.append(
                    ContactBracket(
                        transition.transition_id,
                        (first_index, second_index),
                        np.array([left.u[0], right.u[0]]),
                        tuple(refined),
                        True,
                        "scan alias dissolved by endpoint and midpoint rescans with unchanged wells",
                        tuple(scan_samples),
                    )
                )
                continue
            intervals, samples = _localization_intervals(
                field, critical, polyline, transition, left, right, config
            )
            for left, right, reason in intervals:
                localized = bool(right.u[0] - left.u[0] <= config.u_tolerance)
                geometry = None
                if localized:
                    geometry = (
                        _sampled_event_geometry(field, left, critical.period, config)
                        if left is right
                        else _solve_equal_height(
                            field, left, right, critical.period, config
                        )
                    )
                occurrence = ContactBracket(
                    transition.transition_id,
                    (first_index, second_index),
                    np.array([left.u[0], right.u[0]]),
                    samples,
                    localized and geometry is not None,
                    (
                        reason
                        if geometry is not None
                        else reason + "; equal-height event not certified"
                    ),
                    tuple(scan_samples),
                )
                _register_event(
                    events,
                    geometry,
                    occurrence,
                    critical.period,
                    config.event_tolerance,
                )
        for index, status in enumerate(transition.sample_status):
            if status not in (TransitionStatus.MULTIWAY, TransitionStatus.TANGENT):
                continue
            sample = _single_sample(transition, index)
            geometry = (
                _sampled_event_geometry(field, sample, critical.period, config)
                if status is TransitionStatus.MULTIWAY
                else None
            )
            occurrence = ContactBracket(
                transition.transition_id,
                (index, index),
                np.repeat(transition.u[index], 2),
                (sample,),
                geometry is not None,
                "sampled nongeneric event: " + transition.sample_failure_reason[index],
            )
            _register_event(
                events, geometry, occurrence, critical.period, config.event_tolerance
            )
    return LocalizedTransitions(
        tuple(transitions), tuple(events), scan_artifacts=tuple(artifacts)
    )


def _event_well(field, event, period, controls):
    """Find the outer ordinary crossings around the certified marginal maxima."""
    s, alpha = event.field_line_identity
    iota = float(field.iota(s))
    sigma = float(np.sign(float(field.G(s)) + iota * float(field.I(s))))
    marginal_zeta = np.asarray(event.zeta_unwrapped)
    ordered = marginal_zeta[np.argsort(sigma * marginal_zeta)]
    crossings = []
    for zeta, direction in ((ordered[0], -sigma), (ordered[-1], sigma)):
        trace = _directional_crossing(
            field,
            b=event.occurrences[0].samples[0].b,
            s=s,
            theta_m=alpha + iota * zeta,
            zeta_m=zeta,
            direction=direction,
            iota=iota,
            period=period,
            config=controls,
        )
        if trace.status is not TransitionStatus.REGULAR:
            raise ValueError("event outer crossing: " + trace.status.name)
        crossings.append(trace.zeta)
    roots = np.concatenate(([crossings[0]], ordered, [crossings[1]]))
    return roots, sigma


def _event_limit(field, event, sample, period, roots, sigma):
    """Evaluate a regular arc's limiting wells at its exact event endpoint.

    The selected entry/exit are continued along the lifted field line from
    that arc's last regular sample. Actions are independently integrated;
    endpoint limits do not assign connectivity at the nongeneric event.
    """
    if sample.status is not TransitionStatus.REGULAR:
        raise ValueError("event has no regular one-sided sample")
    s, alpha = event.field_line_identity
    iota = float(field.iota(s))
    near = sample.marginal_points[0]
    marginal_index = int(
        np.argmin(
            [
                np.linalg.norm(_periodic_delta(near, point, period))
                for point in event.marginal_points
            ]
        )
    )
    zeta_m = float(event.zeta_unwrapped[marginal_index])
    old = sample.event_zeta_unwrapped[0]
    shift = period * round((old[1] - zeta_m) / period)
    theta_m = alpha + iota * zeta_m
    old_theta = (
        sample.field_line_identity[0, 1]
        + float(field.iota(sample.field_line_identity[0, 0])) * old[1]
    )
    theta_shift = 2 * np.pi * round((old_theta - theta_m) / (2 * np.pi))
    alpha = alpha - iota * shift + theta_shift
    zeta_m += shift
    shifted_roots = roots + shift
    entry = float(shifted_roots[np.argmin(np.abs(shifted_roots - old[0]))])
    exit = float(shifted_roots[np.argmin(np.abs(shifted_roots - old[2]))])
    if not sigma * entry < sigma * zeta_m < sigma * exit:
        raise ValueError("event limiting well does not enclose its source maximum")
    controls = sample.controls
    distance = sigma * (exit - entry)
    step = _scan_step(field, iota, period, controls)
    count = max(2, int(np.ceil(distance / step)) + 1)
    extrema = [
        sigma * (zeta - entry)
        for zeta in shifted_roots
        if 0 < sigma * (zeta - entry) < distance
    ]
    # Keep the production root-scan resolution, including every internal
    # extremum as a quadrature breakpoint. Chunking bounds the Fourier arrays.
    for start in range(0, count - 1, 4096):
        grid = np.linspace(0, distance, count)[start : min(count, start + 4097)]
        zeta = entry + sigma * grid
        derivative = np.asarray(field.D_B(s, alpha + iota * zeta, zeta))
        for index in np.flatnonzero(derivative[:-1] * derivative[1:] < 0):
            root_distance = brentq(
                lambda q: float(
                    field.D_B(s, alpha + iota * (entry + sigma * q), entry + sigma * q)
                ),
                grid[index],
                grid[index + 1],
                xtol=controls.root_atol_zeta,
                rtol=max(controls.root_rtol, 4 * np.finfo(float).eps),
            )
            extrema.append(root_distance)
    extrema = np.array(sorted(set(extrema)))
    middle = sigma * (zeta_m - entry)
    action1, error1 = _refined_action(
        field,
        b=sample.b,
        s=s,
        alpha=alpha,
        sigma=sigma,
        start_zeta=entry,
        distance=middle,
        breakpoints=extrema[(extrema > 0) & (extrema < middle)],
        config=controls,
    )
    action3, error3 = _refined_action(
        field,
        b=sample.b,
        s=s,
        alpha=alpha,
        sigma=sigma,
        start_zeta=zeta_m,
        distance=distance - middle,
        breakpoints=extrema[(extrema > middle) & (extrema < distance)] - middle,
        config=controls,
    )
    action_parent, error_parent = _adaptive_parent_action(
        field,
        b=sample.b,
        s=s,
        alpha=alpha,
        sigma=sigma,
        start_zeta=entry,
        distance=distance,
        tangent_distance=middle,
        breakpoints=extrema,
        config=controls,
    )
    actions = (action_parent, action1, action3)
    errors = (error_parent, error1, error3)
    if any(
        error
        > controls.action_quadrature_atol
        + controls.action_quadrature_rtol * abs(action)
        for action, error in zip(actions, errors)
    ):
        raise ValueError("event limiting action quadrature failed")
    residual = action_parent - action1 - action3
    if abs(residual) > controls.additivity_atol + controls.additivity_rtol * abs(
        action_parent
    ):
        raise ValueError("event limiting action additivity failed")

    def point(zeta):
        theta = alpha + iota * zeta
        return np.array(
            [np.sqrt(s) * np.cos(theta), np.sqrt(s) * np.sin(theta), zeta % period]
        )

    ports = tuple(
        replace(
            port,
            points=np.array([point(zeta)]),
            zeta_unwrapped=np.array([zeta]),
            action_values=np.array([action]),
            quadrature_error=np.array([error]),
            source_vertex_ids=np.array([-1]),
        )
        for port, zeta, action, error in zip(
            sample.ports, (entry, entry, zeta_m), actions, errors
        )
    )
    return replace(
        sample,
        ports=ports,
        marginal_points=np.array([point(zeta_m)]),
        field_line_identity=np.array([[s, alpha]]),
        event_zeta_unwrapped=np.array([[entry, zeta_m, exit]]),
        additivity_residual=np.array([residual]),
    )


def _combine_arc(field, source, samples, period, arc_id, certified, parameters):
    """Form one continuous open arc with its own authoritative PL arc length."""
    points = np.concatenate([item.marginal_points for item in samples])
    # Keep the source's authoritative parameter. Localization inserts samples
    # on that parameter; it must not change its total length as the budget
    # changes. source_u on the arc preserves the unshifted correspondence.
    u = np.asarray(parameters) - parameters[0]
    if np.any(np.diff(u) <= 0):
        raise ValueError("distinct arc samples coincide at an event")
    ports = tuple(
        replace(
            port,
            **{
                name: np.concatenate(
                    [getattr(item.ports[index], name) for item in samples]
                )
                for name in (
                    "points",
                    "zeta_unwrapped",
                    "action_values",
                    "quadrature_error",
                    "source_vertex_ids",
                )
            },
        )
        for index, port in enumerate(source.ports)
    )
    # Singleton localization/refinement traces have their own periodic lifts.
    # Align every port to the continuously lifted marginal arc, preserving
    # each trace's entry/marginal/exit offsets on the same field line.
    old_events = np.concatenate([item.event_zeta_unwrapped for item in samples])
    zeta = np.unwrap(points[:, 2], period=period)
    zeta += period * round((old_events[0, 1] - zeta[0]) / period)
    shift = zeta - old_events[:, 1]
    ports = tuple(
        replace(port, zeta_unwrapped=port.zeta_unwrapped + shift) for port in ports
    )
    identities = np.concatenate([item.field_line_identity for item in samples]).copy()
    iota = np.asarray(field.iota(identities[:, 0]))
    theta = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    first_theta = (
        identities[0, 1]
        + np.ravel(np.broadcast_to(iota, len(points)))[0] * old_events[0, 1]
    )
    theta += 2 * np.pi * round((first_theta - theta[0]) / (2 * np.pi))
    identities[:, 1] = theta - iota * zeta
    return replace(
        source,
        transition_id=arc_id,
        u=u,
        total_u_length=float(u[-1]),
        ports=ports,
        marginal_points=points,
        field_line_identity=identities,
        event_zeta_unwrapped=old_events + shift[:, None],
        additivity_residual=np.concatenate(
            [item.additivity_residual for item in samples]
        ),
        sample_status=tuple(item.sample_status[0] for item in samples),
        sample_failure_reason=tuple(item.sample_failure_reason[0] for item in samples),
        interior_maximum_count=np.concatenate(
            [item.interior_maximum_count for item in samples]
        ),
        barrier_margin=np.concatenate([item.barrier_margin for item in samples]),
        status=TransitionStatus.REGULAR if certified else TransitionStatus.UNRESOLVED,
        contact_sample_pairs=None,
        sampling_samples_used=len(samples),
        authoritative_sample_count=len(samples),
        sampling_certified=certified,
        sampling_unresolved_intervals=(
            None if certified else np.array([[0, len(samples) - 1]])
        ),
        sampling_reason=(
            "regular source arc with one-sided event limits"
            if certified
            else "source arc remains uncertified"
        ),
    )


def _certify_arc_samples(field, critical, polyline, source, values, parameters, budget):
    """Certify one arc using remaining authoritative-vertex work (§10.2).

    Every existing trace is retained. Only original critical-curve vertices
    can consume the remaining source budget; contact-localization traces have
    their separate explicit bisection budget. Unknown intervals and new
    physical failures remain unresolved, independently of other arcs.
    """
    if source.sampling_certified:
        return values, parameters, True, [], "", 0
    low, high = parameters[0], parameters[-1]
    known = dict(zip(map(float, parameters), values))
    candidates = {u: (sample.marginal_points[0], -1) for u, sample in known.items()}
    for index, original_u in enumerate(polyline.u):
        for cycle in (0, 1) if polyline.closed else (0,):
            u = float(original_u + cycle * polyline.total_length)
            if low < u < high:
                candidates[u] = (critical.points[polyline.vertex_ids[index]], index)
    coordinates = np.array(sorted(candidates))
    points = np.array([candidates[u][0] for u in coordinates])
    cache = {index: known[u] for index, u in enumerate(coordinates) if u in known}
    tags = np.array(
        [
            (
                critical.boundary_tags[polyline.vertex_ids[candidates[u][1]]]
                if candidates[u][1] >= 0
                else 0
            )
            for u in coordinates
        ]
    )
    curve = replace(
        polyline,
        vertex_ids=np.arange(len(points)),
        segment_ids=np.empty(0, dtype=np.int64),
        u=coordinates - low,
        total_length=float(high - low),
        closed=False,
    )
    local = replace(
        critical,
        points=points,
        segments=np.empty((0, 2), dtype=np.int64),
        D2_B=np.zeros(len(points)),
        point_kind=np.full(len(points), CriticalKind.GAMMA_MAX.value),
        segment_kind=np.empty(0, dtype=np.int64),
        boundary_tags=tags,
        polylines=(curve,),
    )
    uncertain = []
    for left, right in source.sampling_unresolved_intervals:
        a, b = float(polyline.u[left]), float(polyline.u[right])
        if right <= left:
            b += polyline.total_length
        for cycle in (-1, 0, 1):
            uncertain.append(
                (a + cycle * polyline.total_length, b + cycle * polyline.total_length)
            )
    selected = sorted(cache)
    pending = [
        (a, b)
        for a, b in zip(selected[:-1], selected[1:])
        if _interval_midpoint_index(curve, a, b) is not None
        and (
            not uncertain
            or any(coordinates[a] < y and coordinates[b] > x for x, y in uncertain)
        )
    ]
    added = 0
    failures = []
    while pending:
        interval = max(
            pending,
            key=lambda pair: _interval_priority(local, curve, *pair, source.controls)[
                0
            ],
        )
        pending.remove(interval)
        left, right = interval
        middle = _interval_midpoint_index(curve, left, right)
        if middle is None:
            continue
        if budget[0] is not None and budget[0] <= 0:
            pending.append(interval)
            break
        original_index = candidates[float(coordinates[middle])][1]
        sample = _map_polyline_samples(
            field,
            critical,
            polyline,
            source.transition_id,
            source.controls,
            np.array([original_index]),
        )
        cache[middle] = sample
        if budget[0] is not None:
            budget[0] -= 1
        added += 1
        passed, _, _, reason = _certify_interval(
            local, curve, left, middle, right, cache, source.controls
        )
        if any(
            cache[index].status is not TransitionStatus.REGULAR
            for index in (left, middle, right)
        ):
            failures.append(interval)
            continue
        if (
            len(
                {
                    int(cache[index].interior_maximum_count[0])
                    for index in (left, middle, right)
                }
            )
            > 1
        ):
            failures.append(interval)
            continue
        if not passed:
            pending.extend(
                pair
                for pair in ((left, middle), (middle, right))
                if _interval_midpoint_index(curve, *pair) is not None
            )
    selected = sorted(cache)
    reason = (
        "additional nongeneric or failed sample in arc"
        if failures
        else "source sampling budget exhausted on this arc" if pending else ""
    )
    unresolved = [
        (float(coordinates[a]), float(coordinates[b])) for a, b in failures + pending
    ]
    return (
        [cache[i] for i in selected],
        coordinates[selected].tolist(),
        not unresolved,
        unresolved,
        reason,
        added,
    )


def build_transition_arcs(field, critical, localized):
    """Construct one-sided regular arcs without resolving the event hyperedges.

    Regular intervals retain every source and localized midpoint sample. The
    small final event bracket is replaced by its independently solved endpoint
    and limiting actions; event incidence remains unresolved. Unsupported
    events and source sampling failures cannot be promoted to regular cuts.
    """
    arcs = []
    wells = {}
    for source in localized.transitions:
        polyline = [p for p in critical.polylines if p.kind is CriticalKind.GAMMA_MAX][
            source.transition_id
        ]
        remaining_budget = [
            (
                None
                if source.controls.max_curve_samples is None
                else max(0, source.controls.max_curve_samples - len(source.u))
            )
        ]
        occurrences = sorted(
            [
                (float(np.mean(occurrence.u_interval)), event, occurrence)
                for event in localized.events
                for occurrence in event.occurrences
                if occurrence.source_transition_id == source.transition_id
            ],
            key=lambda item: item[0],
        )
        artifacts = [
            item
            for item in localized.scan_artifacts
            if item.source_transition_id == source.transition_id
        ]
        if not occurrences and not artifacts:
            arcs.append(
                TransitionArc(
                    replace(source, transition_id=len(arcs)),
                    source.transition_id,
                    (-1, -1),
                    (
                        ""
                        if source.status is TransitionStatus.REGULAR
                        else source.sampling_reason
                        + "; "
                        + ", ".join(sorted(set(source.sample_failure_reason)))
                    ),
                )
            )
            continue
        samples = {
            float(u): _single_sample(source, index) for index, u in enumerate(source.u)
        }
        for _, event, occurrence in occurrences:
            for sample in occurrence.samples:
                samples[float(sample.u[0])] = sample
        for artifact in artifacts:
            for sample in artifact.samples:
                samples[float(sample.u[0])] = sample
        # Closed source curves can have a bracket straddling u=total_length.
        length = source.total_u_length
        samples = {
            (u % length if polyline.closed else u): sample
            for u, sample in samples.items()
        }
        occurrences = sorted(
            [(u % length if polyline.closed else u, e, o) for u, e, o in occurrences],
            key=lambda item: item[0],
        )
        if not occurrences:
            rows = sorted(samples.items())
            if polyline.closed:
                rows.append((length, rows[0][1]))
            values, parameters, certified, unresolved, reason, added = (
                _certify_arc_samples(
                    field,
                    critical,
                    polyline,
                    source,
                    [row[1] for row in rows],
                    [row[0] for row in rows],
                    remaining_budget,
                )
            )
            regular = all(value.status is TransitionStatus.REGULAR for value in values)
            same_count = (
                len({int(value.interior_maximum_count[0]) for value in values}) == 1
            )
            certified = certified and regular and same_count
            if not certified and not reason:
                reason = (
                    "unresolved sample or itinerary variation after scan refinement"
                )
            if polyline.closed:
                values, parameters = values[:-1], parameters[:-1]
            curve = _combine_arc(
                field, source, values, critical.period, len(arcs), certified, parameters
            )
            curve = replace(
                curve,
                total_u_length=length,
                sampling_reason=reason
                or "scan aliases removed; regular curve certified",
            )
            if reason == "source sampling budget exhausted on this arc":
                curve = replace(curve, status=TransitionStatus.BUDGET_INSUFFICIENT)
            arcs.append(
                TransitionArc(
                    curve,
                    source.transition_id,
                    (-1, -1),
                    reason,
                    np.asarray(parameters),
                    added,
                    tuple(unresolved),
                    (0.0, length),
                )
            )
            continue
        intervals = list(zip(occurrences[:-1], occurrences[1:]))
        if polyline.closed:
            u, event, occurrence = occurrences[0]
            intervals.append((occurrences[-1], (u + length, event, occurrence)))
        else:
            intervals.insert(0, ((0.0, None, None), occurrences[0]))
            intervals.append((occurrences[-1], (length, None, None)))
        for left, right in intervals:
            low, left_event, left_occurrence = left
            high, right_event, right_occurrence = right
            if high <= low:
                # A sampled event at an open source endpoint has only its
                # interior incident arc, not a fictitious zero-length arc.
                continue
            rows = sorted(
                [
                    (u + k * length, sample)
                    for u, sample in samples.items()
                    for k in (0, 1)
                    if low < u + k * length < high
                ],
                key=lambda item: item[0],
            )
            event_ids = tuple(
                -1 if event is None else event.event_id
                for event in (left_event, right_event)
            )
            added = 0
            unresolved_intervals = []
            parameters = []
            try:
                if not rows:
                    raise ValueError("event interval has no regular arc sample")
                for event, occurrence in (
                    (left_event, left_occurrence),
                    (right_event, right_occurrence),
                ):
                    if event is not None and not occurrence.localized:
                        raise ValueError(
                            "event geometry remains unresolved: " + occurrence.reason
                        )
                values = [sample for _, sample in rows]
                if left_event is not None:
                    if left_event.event_id not in wells:
                        wells[left_event.event_id] = _event_well(
                            field, left_event, critical.period, source.controls
                        )
                    values.insert(
                        0,
                        _event_limit(
                            field,
                            left_event,
                            values[0],
                            critical.period,
                            *wells[left_event.event_id],
                        ),
                    )
                else:
                    values.insert(0, _single_sample(source, 0))
                if right_event is not None:
                    if right_event.event_id not in wells:
                        wells[right_event.event_id] = _event_well(
                            field, right_event, critical.period, source.controls
                        )
                    values.append(
                        _event_limit(
                            field,
                            right_event,
                            values[-1],
                            critical.period,
                            *wells[right_event.event_id],
                        )
                    )
                else:
                    values.append(_single_sample(source, len(source.u) - 1))
                parameters = [low, *[u for u, _ in rows], high]
                values, parameters, sampling_ok, unresolved_intervals, reason, added = (
                    _certify_arc_samples(
                        field,
                        critical,
                        polyline,
                        source,
                        values,
                        parameters,
                        remaining_budget,
                    )
                )
                regular = all(
                    sample.status is TransitionStatus.REGULAR for sample in values
                )
                same_count = (
                    len({int(sample.interior_maximum_count[0]) for sample in values})
                    == 1
                )
                certified = sampling_ok and regular and same_count
                curve = _combine_arc(
                    field,
                    source,
                    values,
                    critical.period,
                    len(arcs),
                    certified,
                    parameters,
                )
                if not certified and not reason:
                    reason = (
                        "source arc contains a failed trace"
                        if not regular
                        else "unexplained interior-maximum count change within arc"
                    )
                if reason == "source sampling budget exhausted on this arc":
                    curve = replace(
                        curve,
                        status=TransitionStatus.BUDGET_INSUFFICIENT,
                        sampling_reason=reason,
                    )
            except (ValueError, FloatingPointError) as error:
                curve = replace(
                    source,
                    transition_id=len(arcs),
                    status=TransitionStatus.UNRESOLVED,
                    sampling_certified=False,
                    sampling_reason=str(error),
                )
                reason = str(error)
                unresolved_intervals = [(low, high)]
            arcs.append(
                TransitionArc(
                    curve,
                    source.transition_id,
                    event_ids,
                    reason,
                    np.asarray(parameters),
                    added,
                    tuple(unresolved_intervals),
                    (low, high),
                )
            )
    return replace(localized, arcs=tuple(arcs))
