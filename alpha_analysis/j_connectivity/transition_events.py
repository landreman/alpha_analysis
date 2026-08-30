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

    def __post_init__(self):
        if self.max_bisections < 1:
            raise ValueError("max_bisections must be positive")
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


@dataclass(frozen=True)
class TransitionEvent:
    """One physical nongeneric event, possibly encountered at several ``u``s.

    ``marginal_points`` contains the distinct simultaneous maxima on one lifted
    field line, reduced to the periodic logical domain. An unresolved bracket
    without an established common event keeps an empty point array instead of
    manufacturing an incidence. ``occurrences`` preserves source provenance.
    """

    event_id: int
    marginal_points: np.ndarray
    occurrences: tuple[ContactBracket, ...]
    unresolved: bool = True
    field_line_identity: np.ndarray | None = None
    zeta_unwrapped: np.ndarray | None = None


@dataclass(frozen=True)
class TransitionArc:
    """A continuous regular correspondence with explicit event endpoints.

    ``curve`` stores one-sided limiting values at an event, not a proposed
    resolution of that event. Event connectivity remains a separate hyperedge.
    ``endpoint_event_ids`` is in the arc's order, with -1 for an ordinary end.
    """

    curve: object
    source_transition_id: int
    endpoint_event_ids: tuple[int, int]
    unresolved_reason: str = ""


@dataclass(frozen=True)
class LocalizedTransitions:
    """Source mappings and explicit localized/unresolved nongeneric events."""

    transitions: tuple
    events: tuple[TransitionEvent, ...]
    arcs: tuple[TransitionArc, ...] = ()


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

        solved = root(residual, [s, theta_m, zeta_m, tau], tol=1.0e-11)
        ss, theta, zeta, offset = solved.x
        if not np.all(np.isfinite(solved.x)) or not 0 < ss <= 1:
            continue
        values = residual(solved.x)
        controls = sample.controls
        if (
            np.max(np.abs(values[[0, 2]])) > controls.root_atol_B
            or np.max(np.abs(values[[1, 3]])) > controls.tangent_atol_B
        ):
            continue
        theta2 = theta + float(field.iota(ss)) * offset
        if (
            abs(offset) <= controls.root_atol_zeta
            or float(field.D2_B(ss, theta, zeta)) >= -controls.D2_tolerance
            or float(field.D2_B(ss, theta2, zeta + offset)) >= -controls.D2_tolerance
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
    for transition in transitions:
        polyline = maxima[transition.transition_id]
        for pair in transition.contact_sample_pairs:
            first_index, second_index = map(int, pair)
            left = _single_sample(transition, first_index)
            right_u = float(transition.u[second_index])
            if second_index < first_index:
                right_u += transition.total_u_length
            right = _single_sample(transition, second_index, u=right_u)
            samples = [left, right]
            reason = "contact localization budget exhausted"
            for _ in range(config.max_bisections):
                if right.u[0] - left.u[0] <= config.u_tolerance:
                    reason = "localized count-change interval"
                    break
                first, second = left.marginal_points[0], right.marginal_points[0]
                middle = first + 0.5 * _periodic_delta(first, second, critical.period)
                point = _projected_midpoint(
                    first,
                    second,
                    middle,
                    field,
                    critical.b,
                    critical.period,
                    CriticalCurveConfig(
                        B_tolerance=transition.controls.root_atol_B,
                        g_tolerance=transition.controls.tangent_atol_B,
                    ),
                )
                if point is None:
                    reason = "critical-curve midpoint projection failed"
                    break
                sample = _map_point(
                    field,
                    critical,
                    polyline,
                    transition,
                    point,
                    0.5 * (left.u[0] + right.u[0]),
                )
                samples.append(sample)
                if sample.status is not TransitionStatus.REGULAR:
                    reason = "midpoint trace: " + sample.sample_failure_reason[0]
                    break
                count = sample.interior_maximum_count[0]
                if count == left.interior_maximum_count[0]:
                    left = sample
                elif count == right.interior_maximum_count[0]:
                    right = sample
                else:
                    reason = "additional interior-maximum count in contact bracket"
                    break
            localized = right.u[0] - left.u[0] <= config.u_tolerance
            geometry = (
                _solve_equal_height(field, left, right, critical.period, config)
                if localized
                else None
            )
            points = None if geometry is None else geometry[0]
            occurrence = ContactBracket(
                transition.transition_id,
                (first_index, second_index),
                np.array([left.u[0], right.u[0]]),
                tuple(samples),
                localized and points is not None,
                (
                    reason
                    if points is not None
                    else reason + "; equal-height event not certified"
                ),
            )
            match = None
            if points is not None:
                for index, event in enumerate(events):
                    if len(event.marginal_points) != len(points):
                        continue
                    distances = np.array(
                        [
                            [
                                np.linalg.norm(_periodic_delta(a, b, critical.period))
                                for b in event.marginal_points
                            ]
                            for a in points
                        ]
                    )
                    if np.all(np.min(distances, axis=1) <= config.event_tolerance):
                        match = index
                        break
            if match is None:
                events.append(
                    TransitionEvent(
                        len(events),
                        np.empty((0, 3)) if points is None else points,
                        (occurrence,),
                        field_line_identity=None if geometry is None else geometry[1],
                        zeta_unwrapped=None if geometry is None else geometry[2],
                    )
                )
            else:
                events[match] = replace(
                    events[match], occurrences=events[match].occurrences + (occurrence,)
                )
    return LocalizedTransitions(tuple(transitions), tuple(events))


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


def _combine_arc(source, samples, period, arc_id, certified):
    """Form one continuous open arc with its own authoritative PL arc length."""
    points = np.concatenate([item.marginal_points for item in samples])
    u = np.concatenate(
        (
            [0.0],
            np.cumsum(
                [
                    np.linalg.norm(_periodic_delta(a, b, period))
                    for a, b in zip(points[:-1], points[1:])
                ]
            ),
        )
    )
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
    return replace(
        source,
        transition_id=arc_id,
        u=u,
        total_u_length=float(u[-1]),
        ports=ports,
        marginal_points=points,
        field_line_identity=np.concatenate(
            [item.field_line_identity for item in samples]
        ),
        event_zeta_unwrapped=np.concatenate(
            [item.event_zeta_unwrapped for item in samples]
        ),
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
        sampling_unresolved_intervals=None,
        sampling_reason=(
            "regular source arc with one-sided event limits"
            if certified
            else "source arc remains uncertified"
        ),
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
        occurrences = sorted(
            [
                (float(np.mean(occurrence.u_interval)), event, occurrence)
                for event in localized.events
                for occurrence in event.occurrences
                if occurrence.source_transition_id == source.transition_id
            ],
            key=lambda item: item[0],
        )
        if not occurrences:
            arcs.append(
                TransitionArc(
                    replace(source, transition_id=len(arcs)),
                    source.transition_id,
                    (-1, -1),
                )
            )
            continue
        samples = {
            float(u): _single_sample(source, index) for index, u in enumerate(source.u)
        }
        for _, event, occurrence in occurrences:
            for sample in occurrence.samples:
                samples[float(sample.u[0])] = sample
        # Closed source curves can have a bracket straddling u=total_length.
        length = source.total_u_length
        samples = {u % length: sample for u, sample in samples.items()}
        occurrences = sorted(
            [(u % length, e, o) for u, e, o in occurrences], key=lambda item: item[0]
        )
        polyline = [p for p in critical.polylines if p.kind is CriticalKind.GAMMA_MAX][
            source.transition_id
        ]
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
                certified = source.sampling_certified and all(
                    sample.status is TransitionStatus.REGULAR for sample in values
                )
                curve = _combine_arc(
                    source, values, critical.period, len(arcs), certified
                )
                reason = (
                    ""
                    if certified
                    else "source arc sampling or trace status remains unresolved"
                )
            except (ValueError, FloatingPointError) as error:
                curve = replace(
                    source, transition_id=len(arcs), status=TransitionStatus.UNRESOLVED
                )
                reason = str(error)
            arcs.append(TransitionArc(curve, source.transition_id, event_ids, reason))
    return replace(localized, arcs=tuple(arcs))
