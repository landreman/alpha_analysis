"""Render pitch surfaces together with their critical curves and transitions.

This is ``plot_pitch_surfaces_pyvista_logical.py`` with the milestone-8 and
milestone-9 structures drawn on top of it.  The same ``BackgroundMesh`` is
built for the same boozmn equilibrium and the same ``B = b`` level surfaces
(DESIGN.md §8.3) are extracted and drawn in the dimensionless logical cylinder
``(x, y, zeta)`` of DESIGN.md §3.2, with the two physical-sign halves of
DESIGN.md §5.1, ``g = B D_parallel B / (G + iota I)``, distinguished by their
triangle-edge color: the incoming half ``g < 0`` in black, the outgoing half
``g > 0`` in red.

For one selected level ``b`` -- one of the surfaces actually drawn -- the
following are added:

* ``Gamma_min`` and ``Gamma_max`` (DESIGN.md §5.2), the two classes of
  ``B = b, g = 0`` boundary curve, in different colors.  Segments that
  classification could not resolve stay explicitly ``DEGENERATE`` in a third
  color rather than being drawn as one of the two regular classes.
* The companion curve ``T`` of DESIGN.md §10.2 in a fourth color.  ``T`` is
  the locus of the incoming crossings ``a(u)`` that precede the marginal
  points ``m(u) in Gamma_max``: it is where the parent well is cut in two, so
  it marks where wells split.
* The trapped-particle path of each sampled parent well, the field-line arc
  from ``a(u)`` through the tangent contact ``m(u)`` to the outgoing crossing
  ``d(u)``.  At fixed ``(s, alpha)`` the poloidal angle is
  ``theta = alpha + iota zeta``, so in logical coordinates the path is a
  helical curve, not the straight chord that merely connecting ``a`` to ``m``
  would suggest.  ``A_1 = A[a,m]``, ``A_3 = A[m,d]`` and ``A_W = A[a,d]`` are
  the actions integrated along exactly these arcs.
* A marker on each sampled ``m(u)``, which is where the parent well splits,
  and one between the samples that bracket an equal-height contact the
  sampling stepped over (DESIGN.md §5.4): the split there is somewhere on the
  polyline between the two, which milestone 10 still owns locating.

Each drawn triple ``(a, m, d)`` is rigidly translated in zeta by the whole
number of periods that puts its marginal point on the drawn ``Gamma_max``
vertex.  A whole-period translation is an exact symmetry of the field, so the
arcs are the physical ones; only their branch of the covering space is chosen
for legibility.  A well longer than ``--max-path-periods`` field periods, and
its point on ``T``, are listed by span and left undrawn rather than clipped;
every nonregular sample is reported by status and failure reason; and the
critical-curve and transition statuses are printed with the counts behind
them.  Each run also re-derives three checks from the drawn geometry itself:
``max(B) - b`` on every drawn arc, the whole-period branch residual, and
whether each arc runs from ``g < 0`` to ``g > 0``.

``--zeta-scale`` stretches the zeta axis for legibility only; it is a viewing
transformation and changes no stored quantity.

PyVista renders and stays outside the numerical core (DESIGN.md §19.2): the
NumPy arrays on ``SurfaceMesh``, ``CriticalCurves`` and ``TransitionCurve``
remain authoritative.

Example::

    python examples/plot_transitions_pyvista_logical.py \
        --n-surfaces 4 --transition-index 2 --n-periods 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_volume_mesh_pyvista_logical import (  # noqa: E402
    DEFAULT_BOOZMN,
    B_LABEL,
    screen_size,
)
from plot_pitch_surfaces_pyvista_logical import to_logical_surface  # noqa: E402

from alpha_analysis.boozer_field import BoozerField  # noqa: E402
from alpha_analysis.j_connectivity.background_mesh import (  # noqa: E402
    BackgroundMeshConfig,
    GmshBackgroundMeshBackend,
    GmshBackgroundMeshConfig,
    StructuredPrismMeshBackend,
)
from alpha_analysis.j_connectivity.critical_curves import (  # noqa: E402
    CriticalCurveStatus,
    CriticalKind,
    extract_critical_curves,
)
from alpha_analysis.j_connectivity.surface_extract import (  # noqa: E402
    MarchingTetrahedraExtractor,
    PyVistaSurfaceExtractor,
    SurfaceExtractionError,
)
from alpha_analysis.j_connectivity.transitions import (  # noqa: E402
    TransitionMappingConfig,
    map_transitions,
)
from alpha_analysis.j_connectivity.types import TransitionStatus  # noqa: E402

# One color per structure. The surfaces themselves are viridis with black
# (incoming) and red (outgoing) triangle edges, as in the plain pitch-surface
# script, so these five stay away from both ends of that map.
GAMMA_MIN_COLOR = "blue"
GAMMA_MAX_COLOR = "magenta"
DEGENERATE_COLOR = "yellow"
T_COLOR = "lime"
PATH_COLOR = "cyan"
MARGINAL_COLOR = "white"
CONTACT_COLOR = "orange"

CRITICAL_COLORS = {
    CriticalKind.GAMMA_MIN: GAMMA_MIN_COLOR,
    CriticalKind.GAMMA_MAX: GAMMA_MAX_COLOR,
    CriticalKind.DEGENERATE: DEGENERATE_COLOR,
}


def polyline_zeta(curves, polyline) -> np.ndarray:
    """Return drawn logical zeta for one critical polyline's vertices.

    The stored zeta lies in ``[0, period)``, so a curve that crosses the seam
    would otherwise be drawn jumping across the whole period. The vertices are
    unwrapped in polyline order into the covering space of DESIGN.md §8.1 and
    then translated by whole periods so the curve's mean zeta is in
    ``[0, period)``. Whole-period translation is exact, not a fit.
    """
    ids = np.asarray(polyline.vertex_ids, dtype=np.int64)
    zeta = np.unwrap(
        np.asarray(curves.points[ids, 2], dtype=float), period=curves.period
    )
    return zeta - curves.period * np.floor(np.mean(zeta) / curves.period)


def polyline_polydata(curves, polyline, zeta_scale: float) -> pv.PolyData:
    """Return one critical polyline as a PyVista line, seam unwrapped."""
    ids = np.asarray(polyline.vertex_ids, dtype=np.int64)
    points = np.asarray(curves.points[ids], dtype=float).copy()
    points[:, 2] = polyline_zeta(curves, polyline)
    if polyline.closed and len(points) > 2:
        closing = points[0].copy()
        difference = closing[2] - points[-1, 2]
        closing[2] -= curves.period * np.round(difference / curves.period)
        points = np.vstack((points, closing))
    points[:, 2] *= zeta_scale
    return pv.lines_from_points(points)


def broken_polyline(points: np.ndarray, breaks: np.ndarray) -> pv.PolyData | None:
    """Return a line through ``points``, omitting segments marked in ``breaks``.

    ``breaks[i]`` suppresses the segment from sample ``i`` to sample ``i+1``.
    A suppressed segment is a sample the mapping could not resolve or a jump
    of one field period between adjacent samples: it is left out rather than
    bridged, so no drawn line claims a correspondence that was not computed.
    """
    cells: list[int] = []
    for index in range(len(points) - 1):
        if breaks[index]:
            continue
        cells.extend((2, index, index + 1))
    if not cells:
        return None
    line = pv.PolyData(points)
    line.lines = np.asarray(cells, dtype=np.int64)
    return line


def as_tube(line: pv.PolyData, radius: float) -> pv.PolyData:
    """Return real tube geometry for a curve that lies on a pitch surface.

    A critical curve and ``T`` are coplanar with the surface they live on, so
    a flat line renders as a stipple of z-fighting fragments. A tube of
    nonzero radius pokes out of the surface on both sides and is drawn.
    """
    return line.tube(radius=radius, n_sides=12)


def as_spheres(points: np.ndarray, radius: float) -> pv.PolyData:
    """Return real sphere geometry at ``points``, for the same reason."""
    return pv.PolyData(np.asarray(points, dtype=float)).glyph(
        geom=pv.Sphere(radius=radius, theta_resolution=16, phi_resolution=16),
        scale=False,
        orient=False,
    )


def field_line_arc(
    field,
    s: float,
    alpha: float,
    zeta_start: float,
    zeta_end: float,
    period: float,
    zeta_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the logical points and ``|B|`` of one field-line arc.

    The arc is at fixed normalized flux ``s`` and fixed straight-field-line
    label ``alpha``, so ``theta = alpha + iota(s) zeta`` (DESIGN.md §3.2) and
    zeta parameterizes it monotonically. Sampling is proportional to the
    zeta span so long wells are not drawn more coarsely than short ones.

    ``zeta_offset`` is a whole number of field periods and moves only the
    drawn zeta. ``B`` is periodic in zeta at fixed theta, so holding theta
    fixed and translating zeta is the exact symmetry that carries the well to
    another period; recomputing theta from the shifted zeta instead would
    walk a *different* field line and the returned ``B`` would no longer be
    the well's.
    """
    span = abs(zeta_end - zeta_start)
    n_samples = int(max(64, np.ceil(256.0 * span / period)))
    zeta = np.linspace(zeta_start, zeta_end, n_samples)
    iota = float(np.asarray(field.iota(s)).ravel()[0])
    theta = alpha + iota * zeta
    rho = float(np.sqrt(s))
    points = np.column_stack(
        (rho * np.cos(theta), rho * np.sin(theta), zeta + zeta_offset)
    )
    B = np.asarray(field.B(np.full_like(zeta, s), theta, zeta), dtype=float)
    return points, B


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boozmn", type=Path, default=DEFAULT_BOOZMN)
    parser.add_argument(
        "--backend", choices=("structured", "gmsh"), default="structured"
    )
    parser.add_argument(
        "--extractor",
        choices=("marching", "pyvista"),
        default="marching",
        help="surface-extraction method: marching tetrahedra or VTK contour",
    )
    parser.add_argument(
        "--n-surfaces",
        type=int,
        default=4,
        help="number of pitch surfaces strictly between min(B) and max(B)",
    )
    parser.add_argument(
        "--transition-index",
        type=int,
        default=None,
        help="which drawn surface carries the critical/transition overlay; "
        "the default is the lowest b that carries both Gamma_min and "
        "Gamma_max, falling back to the most Gamma_max vertices",
    )
    parser.add_argument(
        "--max-curve-samples",
        type=int,
        default=12,
        help="adaptive mapping work budget per Gamma_max polyline; an "
        "uncertified curve remains explicitly budget-insufficient. 0 maps "
        "every authoritative critical-curve vertex",
    )
    parser.add_argument(
        "--max-path-periods",
        type=float,
        default=4.0,
        help="do not draw a well, or its point on T, whose zeta span "
        "exceeds this many field periods; skipped wells are listed by span, "
        "never clipped",
    )
    parser.add_argument("--n-radial", type=int, default=6)
    parser.add_argument("--n-poloidal", type=int, default=24)
    parser.add_argument("--n-zeta", type=int, default=12)
    parser.add_argument(
        "--target-size", type=float, default=0.25, help="gmsh backend edge length"
    )
    parser.add_argument(
        "--n-periods",
        type=int,
        default=1,
        help="field periods to draw; copies are translations in logical zeta",
    )
    parser.add_argument(
        "--zeta-scale",
        type=float,
        default=2.0,
        help="stretch the logical zeta axis for legibility only",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.28,
        help="face opacity of the surface carrying the overlay",
    )
    parser.add_argument(
        "--other-opacity",
        type=float,
        default=0.07,
        help="face opacity of the remaining pitch surfaces",
    )
    parser.add_argument(
        "--tube-radius",
        type=float,
        default=0.008,
        help="drawn radius of the overlay curves, in logical units; the "
        "curves lie on the surfaces, so they need thickness to be visible",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="render window size in pixels; the default fills the screen",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="write a PNG off screen instead of opening a window",
    )
    return parser.parse_args()


def build_background(args: argparse.Namespace, field):
    if args.backend == "gmsh":
        return GmshBackgroundMeshBackend(
            GmshBackgroundMeshConfig(target_size=args.target_size)
        ).build(field)
    return StructuredPrismMeshBackend(
        BackgroundMeshConfig(args.n_radial, args.n_poloidal, args.n_zeta)
    ).build(field)


def kind_vertex_count(curves, kind: CriticalKind) -> int:
    return sum(
        len(polyline.vertex_ids)
        for polyline in curves.polylines
        if polyline.kind is kind
    )


def default_overlay(critical: dict) -> tuple[int, str]:
    """Pick the level to overlay, and say why.

    A ``Gamma_max`` polyline is what milestone 9 maps, so a level without one
    has no transition to draw. Among those, a level that also carries
    ``Gamma_min`` shows both classes of §5.2 boundary curve at once, which is
    the more informative picture; ties go to the lowest ``b``.
    """
    with_max = [
        index
        for index in sorted(critical)
        if kind_vertex_count(critical[index], CriticalKind.GAMMA_MAX)
    ]
    if not with_max:
        return min(critical), "no level has a Gamma_max curve"
    with_both = [
        index
        for index in with_max
        if kind_vertex_count(critical[index], CriticalKind.GAMMA_MIN)
    ]
    if with_both:
        return with_both[0], "lowest b carrying both Gamma_min and Gamma_max"
    return (
        max(
            with_max,
            key=lambda index: kind_vertex_count(
                critical[index], CriticalKind.GAMMA_MAX
            ),
        ),
        "no level carries both classes; most Gamma_max vertices",
    )


def add_surfaces(plotter, args, extractions, clim, overlay: int) -> tuple[float, ...]:
    """Draw every extracted pitch surface, incoming black and outgoing red.

    The surface carrying the overlay keeps ``--opacity``; the others are
    drawn at ``--other-opacity`` so they give context without hiding the
    curves.
    """
    first = True
    bounds = np.array([np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf])
    for index, (_, extraction) in enumerate(extractions):
        opacity = args.opacity if index == overlay else args.other_opacity
        for half, edge_color in (
            (extraction.incoming, "black"),
            (extraction.outgoing, "red"),
        ):
            mesh = to_logical_surface(half, args.zeta_scale)
            if mesh is None:
                continue
            for period in range(args.n_periods):
                shift = args.zeta_scale * half.period * period
                copy = mesh.translate((0.0, 0.0, shift), inplace=False)
                bounds[0::2] = np.minimum(bounds[0::2], copy.bounds[0::2])
                bounds[1::2] = np.maximum(bounds[1::2], copy.bounds[1::2])
                plotter.add_mesh(
                    copy,
                    scalars=B_LABEL,
                    cmap="viridis",
                    clim=clim,
                    opacity=opacity,
                    show_edges=True,
                    edge_color=edge_color,
                    line_width=1,
                    show_scalar_bar=first,
                    scalar_bar_args={"title": "|B| [T]"} if first else None,
                )
                first = False
    return tuple(bounds)


def add_periodic(plotter, mesh, args, period: float, **kwargs) -> None:
    """Add one overlay mesh once per drawn field period.

    Overlay actors are unlit: a shaded tube reads as a different color than
    the one the legend names, and these colors are the classification.
    """
    for index in range(args.n_periods):
        shift = args.zeta_scale * period * index
        plotter.add_mesh(
            mesh.translate((0.0, 0.0, shift), inplace=False),
            lighting=False,
            **kwargs,
        )


def add_critical_curves(plotter, args, curves) -> set[CriticalKind]:
    """Draw every classified critical polyline; return the kinds drawn.

    A whole ``DEGENERATE`` polyline is drawn like the other two classes. A
    degenerate *point* between a ``Gamma_min`` arm and a ``Gamma_max`` arm is
    isolated -- it belongs to no degenerate polyline -- so it is drawn as a
    marker of its own. It is the reason a slice reports ``DEGENERATE``, and
    leaving it out would make that status look like a contradiction.
    """
    drawn: set[CriticalKind] = set()
    drawn_zeta: dict[int, float] = {}
    for polyline in curves.polylines:
        for vertex, zeta in zip(
            np.asarray(polyline.vertex_ids, dtype=np.int64),
            polyline_zeta(curves, polyline),
        ):
            drawn_zeta.setdefault(int(vertex), float(zeta))
        if len(polyline.vertex_ids) < 2:
            continue
        add_periodic(
            plotter,
            as_tube(
                polyline_polydata(curves, polyline, args.zeta_scale),
                args.tube_radius,
            ),
            args,
            curves.period,
            color=CRITICAL_COLORS[polyline.kind],
        )
        drawn.add(polyline.kind)

    degenerate = np.flatnonzero(curves.point_kind == CriticalKind.DEGENERATE.value)
    if len(degenerate):
        points = np.asarray(curves.points[degenerate], dtype=float).copy()
        # Use the zeta branch the curve through the point was drawn on, so the
        # marker sits on the curve rather than a period away from it.
        points[:, 2] = [
            drawn_zeta.get(int(vertex), float(curves.points[vertex, 2]))
            for vertex in degenerate
        ]
        points[:, 2] *= args.zeta_scale
        add_periodic(
            plotter,
            as_spheres(points, 2.5 * args.tube_radius),
            args,
            curves.period,
            color=DEGENERATE_COLOR,
        )
        drawn.add(CriticalKind.DEGENERATE)
    return drawn


def sample_offsets(curves, polyline, transition) -> tuple[np.ndarray, float]:
    """Return the whole-period zeta shift that draws each sample's well.

    The trace lifts zeta out of ``[0, period)``; the drawn ``Gamma_max``
    polyline lifts it independently. The difference is a whole number of
    periods, and translating a well by whole periods is an exact symmetry of
    the field. The largest residual after rounding is returned so a
    disagreement that is *not* a whole period cannot pass unnoticed.
    """
    drawn_zeta = polyline_zeta(curves, polyline)
    lookup = dict(zip(np.asarray(polyline.vertex_ids, dtype=np.int64), drawn_zeta))
    source_ids = next(
        port.source_vertex_ids for port in transition.ports if port.role == "child_3"
    )
    lifted = np.asarray(transition.event_zeta_unwrapped[:, 1], dtype=float)
    raw = np.array([lookup[int(vertex)] for vertex in source_ids]) - lifted
    offsets = curves.period * np.round(raw / curves.period)
    residual = float(np.max(np.abs(raw - offsets))) if len(raw) else 0.0
    return offsets, residual


def add_transition(plotter, args, field, curves, polyline, transition) -> dict:
    """Draw ``T``, the sampled wells, and the split points of one transition.

    Returns the diagnostics that the caller prints: they are what makes the
    picture checkable rather than merely plausible.
    """
    period = curves.period
    offsets, offset_residual = sample_offsets(curves, polyline, transition)
    parent = next(port for port in transition.ports if port.role == "parent")
    regular = np.array(
        [status is TransitionStatus.REGULAR for status in transition.sample_status]
    )
    finite = np.all(np.isfinite(parent.points), axis=1)
    valid = regular & finite

    # A well may be many field periods long. Such a well and its point on T
    # are real, but drawing them puts the structure being looked at into a
    # corner of the frame, so they are gated by --max-path-periods and listed
    # in the report instead of being clipped or quietly dropped.
    zeta_a = np.asarray(transition.event_zeta_unwrapped[:, 0], dtype=float)
    zeta_d = np.asarray(transition.event_zeta_unwrapped[:, 2], dtype=float)
    span = np.abs(zeta_d - zeta_a)
    short = span <= args.max_path_periods * period
    drawable = valid & short
    skipped = [
        (int(index), float(span[index] / period))
        for index in np.flatnonzero(valid & ~short)
    ]

    # T: the incoming crossings a(u), drawn at the same branch as their well.
    T_points = np.asarray(parent.points, dtype=float).copy()
    T_points[:, 2] = np.asarray(parent.zeta_unwrapped, dtype=float) + offsets
    T_points[~drawable] = 0.0
    T_points[:, 2] *= args.zeta_scale
    # Break T where a sample was not drawn, and where two adjacent samples
    # were drawn on different branches -- a period jump is not a piece of T.
    branch_jump = np.abs(np.diff(offsets)) > 0.5 * period
    breaks = (~drawable[:-1]) | (~drawable[1:]) | branch_jump
    T_line = broken_polyline(T_points, breaks)
    if T_line is not None:
        add_periodic(
            plotter,
            as_tube(T_line, args.tube_radius),
            args,
            period,
            color=T_COLOR,
        )

    # The trapped-particle paths a -> m -> d, as field-line arcs.
    drawn_paths = 0
    max_B_excess = -np.inf
    wrong_half = 0
    for index in np.flatnonzero(drawable):
        s, alpha = transition.field_line_identity[index]
        points, B = field_line_arc(
            field,
            float(s),
            float(alpha),
            zeta_a[index],
            zeta_d[index],
            period,
            offsets[index],
        )
        max_B_excess = max(max_B_excess, float(np.max(B)) - transition.b)
        # The well runs from the incoming half to the outgoing half: with
        # g = B D_parallel B / (G + iota I) of DESIGN.md §5.1, g(a) < 0 and
        # g(d) > 0. Anything else means the drawn arc is not a well.
        iota = float(np.asarray(field.iota(s)).ravel()[0])
        ends = np.array([zeta_a[index], zeta_d[index]])
        g_ends = (
            np.asarray(field.B(np.full(2, s), alpha + iota * ends, ends), dtype=float)
            * np.asarray(
                field.D_B(np.full(2, s), alpha + iota * ends, ends), dtype=float
            )
            / float(np.asarray(field.C(s)).ravel()[0])
        )
        if not (g_ends[0] < 0.0 < g_ends[1]):
            wrong_half += 1
        points[:, 2] *= args.zeta_scale
        arc = pv.lines_from_points(points)
        add_periodic(
            plotter,
            as_tube(arc, 0.7 * args.tube_radius),
            args,
            period,
            color=PATH_COLOR,
        )
        drawn_paths += 1

    # The marginal points m(u): each is where the parent well splits in two.
    marginal = np.asarray(transition.marginal_points, dtype=float).copy()
    marginal[:, 2] = np.asarray(transition.event_zeta_unwrapped[:, 1]) + offsets
    marginal[:, 2] *= args.zeta_scale
    if np.any(drawable):
        add_periodic(
            plotter,
            as_spheres(marginal[drawable], 2.5 * args.tube_radius),
            args,
            period,
            color=MARGINAL_COLOR,
        )

    # Equal-height contacts the sampling stepped over (DESIGN.md §5.4): the
    # split is somewhere on the polyline between these two samples.
    contacts = np.asarray(transition.contact_sample_pairs, dtype=np.int64)
    if len(contacts):
        midpoints = 0.5 * (marginal[contacts[:, 0]] + marginal[contacts[:, 1]])
        add_periodic(
            plotter,
            as_spheres(midpoints, 3.5 * args.tube_radius),
            args,
            period,
            color=CONTACT_COLOR,
        )

    residual = np.asarray(transition.additivity_residual, dtype=float)[valid]
    return {
        "transition_id": transition.transition_id,
        "status": transition.status.name,
        "n_samples": len(transition.u),
        "n_regular": int(np.count_nonzero(valid)),
        "n_paths": drawn_paths,
        "wrong_half": wrong_half,
        "skipped": skipped,
        "contacts": contacts.tolist(),
        "offset_residual": offset_residual,
        "max_B_excess": max_B_excess,
        "max_additivity_residual": (
            float(np.max(np.abs(residual))) if len(residual) else float("nan")
        ),
        "failures": sorted(
            {
                f"{status.name}:{reason}"
                for status, reason in zip(
                    transition.sample_status, transition.sample_failure_reason
                )
                if status is not TransitionStatus.REGULAR
            }
        ),
    }


def main() -> None:
    args = parse_arguments()
    if args.n_surfaces < 1:
        raise SystemExit("--n-surfaces must be at least 1")
    field = BoozerField.from_boozmn(args.boozmn)
    if not 1 <= args.n_periods <= field.nfp:
        raise SystemExit(f"--n-periods must be between 1 and nfp = {field.nfp}")
    background = build_background(args, field)

    B_min, B_max = float(background.B.min()), float(background.B.max())
    # The endpoints are excluded: B = min(B) and B = max(B) are attained at
    # isolated points, not on a surface.
    levels = np.linspace(B_min, B_max, args.n_surfaces + 2)[1:-1]
    print(f"Bmin: {B_min:.6g}, Bmax: {B_max:.6g}")
    extractor = (
        PyVistaSurfaceExtractor()
        if args.extractor == "pyvista"
        else MarchingTetrahedraExtractor()
    )

    extractions = []
    critical = {}
    for level in levels:
        try:
            extraction = extractor.extract(background, field, level)
        except SurfaceExtractionError as error:
            print(f"b = {level:.6g}: {error.status.name}: {error}")
            continue
        index = len(extractions)
        extractions.append((level, extraction))
        curves = extract_critical_curves(extraction, field, float(level))
        critical[index] = curves
        lambda_n = (level - B_min) / (B_max - B_min)
        kinds = ", ".join(
            f"{polyline.kind.name}({len(polyline.vertex_ids)})"
            for polyline in curves.polylines
        )
        print(
            f"[{index}] b = {level:.6g} (lambda_n = {lambda_n:.3f}): "
            f"surface {extraction.status.name}, critical {curves.status.name}: "
            f"{kinds or 'no critical curves'}"
        )
    if not extractions:
        raise SystemExit("no pitch surface could be extracted")

    if args.transition_index is None:
        overlay, reason = default_overlay(critical)
        print(f"overlay chosen automatically: {reason}")
    else:
        overlay = args.transition_index
        if not 0 <= overlay < len(extractions):
            raise SystemExit(
                f"--transition-index must be between 0 and {len(extractions) - 1}"
            )
    b = float(extractions[overlay][0])
    curves = critical[overlay]
    print(f"overlay: [{overlay}] b = {b:.7g}, critical status {curves.status.name}")
    if curves.status is not CriticalCurveStatus.REGULAR:
        # The status is not decoration: say which counts produced it, so a
        # picture drawn on an unresolved slice cannot be read as a clean one.
        report = curves.report
        print(
            f"    degenerate points "
            f"{int(np.count_nonzero(curves.point_kind == CriticalKind.DEGENERATE.value))}"
            f", degenerate segments "
            f"{int(np.count_nonzero(curves.segment_kind == CriticalKind.DEGENERATE.value))}"
            f", unresolved segments {report.unresolved_segment_count}"
            f", degenerate solve failures {report.degenerate_solve_failure_count}"
            f", unresolved endpoints {report.unresolved_endpoint_count}"
            f", source unresolved splits {report.source_unresolved_split_count}"
        )

    config = TransitionMappingConfig(
        max_curve_samples=args.max_curve_samples if args.max_curve_samples > 0 else None
    )
    maxima = [
        polyline
        for polyline in curves.polylines
        if polyline.kind is CriticalKind.GAMMA_MAX
    ]
    transitions = map_transitions(field, curves, config) if maxima else ()

    off_screen = args.screenshot is not None
    window_size = list(args.window_size) if args.window_size else list(screen_size())
    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
    # Without depth peeling the translucent surfaces are composited after the
    # curves and hide the parts of them that lie inside a surface, which is
    # most of the overlay: the curves live on the surfaces.
    plotter.enable_depth_peeling(number_of_peels=12)
    surface_bounds = add_surfaces(plotter, args, extractions, (B_min, B_max), overlay)
    drawn_kinds = add_critical_curves(plotter, args, curves)

    reports = [
        add_transition(plotter, args, field, curves, polyline, transition)
        for polyline, transition in zip(maxima, transitions)
    ]
    for report in reports:
        print(
            f"transition {report['transition_id']}: {report['status']}, "
            f"{report['n_regular']} of {report['n_samples']} samples regular, "
            f"{report['n_paths']} wells drawn"
        )
        print(
            f"    max |A_W - A_1 - A_3| = {report['max_additivity_residual']:.3e}, "
            f"max (B - b) on the drawn arcs = {report['max_B_excess']:.3e}, "
            f"max branch residual = {report['offset_residual']:.3e}"
        )
        if report["wrong_half"]:
            print(
                f"    WARNING: {report['wrong_half']} drawn arcs do not run "
                "from g < 0 to g > 0; they are not wells between the halves"
            )
        if report["contacts"]:
            print(
                "    equal-height contacts bracketed between samples "
                f"(DESIGN.md §5.4): {report['contacts']}"
            )
        if report["skipped"]:
            spans = ", ".join(
                f"sample {index} spans {span:.1f} periods"
                for index, span in report["skipped"]
            )
            print(f"    wells not drawn (longer than --max-path-periods): {spans}")
        if report["failures"]:
            print(f"    nonregular samples: {', '.join(report['failures'])}")

    legend: list[list[str]] = []
    for kind in (
        CriticalKind.GAMMA_MIN,
        CriticalKind.GAMMA_MAX,
        CriticalKind.DEGENERATE,
    ):
        if kind in drawn_kinds:
            label = (
                "DEGENERATE (curve or junction point)"
                if kind is CriticalKind.DEGENERATE
                else kind.name
            )
            legend.append([label, CRITICAL_COLORS[kind]])
    if reports:
        legend.append(["T (well splits here)", T_COLOR])
        legend.append(["trapped-particle path a-m-d", PATH_COLOR])
        legend.append(["marginal point m", MARGINAL_COLOR])
        if any(report["contacts"] for report in reports):
            legend.append(["bracketed equal-height contact", CONTACT_COLOR])
    legend.append(["incoming g < 0 edges", "black"])
    legend.append(["outgoing g > 0 edges", "red"])
    plotter.add_legend(legend, bcolor="grey", size=(0.22, 0.22))

    plotter.add_axes(xlabel="x", ylabel="y", zlabel="zeta")
    plotter.add_text(
        f"{args.boozmn.name}\n{args.backend} background mesh, "
        f"{args.extractor} extraction: {len(extractions)} of "
        f"{len(levels)} pitch surfaces in "
        f"[{levels[0]:.4g}, {levels[-1]:.4g}], "
        f"{args.n_periods} of {field.nfp} field periods\n"
        f"overlay at b = {b:.7g} (critical {curves.status.name}): "
        "critical curves, T, and "
        f"{sum(report['n_paths'] for report in reports)} trapped-particle paths\n"
        "logical coordinates, zeta stretched by "
        f"{args.zeta_scale:g} for legibility",
        font_size=8,
    )
    # Frame the surfaces, not the overlay: a well several field periods long
    # is real and is drawn in full, but letting it set the bounds would
    # shrink everything else to a sliver. If no surface has triangles there
    # is nothing to frame, and PyVista's own fit is used.
    if not np.all(np.isfinite(surface_bounds)):
        surface_bounds = None
    plotter.show_bounds(
        bounds=surface_bounds,
        xtitle="x",
        ytitle="y",
        ztitle=f"{args.zeta_scale:g} x zeta [rad]",
        location="outer",
        grid="back",
    )
    plotter.view_isometric()
    plotter.reset_camera(bounds=surface_bounds)
    # Tell PyVista the camera is chosen, or show() resets it to fit every
    # actor and the framing above is lost.
    plotter.camera_set = True
    if off_screen:
        plotter.show(screenshot=args.screenshot)
        print(f"wrote {args.screenshot}")
    else:
        plotter.show(window_size=window_size)


if __name__ == "__main__":
    main()
