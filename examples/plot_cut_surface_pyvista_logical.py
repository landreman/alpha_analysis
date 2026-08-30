"""Render the milestone-10 constrained cut: sheets, cut seam, and split wells.

This is the milestone-10 companion to
``plot_transitions_pyvista_logical.py``. The same ``BackgroundMesh`` is built
for the same boozmn equilibrium, one ``B = b`` level surface (DESIGN.md §8.3)
is extracted, its critical curves are classified (§5.2), its transitions are
mapped (§10.2), and the incoming half ``g < 0`` is then *cut* along the
companion curve ``T`` by ``cut_surface_at_transitions`` (§§10.3--10.5). What is
drawn is the ``CutSurface`` that comes out, in the dimensionless logical
cylinder ``(x, y, zeta)`` of DESIGN.md §3.2.

The picture is built around the two things the cut produces:

* **Sheets.** Every triangle is colored by its cell-located ``sheet_id``, the
  union-find label of §10.5, and each sheet is labeled at its own centroid.
  Where a transition was resolved, the sheets on the two sides of ``T`` are
  distinct labels even though they meet along coincident geometry: that
  separation *is* the cut. Where no transition could be resolved, the labels
  are the ordinary connected components of the surface, which is what the
  algorithm is entitled to claim there.
* **A continuous on each sheet.** ``--color-by action`` colors the same mesh by
  the branch-specific half-bounce action ``A``. Along ``T`` the parent value
  and the child-1 value disagree by a finite jump; after the cut the two values
  live on two *different* vertices at the same point, so no triangle spans the
  jump. The run re-derives that from the drawn object rather than asserting it.

On top of the sheets it draws, for each transition:

* the authoritative duplicated cut edges (``CutSurface.cut_edges``) as a wide
  translucent seam -- both copies are coincident, so one tube is the honest
  drawing of them;
* the two duplicated ``T`` vertices at every mapped sample: a wide
  translucent parent sphere around a small solid child-1 sphere, because they
  are two distinct vertex IDs at one point and that duplication is the cut.
  The inserted ``GAMMA_MAX`` samples that carry child-3 are drawn as a curve
  of their own;
* the wells that the transition splits. At each drawn sample the parent well
  ``W = [a, d]`` is a wide translucent field-line tube and the two children
  ``w_1 = [a, m]`` and ``w_3 = [m, d]`` are thin solid tubes of their own
  colors inside it, meeting at the marginal point ``m`` where the parent well
  is pinched in two. ``A_W``, ``A_1`` and ``A_3`` are the actions integrated
  along exactly these arcs, and the port a well belongs to is the port that
  carries that action onto its sheet.

Each drawn triple ``(a, m, d)`` is rigidly translated in zeta by the whole
number of periods that puts its marginal point on the drawn ``GAMMA_MAX``
vertex; a whole-period translation is an exact symmetry of the field, so the
arcs are physical and only their branch of the covering space is chosen for
legibility. The inserted curves themselves are drawn where their vertices
actually sit in the cut mesh, so a well's ``a`` may be drawn one period away
from the drawn ``T`` it belongs to.

A transition the cut refused still gets its wells and its mapped ``T`` and
``GAMMA_MAX`` samples drawn -- that geometry is what milestone 9 established
and the mesh's inability to represent the cut does not retract it -- but with
no seam and no child-1 vertex, so the duplication that did not happen is
visibly absent rather than merely unmentioned. On a real equilibrium this is
the usual outcome: the coarse matrix in
``docs/validation/milestone10-real-equilibria.md`` resolves no cut at all.

Everything the cut left explicit is printed rather than hidden: unresolved
transitions with the reason the cut refused them, per-sample transition
statuses and failure reasons, wells longer than ``--max-path-periods`` listed
by span instead of clipped, and non-finite action counts. The run also
re-derives milestone 10's acceptance criteria from the drawn object -- no
triangle spanning a parent/child-1 action jump, every port incident to its own
sheet, port actions equal to the cut mesh's own values, duplicated ``T``
vertices distinct and coincident, and a pickle-free serialization round trip --
and prints the coarse NetworkX sheet/transition graph of §10.5.

``--zeta-scale`` stretches the zeta axis for legibility only; it is a viewing
transformation and changes no stored quantity.

PyVista renders and stays outside the numerical core (DESIGN.md §19.2): the
NumPy arrays on ``CutSurface``, ``CriticalCurves`` and ``TransitionCurve``
remain authoritative.

Example::

    python examples/plot_cut_surface_pyvista_logical.py \
        --lambda-n 0.5 --wells-per-transition 3 --show-before
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_volume_mesh_pyvista_logical import (  # noqa: E402
    DEFAULT_BOOZMN,
    screen_size,
)
from plot_transitions_pyvista_logical import (  # noqa: E402
    as_spheres,
    as_tube,
    field_line_arc,
)

from alpha_analysis.boozer_field import BoozerField  # noqa: E402
from alpha_analysis.j_connectivity.background_mesh import (  # noqa: E402
    BackgroundMeshConfig,
    GmshBackgroundMeshBackend,
    GmshBackgroundMeshConfig,
    StructuredPrismMeshBackend,
)
from alpha_analysis.j_connectivity.critical_curves import (  # noqa: E402
    CriticalKind,
    extract_critical_curves,
)
from alpha_analysis.j_connectivity.denominator import (  # noqa: E402
    BoundsConfig,
    find_global_B_bounds,
)
from alpha_analysis.j_connectivity.mesh_cut import (  # noqa: E402
    ConstrainedCutConfig,
    cut_surface_at_transitions,
    load_cut_surface,
    save_cut_surface,
)
from alpha_analysis.j_connectivity.surface_data import (  # noqa: E402
    evaluate_surface_data,
)
from alpha_analysis.j_connectivity.surface_extract import (  # noqa: E402
    MarchingTetrahedraExtractor,
    PyVistaSurfaceExtractor,
    SurfaceExtractionError,
)
from alpha_analysis.j_connectivity.surface_refine import (  # noqa: E402
    SurfaceDownsamplingConfig,
    downsample_surface,
)
from alpha_analysis.j_connectivity.synthetic_fields import (  # noqa: E402
    SyntheticFourierField,
)
from alpha_analysis.j_connectivity.transitions import (  # noqa: E402
    TransitionMappingConfig,
    map_transitions,
)
from alpha_analysis.j_connectivity.types import TransitionStatus  # noqa: E402
from alpha_analysis.j_connectivity.well_trace import WellTraceConfig  # noqa: E402

ACTION_LABEL = "A action length [length]"
SHEET_LABEL = "sheet ID [integer]"

# Sheet colors are categorical: a sheet ID is a union-find label, not a
# magnitude, so a sequential colormap would invent an ordering between sheets
# that the algorithm does not assert. The palette cycles if a slice has more
# sheets than colors, and the printed table remains authoritative.
SHEET_PALETTE = (
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#b279a2",
    "#9d755d",
    "#ff9da6",
    "#72b7b2",
    "#eeca3b",
    "#bab0ac",
    "#1b9e77",
    "#7570b3",
)

CUT_SEAM_COLOR = "red"
T_PARENT_COLOR = "lime"
T_CHILD_COLOR = "deepskyblue"
GAMMA_MAX_COLOR = "magenta"
WELL_PARENT_COLOR = "white"
WELL_1_COLOR = "cyan"
WELL_3_COLOR = "orange"
MARGINAL_COLOR = "white"
CROSSING_COLOR = "gold"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boozmn", type=Path, default=DEFAULT_BOOZMN)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="use the analytic generic-split field of "
        "test_production_synthetic_surface_has_no_uncut_action_jump instead "
        "of a boozmn equilibrium. The coarse real matrix in "
        "docs/validation/milestone10-real-equilibria.md resolves no cut at "
        "all -- every real transition there stays an explicit unresolved "
        "hyperedge -- so this is the case in which the duplication, the "
        "three sheets and the branch actions are actually visible. It "
        "defaults to b = 1.4 and a (4, 16, 12) structured background",
    )
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
        "--lambda-n",
        type=float,
        default=0.5,
        help="pitch level as b = B_min + lambda_n (B_max - B_min), with the "
        "radially global refined bounds of DESIGN.md §2",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=None,
        help="pitch level in field units; overrides --lambda-n",
    )
    parser.add_argument("--n-radial", type=int, default=None)
    parser.add_argument("--n-poloidal", type=int, default=None)
    parser.add_argument("--n-zeta", type=int, default=None)
    parser.add_argument(
        "--target-size", type=float, default=0.25, help="gmsh backend edge length"
    )
    parser.add_argument(
        "--max-curve-samples",
        type=int,
        default=10,
        help="samples per GAMMA_MAX polyline passed to map_transitions; "
        "0 keeps every critical-curve vertex, at a cost that can run to "
        "tens of minutes on a real equilibrium",
    )
    parser.add_argument(
        "--downsample-reduction",
        type=float,
        default=0.0,
        help="optional topology-preserving triangle reduction applied BEFORE "
        "the cut, to bound the cost of the surface-wide well traces. It is "
        "not free: coarsening can move T out of its local triangle "
        "neighborhood and turn a resolvable transition into an explicit "
        "unresolved one. Coarsening after a cut is forbidden (§8.4)",
    )
    parser.add_argument(
        "--no-actions",
        action="store_true",
        help="skip the surface-wide well traces and cut with NaN action. The "
        "sheets and the cut seam are still authoritative; the action "
        "continuity this milestone is about is then not demonstrable",
    )
    parser.add_argument(
        "--trace-samples-per-period",
        type=int,
        default=None,
        help="well-trace scan resolution; the default is the library's for a "
        "boozmn equilibrium and 96 for --synthetic",
    )
    parser.add_argument(
        "--color-by",
        choices=("sheet", "action"),
        default="sheet",
        help="scalar shown on the cut surface",
    )
    parser.add_argument(
        "--show-before",
        action="store_true",
        help="add a linked panel with the pre-cut mesh colored by action",
    )
    parser.add_argument(
        "--wells-per-transition",
        type=int,
        default=3,
        help="parent/child well triples drawn per transition; a deterministic "
        "uniform subset of the drawable samples. 0 draws none, -1 draws all",
    )
    parser.add_argument(
        "--max-path-periods",
        type=float,
        default=8.0,
        help="do not draw a well whose zeta span exceeds this many field "
        "periods; skipped wells are listed by span, never clipped",
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
        default=None,
        help="stretch the logical zeta axis for legibility only; the default "
        "makes one drawn field period 2.5 logical units, comparable to the "
        "unit disk the x-y plane spans, whatever nfp is",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.75,
        help="face opacity of the cut surface",
    )
    parser.add_argument(
        "--outgoing-opacity",
        type=float,
        default=0.0,
        help="face opacity of the outgoing g > 0 half, drawn for context "
        "only; 0 leaves it out",
    )
    parser.add_argument(
        "--tube-radius",
        type=float,
        default=0.008,
        help="drawn radius of the overlay curves, in logical units; the "
        "curves lie on the surface, so they need thickness to be visible",
    )
    parser.add_argument(
        "--no-sheet-labels",
        action="store_true",
        help="do not label each sheet at its centroid",
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


def triangle_soup(surface) -> tuple[np.ndarray, np.ndarray]:
    """Return per-triangle vertices and faces, unwrapped in zeta.

    Vertices are duplicated per triangle so a seam-crossing cell is unwrapped
    on its own into the covering space of DESIGN.md §8.1 rather than being
    drawn across the whole period. This is a rendering-only transformation.
    """
    triangles = np.asarray(surface.triangles, dtype=np.int64)
    if not len(triangles):
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
    vertices = np.asarray(surface.points, dtype=float)[triangles].copy()
    for index in (1, 2):
        difference = vertices[:, index, 2] - vertices[:, 0, 2]
        vertices[:, index, 2] -= surface.period * np.round(difference / surface.period)
    points = vertices.reshape(-1, 3)
    faces = np.arange(len(points), dtype=np.int64).reshape(-1, 3)
    return points, faces


def to_logical_mesh(surface, action, sheet_ids, zeta_scale: float):
    """Return one mesh as PyVista, with point action and cell sheet arrays."""
    points, faces = triangle_soup(surface)
    if not len(faces):
        return None
    points = points.copy()
    points[:, 2] *= zeta_scale
    cells = np.column_stack((np.full(len(faces), 3, dtype=np.int64), faces)).ravel()
    mesh = pv.PolyData(points, cells)
    triangles = np.asarray(surface.triangles, dtype=np.int64)
    mesh.point_data[ACTION_LABEL] = np.asarray(action, dtype=float)[triangles].ravel()
    if sheet_ids is not None:
        mesh.cell_data[SHEET_LABEL] = np.asarray(sheet_ids, dtype=np.int64)
    return mesh


def drawn_zeta(zeta: np.ndarray, period: float) -> np.ndarray:
    """Return one polyline's zeta unwrapped in order and centered near zero.

    Stored zeta lies in ``[0, period)``, so a curve crossing the seam would be
    drawn jumping across the whole period. The values are unwrapped in
    polyline order and then translated by whole periods -- an exact operation,
    not a fit -- so the curve's mean zeta lands in ``[0, period)``.
    """
    lifted = np.unwrap(np.asarray(zeta, dtype=float), period=period)
    return lifted - period * np.floor(np.mean(lifted) / period)


def segments_polydata(points, edges, period: float, zeta_scale: float):
    """Return the given mesh edges as line geometry, each unwrapped in zeta."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if not len(edges):
        return None
    ends = np.asarray(points, dtype=float)[edges].copy()
    difference = ends[:, 1, 2] - ends[:, 0, 2]
    ends[:, 1, 2] -= period * np.round(difference / period)
    flat = ends.reshape(-1, 3)
    flat[:, 2] *= zeta_scale
    line = pv.PolyData(flat)
    line.lines = np.column_stack(
        (
            np.full(len(edges), 2, dtype=np.int64),
            np.arange(0, 2 * len(edges), 2, dtype=np.int64),
            np.arange(1, 2 * len(edges), 2, dtype=np.int64),
        )
    ).ravel()
    return line


def port_polyline(cut, port, zeta_scale: float):
    """Return one inserted port polyline where its vertices sit in the mesh.

    The vertex IDs are in the transition's common ``u`` order, so consecutive
    IDs are consecutive along the inserted curve.
    """
    ids = np.asarray(port.polyline_vertex_ids, dtype=np.int64)
    if len(ids) < 2 or np.any(ids < 0):
        return None
    points = np.asarray(cut.surface.points[ids], dtype=float).copy()
    points[:, 2] = drawn_zeta(points[:, 2], cut.surface.period) * zeta_scale
    return pv.lines_from_points(points)


def sheet_lookup(sheet_ids: np.ndarray):
    """Return the categorical colors and integer color limits for the sheets."""
    labels = np.unique(np.asarray(sheet_ids, dtype=np.int64))
    if not len(labels):
        return [SHEET_PALETTE[0]], (-0.5, 0.5), {}
    highest = int(labels.max())
    colors = [SHEET_PALETTE[index % len(SHEET_PALETTE)] for index in range(highest + 1)]
    annotations = {float(label): f"sheet {int(label)}" for label in labels}
    return colors, (-0.5, highest + 0.5), annotations


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


def add_surface(plotter, args, mesh, *, color_by: str, clim, sheet_style, period):
    """Draw one pitch-surface mesh, colored by sheet ID or by action."""
    bounds = np.array([np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf])
    colors, sheet_clim, annotations = sheet_style
    for index in range(args.n_periods):
        shift = args.zeta_scale * period * index
        copy = mesh.translate((0.0, 0.0, shift), inplace=False)
        bounds[0::2] = np.minimum(bounds[0::2], copy.bounds[0::2])
        bounds[1::2] = np.maximum(bounds[1::2], copy.bounds[1::2])
        if color_by == "sheet":
            plotter.add_mesh(
                copy,
                scalars=SHEET_LABEL,
                cmap=colors,
                clim=sheet_clim,
                n_colors=len(colors),
                annotations=annotations,
                opacity=args.opacity,
                show_edges=True,
                edge_color="black",
                line_width=1,
                show_scalar_bar=index == 0,
                scalar_bar_args=(
                    {"title": "sheet ID", "n_labels": 0} if index == 0 else None
                ),
            )
        else:
            plotter.add_mesh(
                copy,
                scalars=ACTION_LABEL,
                cmap="viridis",
                clim=clim,
                nan_color="magenta",
                opacity=args.opacity,
                show_edges=True,
                edge_color="black",
                line_width=1,
                show_scalar_bar=index == 0,
                scalar_bar_args=({"title": "A [length]"} if index == 0 else None),
            )
    return bounds


def sheet_label_points(cut, zeta_scale: float):
    """Return one on-surface anchor point and label per sheet.

    The anchor is the triangle center nearest the sheet's centroid, so the
    label sits on the sheet it names rather than in the air beside it.
    """
    triangles = np.asarray(cut.surface.triangles, dtype=np.int64)
    if not len(triangles):
        return np.empty((0, 3)), []
    points, faces = triangle_soup(cut.surface)
    centers = points[faces].mean(axis=1)
    anchors = []
    labels = []
    for label in np.unique(cut.sheet_ids):
        selected = centers[cut.sheet_ids == label]
        centroid = selected.mean(axis=0)
        anchor = selected[int(np.argmin(np.linalg.norm(selected - centroid, axis=1)))]
        anchors.append(anchor)
        labels.append(f"sheet {int(label)}")
    anchors = np.asarray(anchors, dtype=float)
    anchors[:, 2] *= zeta_scale
    return anchors, labels


def sample_offsets(transition, period: float):
    """Return the whole-period zeta shift that draws each sample's wells.

    The trace lifts zeta out of ``[0, period)`` while the marginal point is
    stored reduced. Their difference is a whole number of periods, and
    translating a well by whole periods is an exact symmetry of the field, so
    the drawn arcs are the physical ones and only their branch of the covering
    space is chosen. The largest residual after rounding is returned so a
    disagreement that is *not* a whole period cannot pass unnoticed.

    The anchor is the transition's own marginal point rather than the inserted
    mesh vertex, so an unresolved transition -- which has no inserted vertex,
    and is the usual outcome on a real equilibrium -- still gets its wells
    drawn on the branch that puts ``m`` on the surface.
    """
    marginal = np.asarray(transition.marginal_points, dtype=float)[:, 2]
    lifted = np.asarray(transition.event_zeta_unwrapped[:, 1], dtype=float)
    raw = marginal - lifted
    finite = np.isfinite(raw)
    offsets = np.zeros(len(raw))
    offsets[finite] = period * np.round(raw[finite] / period)
    residual = (
        float(np.max(np.abs(raw[finite] - offsets[finite])))
        if np.any(finite)
        else float("nan")
    )
    return offsets, residual


def drawable_samples(transition, period: float, max_path_periods: float):
    """Return the mask of samples whose wells may be drawn, and the skipped."""
    regular = np.array(
        [status is TransitionStatus.REGULAR for status in transition.sample_status],
        dtype=bool,
    )
    parent = next(port for port in transition.ports if port.role == "parent")
    finite = np.all(np.isfinite(parent.points), axis=1) & np.all(
        np.isfinite(transition.event_zeta_unwrapped), axis=1
    )
    valid = regular & finite
    span = np.abs(
        transition.event_zeta_unwrapped[:, 2] - transition.event_zeta_unwrapped[:, 0]
    )
    short = span <= max_path_periods * period
    skipped = [
        (int(index), float(span[index] / period))
        for index in np.flatnonzero(valid & ~short)
    ]
    return valid & short, valid, skipped, span


def choose_wells(drawable: np.ndarray, count: int) -> np.ndarray:
    """Return a deterministic uniform subset of the drawable sample indices."""
    candidates = np.flatnonzero(drawable)
    if count < 0 or len(candidates) <= count:
        return candidates
    if count == 0:
        return np.empty(0, dtype=np.int64)
    picks = np.linspace(0, len(candidates) - 1, count)
    return candidates[np.unique(np.round(picks).astype(np.int64))]


def add_wells(plotter, args, field, cut, transition, offsets, indices) -> dict:
    """Draw the parent well and its two children at the selected samples.

    ``W = [a, d]`` is a wide translucent tube and the children
    ``w_1 = [a, m]``, ``w_3 = [m, d]`` are thin solid tubes inside it: they
    are subintervals of the same field-line arc, so any side-by-side drawing
    would be a fiction about where they lie.
    """
    period = cut.surface.period
    max_B_excess = -np.inf
    wrong_half = 0
    drawn = 0
    for index in indices:
        s, alpha = transition.field_line_identity[index]
        zeta_a, zeta_m, zeta_d = transition.event_zeta_unwrapped[index]
        offset = float(offsets[index])
        pieces = (
            (zeta_a, zeta_d, WELL_PARENT_COLOR, 2.4, 0.30),
            (zeta_a, zeta_m, WELL_1_COLOR, 0.9, 1.0),
            (zeta_m, zeta_d, WELL_3_COLOR, 0.9, 1.0),
        )
        for start, end, color, radius_scale, opacity in pieces:
            points, B = field_line_arc(
                field, float(s), float(alpha), start, end, period, offset
            )
            max_B_excess = max(max_B_excess, float(np.max(B)) - transition.b)
            points[:, 2] *= args.zeta_scale
            add_periodic(
                plotter,
                as_tube(pv.lines_from_points(points), radius_scale * args.tube_radius),
                args,
                period,
                color=color,
                opacity=opacity,
            )
        # The well runs from the incoming half to the outgoing half: with
        # g = B D_parallel B / (G + iota I) of DESIGN.md §5.1, g(a) < 0 and
        # g(d) > 0. Anything else means the drawn arc is not a well.
        iota = float(np.asarray(field.iota(s)).ravel()[0])
        ends = np.array([zeta_a, zeta_d])
        g_ends = (
            np.asarray(field.B(np.full(2, s), alpha + iota * ends, ends), dtype=float)
            * np.asarray(
                field.D_B(np.full(2, s), alpha + iota * ends, ends), dtype=float
            )
            / float(np.asarray(field.C(s)).ravel()[0])
        )
        if not (g_ends[0] < 0.0 < g_ends[1]):
            wrong_half += 1
        rho = float(np.sqrt(s))
        markers = []
        for zeta_event in (zeta_a, zeta_m, zeta_d):
            theta = alpha + iota * zeta_event
            markers.append(
                (
                    rho * np.cos(theta),
                    rho * np.sin(theta),
                    (zeta_event + offset) * args.zeta_scale,
                )
            )
        markers = np.asarray(markers, dtype=float)
        add_periodic(
            plotter,
            as_spheres(markers[[0, 2]], 2.2 * args.tube_radius),
            args,
            period,
            color=CROSSING_COLOR,
        )
        add_periodic(
            plotter,
            as_spheres(markers[[1]], 3.0 * args.tube_radius),
            args,
            period,
            color=MARGINAL_COLOR,
        )
        drawn += 1
    return {"n_wells": drawn, "max_B_excess": max_B_excess, "wrong_half": wrong_half}


def finite_drawn_points(points, period: float, zeta_scale: float):
    """Return the finite rows of a pre-cut port, on one drawn zeta branch."""
    points = np.asarray(points, dtype=float)
    finite = np.all(np.isfinite(points), axis=1)
    if not np.any(finite):
        return None
    kept = points[finite].copy()
    kept[:, 2] = drawn_zeta(kept[:, 2], period) * zeta_scale
    return kept


def add_cut_geometry(plotter, args, cut, transitions, ports_by_transition) -> bool:
    """Draw the duplicated seam, the ``T`` vertices, and ``GAMMA_MAX``.

    Returns whether any transition was actually cut. A cut transition and an
    unresolved one are drawn differently on purpose: a cut one shows the
    duplicated seam, the parent halo *and* the child-1 core; an unresolved one
    shows only its mapped parent samples, because no second vertex was ever
    created there. The picture must not suggest a duplication the algorithm
    refused to make.
    """
    period = cut.surface.period
    seam = segments_polydata(cut.surface.points, cut.cut_edges, period, args.zeta_scale)
    if seam is not None:
        # cut_edges is the authoritative inserted path -- every mesh edge the
        # cut duplicated, not just the mapped samples -- so it, and not a
        # chord through the samples, is what "T is in the mesh" looks like.
        add_periodic(
            plotter,
            as_tube(seam, 1.8 * args.tube_radius),
            args,
            period,
            color=CUT_SEAM_COLOR,
            opacity=0.45,
        )
    any_cut = False
    for transition in transitions:
        roles = ports_by_transition.get(transition.transition_id, {})
        was_cut = all(
            role in roles and roles[role].sheet_id >= 0
            for role in ("parent", "child_1", "child_3")
        )
        any_cut = any_cut or was_cut
        if was_cut:
            # The parent and child-1 samples are two distinct vertices at one
            # point: that duplication is the cut. A wide translucent parent
            # halo around a small solid child core draws both without
            # pretending they are apart, matching the wells' wide-parent
            # convention.
            for role, color, radius_scale, opacity in (
                ("parent", T_PARENT_COLOR, 3.2, 0.40),
                ("child_1", T_CHILD_COLOR, 1.5, 1.0),
            ):
                ids = np.asarray(roles[role].polyline_vertex_ids, dtype=np.int64)
                points = np.asarray(cut.surface.points[ids], dtype=float).copy()
                points[:, 2] = drawn_zeta(points[:, 2], period) * args.zeta_scale
                add_periodic(
                    plotter,
                    as_spheres(points, radius_scale * args.tube_radius),
                    args,
                    period,
                    color=color,
                    opacity=opacity,
                )
            line = port_polyline(cut, roles["child_3"], args.zeta_scale)
            if line is not None:
                add_periodic(
                    plotter,
                    as_tube(line, args.tube_radius),
                    args,
                    period,
                    color=GAMMA_MAX_COLOR,
                )
            continue

        # Unresolved: nothing entered the mesh, so the pre-cut TransitionCurve
        # is the only authority for where T and GAMMA_MAX are. They are drawn
        # from it, with no child-1 core, so the missing duplication is visible
        # rather than merely absent from the printout.
        parent = next(port for port in transition.ports if port.role == "parent")
        child_3 = next(port for port in transition.ports if port.role == "child_3")
        halo = finite_drawn_points(parent.points, period, args.zeta_scale)
        if halo is not None:
            add_periodic(
                plotter,
                as_spheres(halo, 3.2 * args.tube_radius),
                args,
                period,
                color=T_PARENT_COLOR,
                opacity=0.40,
            )
        gamma = finite_drawn_points(child_3.points, period, args.zeta_scale)
        if gamma is not None and len(gamma) > 1:
            add_periodic(
                plotter,
                as_tube(pv.lines_from_points(gamma), args.tube_radius),
                args,
                period,
                color=GAMMA_MAX_COLOR,
            )
    return any_cut


def cut_checks(cut, before_action) -> dict:
    """Re-derive milestone 10's acceptance criteria from the cut object."""
    triangles = np.asarray(cut.surface.triangles, dtype=np.int64)
    action = np.asarray(cut.action_values, dtype=float)
    valid = [port for port in cut.ports if port.sheet_id >= 0]
    roles = {}
    for port in valid:
        roles.setdefault(port.transition_id, {})[port.role] = port

    spanning = 0
    duplicate_pairs = 0
    distinct_pairs = 0
    max_pair_separation = 0.0
    min_action_gap = np.inf
    for transition_roles in roles.values():
        parent = transition_roles.get("parent")
        child = transition_roles.get("child_1")
        if parent is None or child is None:
            continue
        parent_ids = set(map(int, parent.polyline_vertex_ids))
        child_ids = set(map(int, child.polyline_vertex_ids))
        for triangle in triangles:
            vertices = set(map(int, triangle))
            spanning += bool(
                vertices.intersection(parent_ids) and vertices.intersection(child_ids)
            )
        for first, second in zip(parent.polyline_vertex_ids, child.polyline_vertex_ids):
            duplicate_pairs += 1
            distinct_pairs += int(first != second)
            delta = cut.surface.points[int(first)] - cut.surface.points[int(second)]
            delta[2] -= cut.surface.period * np.round(delta[2] / cut.surface.period)
            max_pair_separation = max(max_pair_separation, float(np.linalg.norm(delta)))
        gap = np.abs(parent.action_values - child.action_values)
        gap = gap[np.isfinite(gap)]
        if len(gap):
            min_action_gap = min(min_action_gap, float(np.min(gap)))

    port_action_error = []
    invalid_incidence = 0
    for port in valid:
        ids = np.asarray(port.polyline_vertex_ids, dtype=np.int64)
        error = np.abs(action[ids] - port.action_values)
        port_action_error.extend(error[np.isfinite(error)])
        for vertex in ids:
            incident = np.flatnonzero(np.any(triangles == int(vertex), axis=1))
            if not np.any(cut.sheet_ids[incident] == port.sheet_id):
                invalid_incidence += 1

    corner_action = action[triangles]
    finite_rows = np.all(np.isfinite(corner_action), axis=1)
    triangle_range = (
        float(
            np.max(
                corner_action[finite_rows].max(axis=1)
                - corner_action[finite_rows].min(axis=1)
            )
        )
        if np.any(finite_rows)
        else float("nan")
    )

    with tempfile.TemporaryDirectory(prefix="milestone10-plot-") as directory:
        path = Path(directory) / "cut.npz"
        save_cut_surface(path, cut)
        restored = load_cut_surface(path)
    round_trip = all(
        np.array_equal(first, second, equal_nan=True)
        for first, second in (
            (cut.surface.points, restored.surface.points),
            (cut.surface.triangles, restored.surface.triangles),
            (cut.action_values, restored.action_values),
            (cut.sheet_ids, restored.sheet_ids),
            (cut.cut_edges, restored.cut_edges),
        )
    )
    return {
        "triangles_spanning_action_jump": spanning,
        "invalid_port_incidence": invalid_incidence,
        "max_port_action_error": (
            float(np.max(port_action_error)) if port_action_error else float("nan")
        ),
        "duplicate_pairs": duplicate_pairs,
        "distinct_pairs": distinct_pairs,
        "max_pair_separation": max_pair_separation,
        "max_triangle_action_range": triangle_range,
        "min_parent_child_action_gap": (
            min_action_gap if np.isfinite(min_action_gap) else float("nan")
        ),
        "nonfinite_action_before": int(np.count_nonzero(~np.isfinite(before_action))),
        "nonfinite_action_after": int(np.count_nonzero(~np.isfinite(action))),
        "serialization_round_trip": round_trip,
    }


def report_transitions(cut, transitions, well_reports, offset_residuals, skipped):
    """Print what each transition is, what the cut did with it, and why."""
    unresolved = {
        int(transition_id): reason
        for transition_id, reason in zip(
            cut.unresolved_transition_ids, cut.unresolved_transition_reasons
        )
    }
    ports = {}
    for port in cut.ports:
        ports.setdefault(port.transition_id, {})[port.role] = port
    for transition in transitions:
        transition_id = transition.transition_id
        print(
            f"transition {transition_id}: {transition.status.name}, "
            f"{len(transition.u)} samples, "
            f"source critical curve {transition.source_critical_status.name}"
        )
        if transition_id in unresolved:
            print(f"    UNRESOLVED by the cut: {unresolved[transition_id]}")
            print(
                "    its ports keep vertex IDs -1 and sheet -1: an explicit "
                "unresolved hyperedge, not a missing connection (§21.2)"
            )
        for role in ("parent", "child_1", "child_3"):
            port = ports.get(transition_id, {}).get(role)
            if port is None:
                continue
            finite = port.action_values[np.isfinite(port.action_values)]
            span = (
                f"A in [{finite.min():.6g}, {finite.max():.6g}]"
                if len(finite)
                else "A all non-finite"
            )
            print(f"    port {role:<8} sheet {port.sheet_id:>3}  {span}")
        residual = np.asarray(transition.additivity_residual, dtype=float)
        residual = residual[np.isfinite(residual)]
        if len(residual):
            print(
                f"    max |A_W - A_1 - A_3| over its samples = "
                f"{np.max(np.abs(residual)):.3e}"
            )
        if np.isfinite(offset_residuals.get(transition_id, np.nan)):
            print(
                "    max branch residual placing wells on their marginal "
                f"point = {offset_residuals[transition_id]:.3e}"
            )
        report = well_reports.get(transition_id)
        if report is not None and report["n_wells"]:
            print(
                f"    {report['n_wells']} well triples drawn, "
                f"max (B - b) on their arcs = {report['max_B_excess']:.3e}"
            )
            if report["wrong_half"]:
                print(
                    f"    WARNING: {report['wrong_half']} drawn arcs do not "
                    "run from g < 0 to g > 0; they are not wells"
                )
        contacts = np.asarray(transition.contact_sample_pairs, dtype=np.int64)
        if len(contacts):
            print(
                "    equal-height contacts or folds bracketed between "
                f"samples (§5.4): {contacts.tolist()}"
            )
        for index, span in skipped.get(transition_id, ()):
            print(
                f"    well at sample {index} not drawn: it spans {span:.1f} "
                "field periods (--max-path-periods)"
            )
        failures = sorted(
            {
                f"{status.name}:{reason}"
                for status, reason in zip(
                    transition.sample_status, transition.sample_failure_reason
                )
                if status is not TransitionStatus.REGULAR
            }
        )
        if failures:
            print(f"    nonregular samples: {', '.join(failures)}")


def report_wells(transitions, drawn_indices):
    """Print the three actions of every drawn well triple."""
    header = False
    for transition in transitions:
        indices = drawn_indices.get(transition.transition_id, ())
        if not len(indices):
            continue
        parent = next(port for port in transition.ports if port.role == "parent")
        child_1 = next(port for port in transition.ports if port.role == "child_1")
        child_3 = next(port for port in transition.ports if port.role == "child_3")
        if not header:
            print(
                "\ndrawn wells (A_W is the parent well [a,d]; A_1 = A[a,m] and "
                "A_3 = A[m,d] are its children)"
            )
            print(
                f"  {'trans':>5} {'k':>3} {'u':>10} {'s':>7} {'alpha':>8} "
                f"{'span/period':>11} {'A_W':>12} {'A_1':>12} {'A_3':>12} "
                f"{'A_W-A_1-A_3':>12}"
            )
            header = True
        for index in indices:
            s, alpha = transition.field_line_identity[index]
            zeta_a, _, zeta_d = transition.event_zeta_unwrapped[index]
            print(
                f"  {transition.transition_id:>5} {index:>3} "
                f"{transition.u[index]:>10.4f} {s:>7.4f} {alpha:>8.4f} "
                f"{abs(zeta_d - zeta_a) / (2 * np.pi):>11.3f} "
                f"{parent.action_values[index]:>12.6g} "
                f"{child_1.action_values[index]:>12.6g} "
                f"{child_3.action_values[index]:>12.6g} "
                f"{transition.additivity_residual[index]:>12.3e}"
            )


def report_graph(cut) -> None:
    """Print the coarse §10.5 sheet/transition graph."""
    try:
        graph = cut.to_networkx()
    except Exception as error:  # networkx is an optional extra
        print(f"\ncoarse sheet/transition graph unavailable: {error}")
        return
    print("\ncoarse sheet/transition graph (DESIGN.md §10.5)")
    print(
        f"  {'sheet':>6} {'triangles':>10} {'logical area':>13} "
        f"{'A min':>12} {'A max':>12} {'edge':>5} {'unresolved':>11}"
    )
    for node, data in sorted(graph.nodes(data=True)):
        if data.get("kind") != "sheet":
            continue
        print(
            f"  {node[1]:>6} {data['triangle_count']:>10} "
            f"{data['logical_area']:>13.6g} {data['action_min']:>12.6g} "
            f"{data['action_max']:>12.6g} {str(data['touches_edge']):>5} "
            f"{str(data['unresolved']):>11}"
        )
    incidences = []
    for first, second, data in graph.edges(data=True):
        transition, sheet = (
            (first, second)
            if first[0] == "transition"
            else (
                second,
                first,
            )
        )
        incidences.append((transition[1], sheet[1], data["role"]))
    if incidences:
        print("  hyperedge incidence:")
        for transition_id, sheet_id, role in sorted(incidences):
            print(f"    transition {transition_id} --{role}--> sheet {sheet_id}")
    else:
        print("  no transition is incident to a sheet: nothing was cut")


def synthetic_field():
    """Return the analytic field whose GAMMA_MAX curve splits generically.

    These are the coefficients of
    ``test_production_synthetic_surface_has_no_uncut_action_jump``: a
    zero-poloidal-mode ``B`` on an ``nfp=1``, ``iota=0.4`` field whose ``b=1.4``
    surface carries one regular transition. It is the case in which the cut
    actually duplicates vertices, so it is what this script draws when a real
    equilibrium leaves every transition unresolved.
    """
    cosine = np.array([[2.0, 0.0], [-1.0, 0.0], [0.3, 0.2]])
    return SyntheticFourierField(
        nfp=1,
        m=np.zeros(3, dtype=np.int64),
        n=np.array([0, 1, 2]),
        cosine_coefficients=cosine,
        sine_coefficients=np.zeros_like(cosine),
        iota_coefficients=np.array([0.4]),
        G_coefficients=np.array([3.0]),
        I_coefficients=np.array([0.0]),
    )


def main() -> None:
    args = parse_arguments()
    default_resolution = (4, 16, 12) if args.synthetic else (6, 24, 12)
    if args.n_radial is None:
        args.n_radial = default_resolution[0]
    if args.n_poloidal is None:
        args.n_poloidal = default_resolution[1]
    if args.n_zeta is None:
        args.n_zeta = default_resolution[2]
    if args.synthetic:
        field = synthetic_field()
        source = "analytic generic-split field"
    else:
        field = BoozerField.from_boozmn(args.boozmn)
        source = args.boozmn.name
    if not 1 <= args.n_periods <= field.nfp:
        raise SystemExit(f"--n-periods must be between 1 and nfp = {field.nfp}")
    bounds = find_global_B_bounds(field, BoundsConfig(17, 32, 32))
    B_min, B_max = bounds.refined_min, bounds.refined_max
    if args.b is not None:
        b = float(args.b)
    elif args.synthetic:
        b = 1.4
    else:
        b = B_min + args.lambda_n * (B_max - B_min)
    lambda_n = (b - B_min) / (B_max - B_min)
    print(
        f"{source}: nfp = {field.nfp}, radially global B in "
        f"[{B_min:.7g}, {B_max:.7g}]"
    )
    print(f"pitch level b = {b:.8g} (lambda_n = {lambda_n:.4f})")

    if args.zeta_scale is None:
        # A viewing choice only: one drawn period is made comparable to the
        # x-y extent so an nfp=1 field is not drawn as a thread.
        args.zeta_scale = 2.5 * field.nfp / (2.0 * np.pi)
        print(f"zeta axis stretched by {args.zeta_scale:.4g} (one period = 2.5)")

    background = build_background(args, field)
    extractor = (
        PyVistaSurfaceExtractor()
        if args.extractor == "pyvista"
        else MarchingTetrahedraExtractor()
    )
    try:
        extraction = extractor.extract(background, field, b)
    except SurfaceExtractionError as error:
        raise SystemExit(f"surface extraction failed: {error.status.name}: {error}")
    surface = extraction.incoming
    print(
        f"surface {extraction.status.name}: incoming half has "
        f"{len(surface.points)} points, {len(surface.triangles)} triangles, "
        f"{len(np.unique(surface.component_ids))} components"
    )

    if args.downsample_reduction > 0.0:
        result = downsample_surface(
            surface,
            field,
            SurfaceDownsamplingConfig(target_reduction=args.downsample_reduction),
        )
        report = result.report
        achieved = 1.0 - report.output_triangle_count / max(
            1, report.input_triangle_count
        )
        print(
            f"downsampled before the cut: {report.input_triangle_count} -> "
            f"{report.output_triangle_count} triangles (requested target "
            f"{report.target_triangle_count}, achieved reduction "
            f"{achieved:.3f})"
        )
        surface = result.surface

    curves = extract_critical_curves(extraction, field, b)
    kinds = ", ".join(
        f"{polyline.kind.name}({len(polyline.vertex_ids)})"
        for polyline in curves.polylines
    )
    print(f"critical curves {curves.status.name}: {kinds or 'none'}")
    n_maxima = sum(
        polyline.kind is CriticalKind.GAMMA_MAX for polyline in curves.polylines
    )
    config = TransitionMappingConfig(
        max_curve_samples=(
            args.max_curve_samples if args.max_curve_samples > 0 else None
        ),
        action_quadrature_order=32,
        max_action_quadrature_order=512,
    )
    started = time.perf_counter()
    transitions = map_transitions(field, curves, config) if n_maxima else ()
    print(
        f"mapped {len(transitions)} transitions from {n_maxima} GAMMA_MAX "
        f"curves in {time.perf_counter() - started:.1f} s"
    )

    if args.no_actions:
        action = np.full(len(surface.points), np.nan)
        print("surface-wide actions skipped (--no-actions): A is all NaN")
    else:
        samples_per_period = args.trace_samples_per_period
        if samples_per_period is None and args.synthetic:
            samples_per_period = 96
        trace_config = (
            None
            if samples_per_period is None
            else WellTraceConfig(samples_per_field_period=samples_per_period)
        )
        started = time.perf_counter()
        data = evaluate_surface_data(surface, field, trace_config)
        action = data.action_length
        print(
            f"traced {len(surface.points)} surface vertices in "
            f"{time.perf_counter() - started:.1f} s: "
            f"{int(np.count_nonzero(data.regular))} regular, "
            f"{int(np.count_nonzero(~data.regular))} explicit non-regular"
        )

    started = time.perf_counter()
    cut = cut_surface_at_transitions(
        surface, action, transitions, field=field, config=ConstrainedCutConfig()
    )
    print(
        f"cut in {time.perf_counter() - started:.1f} s: "
        f"{len(cut.surface.points)} points "
        f"(+{len(cut.surface.points) - len(surface.points)} duplicated), "
        f"{len(np.unique(cut.sheet_ids))} sheets, "
        f"{len(cut.cut_edges)} duplicated cut edges, "
        f"{len([port for port in cut.ports if port.sheet_id >= 0])} of "
        f"{len(cut.ports)} ports on a valid sheet\n"
    )

    ports_by_transition = {}
    for port in cut.ports:
        ports_by_transition.setdefault(port.transition_id, {})[port.role] = port

    off_screen = args.screenshot is not None
    window_size = list(args.window_size) if args.window_size else list(screen_size())
    shape = (1, 2) if args.show_before else (1, 1)
    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size, shape=shape)
    # Without depth peeling the translucent surface is composited after the
    # curves and hides the parts of them that lie inside it, which is most of
    # the overlay: the cut seam and T lie on the surface.
    plotter.enable_depth_peeling(number_of_peels=12)

    finite_action = action[np.isfinite(action)]
    clim = (
        (float(finite_action.min()), float(finite_action.max()))
        if len(finite_action)
        else (0.0, 1.0)
    )
    sheet_style = sheet_lookup(cut.sheet_ids)
    period = cut.surface.period

    if args.show_before:
        plotter.subplot(0, 0)
        before_mesh = to_logical_mesh(surface, action, None, args.zeta_scale)
        if before_mesh is not None:
            add_surface(
                plotter,
                args,
                before_mesh,
                color_by="action",
                clim=clim,
                sheet_style=sheet_style,
                period=period,
            )
        plotter.add_text("before the cut: action A", font_size=9)
        plotter.add_axes(xlabel="x", ylabel="y", zlabel="zeta")
        plotter.view_isometric()
        plotter.subplot(0, 1)

    cut_mesh = to_logical_mesh(
        cut.surface, cut.action_values, cut.sheet_ids, args.zeta_scale
    )
    if cut_mesh is None:
        raise SystemExit("the cut surface has no triangles to draw")
    surface_bounds = add_surface(
        plotter,
        args,
        cut_mesh,
        color_by=args.color_by,
        clim=clim,
        sheet_style=sheet_style,
        period=period,
    )
    if args.outgoing_opacity > 0.0:
        outgoing = to_logical_mesh(
            extraction.outgoing,
            np.zeros(len(extraction.outgoing.points)),
            None,
            args.zeta_scale,
        )
        if outgoing is not None:
            for index in range(args.n_periods):
                plotter.add_mesh(
                    outgoing.translate(
                        (0.0, 0.0, args.zeta_scale * period * index), inplace=False
                    ),
                    color="lightgrey",
                    opacity=args.outgoing_opacity,
                    show_edges=True,
                    edge_color="red",
                    line_width=1,
                    show_scalar_bar=False,
                )

    any_cut = add_cut_geometry(plotter, args, cut, transitions, ports_by_transition)

    well_reports = {}
    offset_residuals = {}
    skipped_wells = {}
    drawn_indices = {}
    for transition in transitions:
        # Wells are drawn for every mapped transition, cut or not: a well that
        # a transition splits is physical geometry the mapping established,
        # and whether the mesh could represent the cut does not change it.
        offsets, residual = sample_offsets(transition, period)
        offset_residuals[transition.transition_id] = residual
        drawable, _, skipped, _ = drawable_samples(
            transition, period, args.max_path_periods
        )
        skipped_wells[transition.transition_id] = skipped
        indices = choose_wells(drawable, args.wells_per_transition)
        drawn_indices[transition.transition_id] = indices
        well_reports[transition.transition_id] = add_wells(
            plotter, args, field, cut, transition, offsets, indices
        )
    report_transitions_input = (well_reports, offset_residuals, skipped_wells)

    if not args.no_sheet_labels:
        anchors, labels = sheet_label_points(cut, args.zeta_scale)
        if len(anchors):
            plotter.add_point_labels(
                anchors,
                labels,
                font_size=11,
                point_size=1,
                shape_opacity=0.55,
                always_visible=True,
            )

    report_transitions(cut, transitions, *report_transitions_input)
    report_wells(transitions, drawn_indices)
    report_graph(cut)

    checks = cut_checks(cut, action)
    print("\nchecks re-derived from the drawn cut surface")
    print(
        "  triangles spanning a parent/child-1 action jump: "
        f"{checks['triangles_spanning_action_jump']} (must be 0)"
    )
    print(
        "  port vertices not incident to their own sheet: "
        f"{checks['invalid_port_incidence']} (must be 0)"
    )
    print(
        f"  max |A(mesh vertex) - A(port sample)| = {checks['max_port_action_error']:.3e}"
    )
    print(
        f"  duplicated T vertex pairs: {checks['distinct_pairs']} of "
        f"{checks['duplicate_pairs']} are distinct IDs, max separation "
        f"{checks['max_pair_separation']:.3e}"
    )
    if np.isfinite(checks["min_parent_child_action_gap"]):
        print(
            f"  largest action range within one triangle = "
            f"{checks['max_triangle_action_range']:.6g}, smallest "
            f"parent/child-1 jump the cut removed = "
            f"{checks['min_parent_child_action_gap']:.6g}"
        )
    else:
        print(
            f"  largest action range within one triangle = "
            f"{checks['max_triangle_action_range']:.6g}; no transition was "
            "cut, so there is no removed jump to compare it against and it "
            "is not evidence of anything"
        )
    print(
        f"  non-finite action: {checks['nonfinite_action_before']} before the "
        f"cut, {checks['nonfinite_action_after']} after"
    )
    print(f"  serialization round trip: {checks['serialization_round_trip']}")
    if (
        np.isfinite(checks["max_triangle_action_range"])
        and np.isfinite(checks["min_parent_child_action_gap"])
        and checks["max_triangle_action_range"] >= checks["min_parent_child_action_gap"]
    ):
        print(
            "  WARNING: a triangle's action range reaches the jump the cut "
            "removed; inspect it before trusting A as continuous per sheet"
        )

    legend: list[list[str]] = []
    if any_cut:
        legend.append(["T: every duplicated cut edge", CUT_SEAM_COLOR])
        legend.append(["T sample, parent vertex", T_PARENT_COLOR])
        legend.append(["T sample, child-1 vertex", T_CHILD_COLOR])
    elif transitions:
        legend.append(["T sample (mapped, not cut)", T_PARENT_COLOR])
    if transitions:
        legend.append(["GAMMA_MAX samples, child-3", GAMMA_MAX_COLOR])
    if any(len(indices) for indices in drawn_indices.values()):
        legend.append(["parent well W = [a,d]", WELL_PARENT_COLOR])
        legend.append(["child w_1 = [a,m]", WELL_1_COLOR])
        legend.append(["child w_3 = [m,d]", WELL_3_COLOR])
        legend.append(["marginal point m", MARGINAL_COLOR])
        legend.append(["crossings a, d", CROSSING_COLOR])
    if args.outgoing_opacity > 0.0:
        legend.append(["outgoing g > 0 half (context)", "red"])
    if legend:
        plotter.add_legend(legend, bcolor="grey", size=(0.22, 0.24))

    unresolved = cut.unresolved_transition_ids.tolist()
    plotter.add_axes(xlabel="x", ylabel="y", zlabel="zeta")
    plotter.add_text(
        f"{source}\n{args.backend} background mesh, "
        f"{args.extractor} extraction, b = {b:.7g} "
        f"(lambda_n = {lambda_n:.3f}), {args.n_periods} of {field.nfp} "
        "field periods\n"
        f"constrained cut: {len(np.unique(cut.sheet_ids))} sheets, "
        f"{len(cut.cut_edges)} duplicated cut edges, "
        f"unresolved transitions {unresolved or 'none'}\n"
        f"colored by {args.color_by}; logical coordinates, zeta stretched by "
        f"{args.zeta_scale:g} for legibility",
        font_size=8,
    )
    if not np.all(np.isfinite(surface_bounds)):
        surface_bounds = None
    else:
        surface_bounds = tuple(surface_bounds)
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
    if args.show_before:
        plotter.link_views()
    # Tell PyVista the camera is chosen, or show() resets it to fit every
    # actor -- including a well several field periods long -- and the framing
    # on the surface is lost.
    plotter.camera_set = True
    if off_screen:
        plotter.show(screenshot=args.screenshot)
        print(f"\nwrote {args.screenshot}")
    else:
        plotter.show(window_size=window_size)


if __name__ == "__main__":
    main()
