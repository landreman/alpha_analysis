"""Interactively render several fixed-``B`` pitch surfaces in real space.

This is the surface companion to ``plot_volume_mesh_pyvista_real_space.py``.
The same ``BackgroundMesh`` is built for the same boozmn equilibrium, then
``B = b`` level surfaces (DESIGN.md §8.3) are extracted at several pitch
values and every vertex is mapped from the logical cylinder of DESIGN.md §3.2
to Cartesian ``(X, Y, Z)`` in meters with the Boozer map implemented in
``plot_volume_mesh_pyvista_real_space.BoozerGeometry``.

The levels are ``numpy.linspace(min(B), max(B), n_surfaces + 2)[1:-1]``, taken
over the background samples: the two endpoints are dropped because the extreme
values of ``B`` degenerate to isolated points rather than surfaces.

Each surface is colored by its own value of ``|B|`` against a shared color
scale. The two physical-sign halves of DESIGN.md §5.1,
``g = B D_parallel B / (G + iota I)``, are distinguished by their edges: the
incoming half ``g < 0`` is drawn with solid triangle edges, the outgoing half
``g > 0`` with dotted ones. The ``g = 0`` boundary curve is where the two meet.

Triangles that straddle the periodic seam are unwrapped in zeta before being
mapped, so the surface stays local instead of drawing a triangle across the
whole device; the real-space map is smooth in zeta, so an unwrapped zeta
outside ``[0, period)`` maps correctly.

PyVista renders and stays outside the numerical core (DESIGN.md §19.2): the
NumPy arrays on ``SurfaceMesh`` remain authoritative.

Example::

    python examples/plot_pitch_surfaces_pyvista_real_space.py \
        --n-surfaces 6 --extractor marching --n-periods 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_volume_mesh_pyvista_real_space import (  # noqa: E402
    DEFAULT_BOOZMN,
    BoozerGeometry,
    screen_size,
)

from alpha_analysis.boozer_field import BoozerField  # noqa: E402
from alpha_analysis.j_connectivity.background_mesh import (  # noqa: E402
    BackgroundMeshConfig,
    GmshBackgroundMeshBackend,
    GmshBackgroundMeshConfig,
    StructuredPrismMeshBackend,
)
from alpha_analysis.j_connectivity.surface_extract import (  # noqa: E402
    MarchingTetrahedraExtractor,
    PyVistaSurfaceExtractor,
    SurfaceExtractionError,
)

B_LABEL = "B [field units]"


def unwrapped_triangle_soup(surface) -> tuple[np.ndarray, np.ndarray]:
    """Return per-triangle vertices and triangle indices, unwrapped in zeta.

    Vertices are duplicated per triangle so that seam-crossing cells can be
    unwrapped independently; this is a rendering-only transformation, exactly
    as in ``visualization._periodic_plot_triangles``.
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


def to_real_space_surface(surface, geometry: BoozerGeometry) -> pv.PolyData | None:
    """Map one ``SurfaceMesh`` to a real-space PyVista triangle mesh."""
    points, faces = unwrapped_triangle_soup(surface)
    if not len(faces):
        return None
    s = np.sum(points[:, :2] ** 2, axis=1)
    theta = np.arctan2(points[:, 1], points[:, 0])
    theta[s == 0.0] = 0.0
    cells = np.column_stack((np.full(len(faces), 3, dtype=np.int64), faces)).ravel()
    mesh = pv.PolyData(geometry.cartesian(s, theta, points[:, 2]), cells)
    triangles = np.asarray(surface.triangles, dtype=np.int64)
    mesh.point_data[B_LABEL] = np.asarray(surface.B, dtype=float)[triangles].ravel()
    return mesh


def dashed_edges(mesh: pv.PolyData, n_dashes: int = 3) -> pv.PolyData:
    """Return the mesh edges with alternating pieces removed, giving dots.

    VTK's OpenGL2 backend has no line stipple, so the dotted style is built
    geometrically: every edge is cut into ``2 * n_dashes - 1`` pieces and the
    even-numbered ones are kept.
    """
    edges = mesh.extract_all_edges()
    lines = np.asarray(edges.lines, dtype=np.int64).reshape(-1, 3)[:, 1:]
    points = np.asarray(edges.points, dtype=float)
    starts, ends = points[lines[:, 0]], points[lines[:, 1]]
    n_pieces = 2 * n_dashes - 1
    fractions = np.linspace(0.0, 1.0, n_pieces + 1)
    keep = np.arange(0, n_pieces, 2)
    first = (
        starts[:, None, :]
        + fractions[keep][None, :, None] * (ends - starts)[:, None, :]
    )
    second = (
        starts[:, None, :]
        + fractions[keep + 1][None, :, None] * (ends - starts)[:, None, :]
    )
    dash_points = np.stack((first, second), axis=2).reshape(-1, 3)
    segments = np.arange(len(dash_points), dtype=np.int64).reshape(-1, 2)
    cells = np.column_stack(
        (np.full(len(segments), 2, dtype=np.int64), segments)
    ).ravel()
    return pv.PolyData(dash_points, lines=cells)


def main() -> None:
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
        default=6,
        help="number of pitch surfaces strictly between min(B) and max(B)",
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
        help="field periods to draw; copies are rigid rotations of the surfaces",
    )
    parser.add_argument(
        "--opacity", type=float, default=0.55, help="surface face opacity"
    )
    parser.add_argument(
        "--n-dashes", type=int, default=10, help="dots per edge on the g > 0 half"
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
    args = parser.parse_args()

    if args.n_surfaces < 1:
        raise SystemExit("--n-surfaces must be at least 1")
    field = BoozerField.from_boozmn(args.boozmn)
    if not 1 <= args.n_periods <= field.nfp:
        raise SystemExit(f"--n-periods must be between 1 and nfp = {field.nfp}")
    if args.backend == "gmsh":
        background = GmshBackgroundMeshBackend(
            GmshBackgroundMeshConfig(target_size=args.target_size)
        ).build(field)
    else:
        background = StructuredPrismMeshBackend(
            BackgroundMeshConfig(args.n_radial, args.n_poloidal, args.n_zeta)
        ).build(field)

    B_min, B_max = float(background.B.min()), float(background.B.max())
    # The endpoints are excluded: B = min(B) and B = max(B) are attained at
    # isolated points, not on a surface.
    levels = np.linspace(B_min, B_max, args.n_surfaces + 2)[1:-1]
    print(f"Bmin: {B_min:.6g}, Bmax: {B_max:.6g}, levels: {levels}, lowest B surface: {levels[0]:.6g}")
    extractor = (
        PyVistaSurfaceExtractor()
        if args.extractor == "pyvista"
        else MarchingTetrahedraExtractor()
    )

    geometry = BoozerGeometry(args.boozmn)
    extractions = []
    for level in levels:
        try:
            extractions.append((level, extractor.extract(background, field, level)))
        except SurfaceExtractionError as error:
            print(f"b = {level:.6g}: {error.status.name}: {error}")

    off_screen = args.screenshot is not None
    window_size = list(args.window_size) if args.window_size else list(screen_size())
    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
    clim = (B_min, B_max)
    first = True
    for level, extraction in extractions:
        for half, dotted in ((extraction.incoming, False), (extraction.outgoing, True)):
            mesh = to_real_space_surface(half, geometry)
            if mesh is None:
                continue
            for period in range(args.n_periods):
                # The field is periodic, so period k is period 0 rotated about
                # the z axis; |B| rides along unchanged.
                angle = 360.0 * period / field.nfp
                copy = mesh.rotate_z(angle, inplace=False)
                plotter.add_mesh(
                    copy,
                    scalars=B_LABEL,
                    cmap="viridis",
                    clim=clim,
                    opacity=args.opacity,
                    show_edges=not dotted,
                    edge_color="black",
                    line_width=1,
                    show_scalar_bar=first,
                    scalar_bar_args={"title": "|B| [T]"} if first else None,
                )
                first = False
                if dotted:
                    plotter.add_mesh(
                        dashed_edges(copy, args.n_dashes),
                        color="black",
                        line_width=1,
                        show_scalar_bar=False,
                    )
    plotter.add_axes(xlabel="X", ylabel="Y", zlabel="Z")
    plotter.add_text(
        f"{args.boozmn.name}\n{args.backend} background mesh, "
        f"{args.extractor} extraction: {len(extractions)} of "
        f"{len(levels)} pitch surfaces in "
        f"[{levels[0]:.4g}, {levels[-1]:.4g}], "
        f"{args.n_periods} of {field.nfp} field periods\n"
        "solid edges: incoming g < 0    dotted edges: outgoing g > 0",
        font_size=8,
    )
    plotter.show_bounds(
        xtitle="X [m]", ytitle="Y [m]", ztitle="Z [m]", location="outer", grid="back"
    )
    if off_screen:
        plotter.show(screenshot=args.screenshot)
        print(f"wrote {args.screenshot}")
    else:
        plotter.show(window_size=window_size)


if __name__ == "__main__":
    main()
