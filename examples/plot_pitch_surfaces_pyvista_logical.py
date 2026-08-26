"""Interactively render several fixed-``B`` pitch surfaces in logical coordinates.

This is the logical-coordinate twin of
``plot_pitch_surfaces_pyvista_real_space.py``: the same ``BackgroundMesh`` is
built for the same boozmn equilibrium and the same ``B = b`` level surfaces
(DESIGN.md §8.3) are extracted, but the vertices are drawn as they are stored
-- in the dimensionless logical cylinder ``(x, y, zeta)`` of DESIGN.md §3.2,
in which ``SurfaceMesh`` is authoritative -- rather than mapped to real space.
It is the surface companion to ``plot_volume_mesh_pyvista_logical.py``.

The levels are ``numpy.linspace(min(B), max(B), n_surfaces + 2)[1:-1]``, taken
over the background samples: the two endpoints are dropped because the extreme
values of ``B`` degenerate to isolated points rather than surfaces.

Each surface is colored by its own value of ``|B|`` against a shared color
scale. The two physical-sign halves of DESIGN.md §5.1,
``g = B D_parallel B / (G + iota I)``, are distinguished by their edges: the
incoming half ``g < 0`` is drawn with solid triangle edges, the outgoing half
``g > 0`` with dotted ones. The ``g = 0`` boundary curve is where the two meet.

Triangles that straddle the periodic seam are unwrapped in zeta so the surface
stays local instead of drawing a triangle across the whole period; the
unwrapped vertices sit just outside ``[0, period)``, which is correct for the
covering space of the quotient in DESIGN.md §8.1. Requesting more than one
field period stacks rigid copies translated by ``period`` in zeta, the logical
counterpart of the rigid rotation used in real space.

``--zeta-scale`` stretches the zeta axis for legibility only; it is a viewing
transformation and changes no stored quantity.

PyVista renders and stays outside the numerical core (DESIGN.md §19.2): the
NumPy arrays on ``SurfaceMesh`` remain authoritative.

Example::

    python examples/plot_pitch_surfaces_pyvista_logical.py \
        --n-surfaces 6 --extractor marching --n-periods 5
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
from plot_pitch_surfaces_pyvista_real_space import (  # noqa: E402
    dashed_edges,
    unwrapped_triangle_soup,
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


def to_logical_surface(surface, zeta_scale: float) -> pv.PolyData | None:
    """Return one ``SurfaceMesh`` as a PyVista mesh in logical coordinates.

    The points are the stored logical ``(x, y, zeta)`` with the seam unwrapped
    and zeta scaled by ``zeta_scale`` for legibility only.
    """
    points, faces = unwrapped_triangle_soup(surface)
    if not len(faces):
        return None
    points = points.copy()
    points[:, 2] *= zeta_scale
    cells = np.column_stack((np.full(len(faces), 3, dtype=np.int64), faces)).ravel()
    mesh = pv.PolyData(points, cells)
    triangles = np.asarray(surface.triangles, dtype=np.int64)
    mesh.point_data[B_LABEL] = np.asarray(surface.B, dtype=float)[triangles].ravel()
    return mesh


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
        help="field periods to draw; copies are translations in logical zeta",
    )
    parser.add_argument(
        "--zeta-scale",
        type=float,
        default=3.0,
        help="stretch the logical zeta axis for legibility only",
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
    print(
        f"Bmin: {B_min:.6g}, Bmax: {B_max:.6g}, levels: {levels}, "
        f"lowest B surface: {levels[0]:.6g}"
    )
    extractor = (
        PyVistaSurfaceExtractor()
        if args.extractor == "pyvista"
        else MarchingTetrahedraExtractor()
    )

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
            mesh = to_logical_surface(half, args.zeta_scale)
            if mesh is None:
                continue
            for period in range(args.n_periods):
                # The field is periodic, so period k is period 0 translated by
                # one period in logical zeta; |B| rides along unchanged.
                shift = args.zeta_scale * half.period * period
                copy = mesh.translate((0.0, 0.0, shift), inplace=False)
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
    plotter.add_axes(xlabel="x", ylabel="y", zlabel="zeta")
    plotter.add_text(
        f"{args.boozmn.name}\n{args.backend} background mesh, "
        f"{args.extractor} extraction: {len(extractions)} of "
        f"{len(levels)} pitch surfaces in "
        f"[{levels[0]:.4g}, {levels[-1]:.4g}], "
        f"{args.n_periods} of {field.nfp} field periods\n"
        "logical coordinates, zeta stretched by "
        f"{args.zeta_scale:g} for legibility\n"
        "solid edges: incoming g < 0    dotted edges: outgoing g > 0",
        font_size=8,
    )
    plotter.show_bounds(
        xtitle="x", ytitle="y", ztitle="zeta [rad]", location="outer", grid="back"
    )
    if off_screen:
        plotter.show(screenshot=args.screenshot)
        print(f"wrote {args.screenshot}")
    else:
        plotter.show(window_size=window_size)


if __name__ == "__main__":
    main()
