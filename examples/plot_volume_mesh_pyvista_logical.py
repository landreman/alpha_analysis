"""Interactively render the volume background mesh for a boozmn equilibrium.

Mesh edges are colored by ``|B|`` and drawn in the logical coordinates of
DESIGN.md §3.2 -- the dimensionless cylinder ``(x, y, zeta)`` in which the
mesh is authoritative -- not in real space. Rendering uses PyVista, which
stays outside the numerical core (DESIGN.md §19.2): the NumPy arrays on
``BackgroundMesh`` remain the source of truth and are only viewed here.

Example::

    python examples/plot_volume_mesh_pyvista.py \
        --boozmn data/boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyvista as pv

from alpha_analysis.boozer_field import BoozerField
from alpha_analysis.j_connectivity.background_mesh import (
    BackgroundMeshConfig,
    GmshBackgroundMeshBackend,
    GmshBackgroundMeshConfig,
    StructuredPrismMeshBackend,
)

DEFAULT_BOOZMN = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
)
B_LABEL = "B [field units]"


def screen_size() -> tuple[int, int]:
    """Return the display size in pixels, so the window opens full screen."""
    probe = pv.Plotter(off_screen=True)
    try:
        width, height = probe.render_window.GetScreenSize()
    finally:
        probe.close()
    return int(width), int(height)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boozmn", type=Path, default=DEFAULT_BOOZMN)
    parser.add_argument(
        "--backend", choices=("structured", "gmsh"), default="structured"
    )
    parser.add_argument("--n-radial", type=int, default=6)
    parser.add_argument("--n-poloidal", type=int, default=24)
    parser.add_argument("--n-zeta", type=int, default=12)
    parser.add_argument(
        "--target-size", type=float, default=0.25, help="gmsh backend edge length"
    )
    parser.add_argument(
        "--zeta-scale",
        type=float,
        default=3.0,
        help="stretch the logical zeta axis for legibility only",
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

    field = BoozerField.from_boozmn(args.boozmn)
    if args.backend == "gmsh":
        mesh = GmshBackgroundMeshBackend(
            GmshBackgroundMeshConfig(target_size=args.target_size)
        ).build(field)
    else:
        mesh = StructuredPrismMeshBackend(
            BackgroundMeshConfig(args.n_radial, args.n_poloidal, args.n_zeta)
        ).build(field)

    grid = mesh.to_pyvista()
    edges = grid.extract_all_edges()
    if args.zeta_scale != 1.0:
        edges.points[:, 2] *= args.zeta_scale

    off_screen = args.screenshot is not None
    window_size = list(args.window_size) if args.window_size else list(screen_size())
    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
    plotter.add_mesh(
        edges,
        scalars=B_LABEL,
        cmap="viridis",
        line_width=2,
        scalar_bar_args={"title": "|B| [T]"},
    )
    plotter.add_axes(xlabel="x", ylabel="y", zlabel="zeta")
    plotter.add_text(
        f"{args.boozmn.name}\n{args.backend} mesh: "
        f"{len(mesh.points)} points, {len(mesh.tetrahedra)} tetrahedra",
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
