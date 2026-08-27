"""Visualize a W7-X incoming pitch surface before and after downsampling.

The example extracts one ``B=b`` surface from the fine structured volume mesh,
selects its incoming ``g < 0`` half (the state space on which bounce integrals
are evaluated), and applies the topology-preserving shortest-edge collapse in
``j_connectivity.surface_refine``.  PyVista displays the original and reduced
meshes side by side in real-space W7-X geometry.  Black curves are triangle
edges, making both the reduction and the edge-size distribution visible.

The default equilibrium is
``data/boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc``.
The level is selected as a fraction of the sampled background range so the
same command remains usable with another boozmn file.

Example::

    python examples/plot_pitch_surface_downsampling.py \
        --target-reduction 0.5 \
        --screenshot pitch_surface_downsampling.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_pitch_surfaces_pyvista_real_space import (  # noqa: E402
    to_real_space_surface,
)
from plot_volume_mesh_pyvista_real_space import (  # noqa: E402
    DEFAULT_BOOZMN,
    BoozerGeometry,
    screen_size,
)

from alpha_analysis.boozer_field import BoozerField  # noqa: E402
from alpha_analysis.j_connectivity.background_mesh import (  # noqa: E402
    BackgroundMeshConfig,
    StructuredPrismMeshBackend,
)
from alpha_analysis.j_connectivity.surface_extract import (  # noqa: E402
    MarchingTetrahedraExtractor,
    surface_flux,
)
from alpha_analysis.j_connectivity.surface_refine import (  # noqa: E402
    SurfaceDownsamplingConfig,
    downsample_surface,
)


def mesh_statistics(surface) -> dict[str, float]:
    """Return logical-coordinate triangle-size statistics for one surface."""
    vertices = np.asarray(surface.points, dtype=float)[surface.triangles].copy()
    for index in (1, 2):
        difference = vertices[:, index, 2] - vertices[:, 0, 2]
        vertices[:, index, 2] -= surface.period * np.round(difference / surface.period)
    edges = np.concatenate(
        (
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 1],
            vertices[:, 0] - vertices[:, 2],
        )
    )
    lengths = np.linalg.norm(edges, axis=1)
    return {
        "triangles": float(len(surface.triangles)),
        "points": float(len(surface.points)),
        "edge_p10": float(np.quantile(lengths, 0.1)),
        "edge_median": float(np.median(lengths)),
        "edge_p90": float(np.quantile(lengths, 0.9)),
    }


def print_statistics(label: str, statistics: dict[str, float]) -> None:
    print(
        f"{label:>6}: {int(statistics['triangles']):7d} triangles, "
        f"{int(statistics['points']):7d} points; logical edge lengths "
        f"p10/median/p90 = {statistics['edge_p10']:.4g} / "
        f"{statistics['edge_median']:.4g} / "
        f"{statistics['edge_p90']:.4g}"
    )


def add_period_copies(plotter, surface, geometry, n_periods: int, color: str) -> None:
    mesh = to_real_space_surface(surface, geometry)
    if mesh is None:
        return
    for period_index in range(n_periods):
        copy = mesh.rotate_z(
            360.0 * period_index / geometry.nfp,
            inplace=False,
        )
        plotter.add_mesh(
            copy,
            color=color,
            show_edges=True,
            edge_color="black",
            line_width=0.5,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boozmn", type=Path, default=DEFAULT_BOOZMN)
    parser.add_argument("--n-radial", type=int, default=12)
    parser.add_argument("--n-poloidal", type=int, default=48)
    parser.add_argument("--n-zeta", type=int, default=32)
    parser.add_argument(
        "--level-fraction",
        type=float,
        default=0.45,
        help="b = Bmin + fraction * (Bmax-Bmin), strictly between 0 and 1",
    )
    parser.add_argument(
        "--target-reduction",
        type=float,
        default=0.5,
        help="requested fraction of incoming-surface triangles to remove",
    )
    parser.add_argument(
        "--n-periods",
        type=int,
        default=None,
        help="field periods to draw; default: all periods",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="write a PNG off screen instead of opening an interactive window",
    )
    args = parser.parse_args()

    if not 0.0 < args.level_fraction < 1.0:
        raise SystemExit("--level-fraction must be strictly between 0 and 1")
    field = BoozerField.from_boozmn(args.boozmn)
    n_periods = field.nfp if args.n_periods is None else args.n_periods
    if not 1 <= n_periods <= field.nfp:
        raise SystemExit(f"--n-periods must be between 1 and nfp = {field.nfp}")

    background = StructuredPrismMeshBackend(
        BackgroundMeshConfig(args.n_radial, args.n_poloidal, args.n_zeta)
    ).build(field)
    B_min = float(np.min(background.B))
    B_max = float(np.max(background.B))
    b = B_min + args.level_fraction * (B_max - B_min)
    extraction = MarchingTetrahedraExtractor().extract(background, field, b)
    original = extraction.incoming
    downsampling_config = SurfaceDownsamplingConfig(
        target_reduction=args.target_reduction
    )
    result = downsample_surface(
        original,
        field,
        downsampling_config,
    )
    reduced = result.surface

    before = mesh_statistics(original)
    after = mesh_statistics(reduced)
    print(f"equilibrium: {args.boozmn}")
    print(f"sampled B range: [{B_min:.6g}, {B_max:.6g}], b = {b:.6g}")
    print_statistics("before", before)
    print_statistics("after", after)
    report = result.report
    print(
        f"triangle target/output: {report.target_triangle_count:,} / "
        f"{report.output_triangle_count:,}; achieved reduction: "
        f"{report.achieved_reduction:.1%}; reached target: {report.reached_target}"
    )
    print(
        "candidate rejections (tag/link/projection/g/face/measure): "
        f"{report.tag_protected_rejections:,} / "
        f"{report.link_condition_rejections:,} / "
        f"{report.projection_rejections:,} / "
        f"{report.g_sign_rejections:,} / "
        f"{report.face_validity_rejections:,} / "
        f"{report.flux_budget_rejections:,}"
    )
    original_flux = surface_flux(original, field)
    reduced_flux = surface_flux(reduced, field)
    relative_flux_drift = abs(reduced_flux - original_flux) / original_flux
    print(
        "|ds wedge d alpha| measure before/after: "
        f"{original_flux:.8g} / {reduced_flux:.8g}; "
        f"relative drift = {relative_flux_drift:.3%} "
        f"(budget {downsampling_config.max_flux_relative_error:.3%})"
    )

    geometry = BoozerGeometry(args.boozmn)
    off_screen = args.screenshot is not None
    if args.window_size:
        window_size = list(args.window_size)
    elif off_screen:
        window_size = [1600, 800]
    else:
        window_size = list(screen_size())
    plotter = pv.Plotter(shape=(1, 2), off_screen=off_screen, window_size=window_size)
    for column, (surface, title, color) in enumerate(
        (
            (original, "Before downsampling", "lightsteelblue"),
            (reduced, "After downsampling", "darkorange"),
        )
    ):
        plotter.subplot(0, column)
        add_period_copies(plotter, surface, geometry, n_periods, color)
        plotter.add_text(
            f"{title}\n{len(surface.triangles):,} triangles",
            font_size=11,
        )
        plotter.add_axes(xlabel="X", ylabel="Y", zlabel="Z")
    plotter.link_views()
    plotter.view_isometric()
    plotter.add_title(
        f"W7-X incoming pitch surface, b={b:.5g}; "
        f"requested reduction={args.target_reduction:.0%}",
        font_size=12,
    )
    if off_screen:
        plotter.show(screenshot=args.screenshot)
        print(f"wrote {args.screenshot}")
    else:
        plotter.show(window_size=window_size)


if __name__ == "__main__":
    main()
