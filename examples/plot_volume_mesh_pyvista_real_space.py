"""Interactively render the volume background mesh in real space.

This is the real-space companion to
``plot_volume_mesh_pyvista_logical.py``: the same ``BackgroundMesh`` for the
same boozmn equilibrium, with the same ``|B|`` coloring on the mesh edges,
but with every vertex mapped from the logical cylinder of DESIGN.md §3.2 to
Cartesian ``(X, Y, Z)`` in meters.

The map is the standard Boozer-coordinate one, read straight from the boozmn
file (the ``BoozerField`` class keeps only ``B``, so the geometry arrays are
loaded here)::

    s = x^2 + y^2,  theta = atan2(y, x)
    R  = sum rmnc_b cos(m theta - n zeta)
    Z  = sum zmns_b sin(m theta - n zeta)
    nu = -sum pmns_b sin(m theta - n zeta)
    phi = zeta - nu,  X = R cos(phi),  Y = R sin(phi)

``pmns_b`` is ``-nu`` in the boozmn format, and the cylindrical angle is
obtained by *subtracting* nu from the Boozer toroidal angle; both follow
``booz_xform`` (``read_boozmn.cpp`` and ``plots.py``). The Fourier amplitudes
live on the half grid and are splined in ``s`` exactly as ``BoozerField``
splines ``bmnc``; on the magnetic axis only the ``m = 0`` modes contribute,
which is imposed rather than extrapolated.

PyVista renders and stays outside the numerical core (DESIGN.md §19.2): the
NumPy arrays on ``BackgroundMesh`` remain authoritative.

Example::

    python examples/plot_volume_mesh_pyvista_real_space.py --n-periods 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.interpolate import CubicSpline
from scipy.io import netcdf_file

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


class BoozerGeometry:
    """Boozer-coordinate position map ``(s, theta, zeta) -> (X, Y, Z)``.

    Amplitudes are in meters, angles in radians. ``ixn_b`` already carries the
    factor of ``nfp``.
    """

    def __init__(self, boozmn_file: Path) -> None:
        f = netcdf_file(boozmn_file, mmap=False)
        try:
            n_surfaces = len(f.variables["iota_b"][()])
            s_full = np.linspace(0.0, 1.0, n_surfaces)
            self.s_half = s_full[1:] - 0.5 * (s_full[1] - s_full[0])
            self.xm = np.asarray(f.variables["ixm_b"][()], dtype=float)
            self.xn = np.asarray(f.variables["ixn_b"][()], dtype=float)
            self.nfp = int(f.variables["nfp_b"][()])
            if bool(f.variables.get("lasym__logical__", np.array(0))[()]):
                raise ValueError(
                    "this example implements the stellarator-symmetric map only"
                )
            splines = {
                name: CubicSpline(
                    self.s_half,
                    np.asarray(f.variables[name][()], dtype=float),
                    axis=0,
                    extrapolate=True,
                )
                for name in ("rmnc_b", "zmns_b", "pmns_b")
            }
        finally:
            f.close()
        self._R_spline = splines["rmnc_b"]
        self._Z_spline = splines["zmns_b"]
        self._p_spline = splines["pmns_b"]

    def cartesian(
        self, s: np.ndarray, theta: np.ndarray, zeta: np.ndarray
    ) -> np.ndarray:
        """Return an ``(n_points, 3)`` array of Cartesian positions in meters."""
        s = np.asarray(s, dtype=float)
        phase = theta[:, None] * self.xm - zeta[:, None] * self.xn
        # On the axis every m > 0 amplitude vanishes; do not trust the spline
        # extrapolation off the half grid to say so.
        axis = (s[:, None] == 0.0) & (self.xm[None, :] > 0.0)
        amplitudes = []
        for spline in (self._R_spline, self._Z_spline, self._p_spline):
            values = np.atleast_2d(np.asarray(spline(s), dtype=float))
            amplitudes.append(np.where(axis, 0.0, values))
        sine, cosine = np.sin(phase), np.cos(phase)
        R = np.sum(amplitudes[0] * cosine, axis=1)
        Z = np.sum(amplitudes[1] * sine, axis=1)
        nu = -np.sum(amplitudes[2] * sine, axis=1)
        phi = zeta - nu
        return np.column_stack((R * np.cos(phi), R * np.sin(phi), Z))


def to_real_space(edges: pv.PolyData, geometry: BoozerGeometry) -> pv.PolyData:
    """Map an edge network from logical ``(x, y, zeta)`` to real space."""
    points = np.asarray(edges.points, dtype=float)
    s = np.sum(points[:, :2] ** 2, axis=1)
    theta = np.arctan2(points[:, 1], points[:, 0])
    theta[s == 0.0] = 0.0
    mapped = edges.copy()
    mapped.points = geometry.cartesian(s, theta, points[:, 2])
    return mapped


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
        "--n-periods",
        type=int,
        default=1,
        help="field periods to draw; copies are rigid rotations of the mesh",
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
    if not 1 <= args.n_periods <= field.nfp:
        raise SystemExit(f"--n-periods must be between 1 and nfp = {field.nfp}")
    if args.backend == "gmsh":
        mesh = GmshBackgroundMeshBackend(
            GmshBackgroundMeshConfig(target_size=args.target_size)
        ).build(field)
    else:
        mesh = StructuredPrismMeshBackend(
            BackgroundMeshConfig(args.n_radial, args.n_poloidal, args.n_zeta)
        ).build(field)

    geometry = BoozerGeometry(args.boozmn)
    edges = to_real_space(mesh.to_pyvista().extract_all_edges(), geometry)

    off_screen = args.screenshot is not None
    window_size = list(args.window_size) if args.window_size else list(screen_size())
    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
    clim = (float(mesh.B.min()), float(mesh.B.max()))
    for period in range(args.n_periods):
        # The field is periodic, so period k is period 0 rotated about the
        # z axis; |B| rides along unchanged.
        copy = edges.rotate_z(360.0 * period / field.nfp, inplace=False)
        plotter.add_mesh(
            copy,
            scalars=B_LABEL,
            cmap="viridis",
            clim=clim,
            line_width=2,
            scalar_bar_args={"title": "|B| [T]"},
        )
    plotter.add_axes(xlabel="X", ylabel="Y", zlabel="Z")
    plotter.add_text(
        f"{args.boozmn.name}\n{args.backend} mesh: "
        f"{len(mesh.points)} points, {len(mesh.tetrahedra)} tetrahedra, "
        f"{args.n_periods} of {field.nfp} field periods",
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
