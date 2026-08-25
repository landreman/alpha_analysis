"""Static diagnostics for J-connectivity numerical stages (DESIGN.md §17)."""

from __future__ import annotations

import numpy as np

from .denominator import DenominatorConvergence, GlobalBBounds


def plot_background_mesh(mesh):
    """Plot wireframe, periodic seam pairs, and quality (DESIGN.md §17.2)."""
    import matplotlib.pyplot as plt

    from .background_mesh import tetrahedron_quality

    figure = plt.figure(figsize=(13, 4), constrained_layout=True)
    wire_axis = figure.add_subplot(1, 3, 1, projection="3d")
    seam_axis = figure.add_subplot(1, 3, 2, projection="3d")
    quality_axis = figure.add_subplot(1, 3, 3)

    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edges = {
        tuple(sorted((tetrahedron[first], tetrahedron[second])))
        for tetrahedron in mesh.tetrahedra
        for first, second in edge_pairs
    }
    for first, second in sorted(edges):
        coordinates = mesh.points[[first, second]]
        near_axis = np.min(np.linalg.norm(coordinates[:, :2], axis=1)) < 0.35
        wire_axis.plot(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            color="tab:red" if near_axis else "0.65",
            linewidth=0.8 if near_axis else 0.35,
        )
    wire_axis.set_title("Periodic cylinder and axis cutaway")

    lower, upper = mesh.periodic_node_pairs.T
    seam_axis.scatter(*mesh.points[lower].T, s=8, label=r"$\zeta=0$")
    seam_axis.scatter(*mesh.points[upper].T, s=8, label=r"$\zeta=L_\zeta$")
    stride = max(1, len(lower) // 24)
    for first, second in zip(lower[::stride], upper[::stride]):
        coordinates = mesh.points[[first, second]]
        seam_axis.plot(*coordinates.T, color="0.5", linewidth=0.5)
    seam_axis.set_title("Periodic seam node pairs")
    seam_axis.legend(fontsize="small")

    quality_axis.hist(tetrahedron_quality(mesh), bins=20, color="tab:blue")
    quality_axis.set_title("Tetrahedron quality")
    quality_axis.set_xlabel("mean-ratio quality")
    quality_axis.set_ylabel("count")
    quality_axis.grid(True)
    for axis in (wire_axis, seam_axis):
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel(r"$\zeta$ [rad]")
    return figure, (wire_axis, seam_axis, quality_axis)


def plot_denominator_convergence(convergence: DenominatorConvergence):
    """Plot ``V_h`` and absolute changes versus periodic grid resolution."""
    import matplotlib.pyplot as plt

    resolution = np.array(
        [max(item.n_theta, item.n_zeta) for item in convergence.estimates]
    )
    values = np.array([item.V_h for item in convergence.estimates])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(resolution, values, "o-")
    axes[0].set_xlabel("maximum periodic resolution")
    axes[0].set_ylabel(r"$V_h$")
    axes[0].grid(True)
    axes[1].semilogy(resolution[1:], convergence.absolute_changes, "o-")
    axes[1].set_xlabel("maximum periodic resolution")
    axes[1].set_ylabel(r"successive $|\Delta V_h|$")
    axes[1].grid(True)
    return figure, axes


def plot_B_extrema_profiles(bounds: GlobalBBounds):
    """Plot locally refined ``B_min(s)`` and ``B_max(s)`` in field units."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(constrained_layout=True)
    axis.plot(bounds.profile_s, bounds.profile_min, label=r"$B_{\min}(s)$")
    axis.plot(bounds.profile_s, bounds.profile_max, label=r"$B_{\max}(s)$")
    axis.set_xlabel(r"normalized toroidal flux $s$")
    axis.set_ylabel(r"magnetic-field strength $B$")
    axis.grid(True)
    axis.legend()
    return figure, axis


def plot_pitch_surface(extraction):
    """Plot full fixed-``B`` components and physical-sign halves (§17.3).

    The authoritative surface uses merged periodic vertex IDs.  Cells that
    touch the seam are duplicated only in this Matplotlib view so they remain
    local in zeta instead of drawing long triangles across the plot.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    figure = plt.figure(figsize=(11, 5), constrained_layout=True)
    component_axis = figure.add_subplot(1, 2, 1, projection="3d")
    split_axis = figure.add_subplot(1, 2, 2, projection="3d")

    full = extraction.full
    full_triangles = _periodic_plot_triangles(full)
    if len(full_triangles):
        color_map = plt.get_cmap("tab20")
        colors = color_map(full.component_ids % color_map.N)
        collection = Poly3DCollection(
            full_triangles,
            facecolors=colors,
            edgecolors="0.25",
            linewidths=0.15,
            alpha=0.8,
        )
        component_axis.add_collection3d(collection)
    component_axis.set_title("Full surface: disconnected component IDs")

    for surface, color, label in (
        (extraction.incoming, "tab:blue", "incoming g < 0"),
        (extraction.outgoing, "tab:orange", "outgoing g > 0"),
    ):
        triangles = _periodic_plot_triangles(surface)
        if len(triangles):
            split_axis.add_collection3d(
                Poly3DCollection(
                    triangles,
                    facecolors=color,
                    edgecolors="none",
                    alpha=0.65,
                    label=label,
                )
            )
    curve_segments = _periodic_plot_segments(extraction.g_zero)
    if len(curve_segments):
        split_axis.add_collection3d(
            Line3DCollection(
                curve_segments,
                colors="black",
                linewidths=1.0,
                label="g = 0",
            )
        )
    split_axis.set_title("Incoming and outgoing halves")

    for axis in (component_axis, split_axis):
        axis.set_xlim(-1.0, 1.0)
        axis.set_ylim(-1.0, 1.0)
        axis.set_zlim(0.0, full.period)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel(r"$\zeta$ [rad]")
    figure.suptitle(f"B = b pitch surface, b = {extraction.b:.8g}")
    return figure, (component_axis, split_axis)


def _periodic_plot_triangles(surface):
    triangles = []
    for triangle in surface.triangles:
        vertices = surface.points[triangle].copy()
        for index in (1, 2):
            difference = vertices[index, 2] - vertices[0, 2]
            vertices[index, 2] -= surface.period * np.round(difference / surface.period)
        mean_zeta = np.mean(vertices[:, 2])
        if mean_zeta < 0.0:
            vertices[:, 2] += surface.period
        elif mean_zeta > surface.period:
            vertices[:, 2] -= surface.period
        triangles.append(vertices)
    return np.asarray(triangles, dtype=float).reshape(-1, 3, 3)


def _periodic_plot_segments(curve):
    segments = []
    for segment in curve.segments:
        points = curve.points[segment].copy()
        difference = points[1, 2] - points[0, 2]
        points[1, 2] -= curve.period * np.round(difference / curve.period)
        mean_zeta = np.mean(points[:, 2])
        if mean_zeta < 0.0:
            points[:, 2] += curve.period
        elif mean_zeta > curve.period:
            points[:, 2] -= curve.period
        segments.append(points)
    return np.asarray(segments, dtype=float).reshape(-1, 2, 3)
