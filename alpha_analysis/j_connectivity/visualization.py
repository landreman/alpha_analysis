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
