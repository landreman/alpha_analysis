"""Static diagnostics for J-connectivity numerical stages (DESIGN.md §17)."""

from __future__ import annotations

import numpy as np

from .denominator import DenominatorConvergence, GlobalBBounds


def plot_critical_curves(curves, *, output_path=None):
    """Plot classified marginal polylines from DESIGN.md §§17.3 and 23.

    Gamma-min, gamma-max, and degenerate portions are visually distinct,
    arms that terminate on the ``s=1`` boundary are marked, and an unresolved
    result is called out in the title rather than hidden.
    """
    import matplotlib.pyplot as plt

    from .critical_curves import CriticalKind
    from .surface_extract import SurfaceMesh

    figure = plt.figure(figsize=(7, 6), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    styles = {
        CriticalKind.GAMMA_MIN: ("tab:blue", r"$\Gamma_{\min}$"),
        CriticalKind.GAMMA_MAX: ("tab:red", r"$\Gamma_{\max}$"),
        CriticalKind.DEGENERATE: ("tab:orange", "degenerate / unresolved"),
    }
    used_labels: set[CriticalKind] = set()
    for polyline in curves.polylines:
        points = curves.points[polyline.vertex_ids]
        if polyline.closed:
            closing = points[:1].copy()
            closing[0, 2] += curves.period * np.round(
                (points[-1, 2] - closing[0, 2]) / curves.period
            )
            points = np.vstack((points, closing))
        color, label = styles[polyline.kind]
        axis.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color,
            linewidth=2.0,
            label=label if polyline.kind not in used_labels else None,
        )
        used_labels.add(polyline.kind)
    degenerate = curves.point_kind == CriticalKind.DEGENERATE
    if np.any(degenerate):
        points = curves.points[degenerate]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color="black",
            marker="x",
            s=36,
            label="degenerate point",
        )
    edge_exit = ((curves.boundary_tags & SurfaceMesh.EDGE) != 0) & (
        np.abs(np.sum(curves.points[:, :2] ** 2, axis=1) - 1.0) <= 1.0e-9
    )
    if np.any(edge_exit):
        points = curves.points[edge_exit]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color="tab:green",
            marker="s",
            s=28,
            label="edge exit (s=1)",
        )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel(r"$\zeta$ [rad]")
    axis.set_title(f"Critical curves: {curves.status.name}")
    if curves.polylines or np.any(degenerate) or np.any(edge_exit):
        axis.legend(fontsize="small")
    if output_path is not None:
        figure.savefig(output_path, dpi=160)
    return figure, axis


def plot_well_profile(field, trace, *, output_path=None, n_samples=513):
    """Plot and optionally save the selected-well diagnostic from §17.4.

    Coordinates are in radians; ``A`` and ``K`` use the length units of the
    field's Boozer current profiles. Entry, exit, classified extrema, both
    integrands and cumulative quadratures, reduced and unwrapped paths, root
    residuals, period count, error estimates, and explicit status are shown.
    """
    import matplotlib.pyplot as plt

    from .well_trace import sample_well_profile

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    if trace.status.name != "REGULAR":
        for axis in axes.ravel():
            axis.axis("off")
        figure.patch.set_facecolor("mistyrose")
        field_axis = axes[0, 0]
        field_axis.axis("on")
        status_axis = axes[1, 2]

        period = 2.0 * np.pi / field.nfp
        s, theta_in, zeta_in = map(float, trace.q_in)
        iota = float(np.asarray(field.iota(s)))
        C = float(np.asarray(field.G(s))) + iota * float(np.asarray(field.I(s)))
        sigma = float(np.sign(C))
        if trace.tangent_zeta_unwrapped.size:
            zeta_end = float(trace.tangent_zeta_unwrapped[-1])
            while sigma * (zeta_end - zeta_in) < 0.0:
                zeta_in -= sigma * period
            u_end = sigma * (zeta_end - zeta_in)
        else:
            u_end = max(1, trace.field_period_count) * period
        if u_end <= 1.0e-12:
            u_end = period
        u = np.linspace(0.0, u_end, n_samples)
        zeta = zeta_in + sigma * u
        theta = theta_in + iota * (zeta - zeta_in)
        B_values = np.asarray(field.B(s, theta, zeta), dtype=float)
        field_axis.plot(zeta, B_values, label=r"$B(\zeta)$")
        field_axis.axhline(trace.b, color="black", linestyle="--", label=r"$b$")
        field_axis.plot(zeta[0], B_values[0], "o", color="tab:blue", label="entry")
        if trace.extrema_zeta_unwrapped.size:
            field_axis.plot(
                trace.extrema_zeta_unwrapped,
                trace.extrema_B,
                "^",
                color="tab:orange",
                linestyle="none",
                label="scanned extrema",
            )
        if trace.tangent_zeta_unwrapped.size:
            field_axis.plot(
                trace.tangent_zeta_unwrapped,
                trace.tangent_B,
                "*",
                color="darkred",
                markersize=11,
                linestyle="none",
                label="tangent candidate",
            )
        field_axis.set_xlabel(r"unwrapped $\zeta$ [rad]")
        field_axis.set_ylabel(r"$B$")
        field_axis.set_title("Non-regular trace profile")
        field_axis.grid(True)
        field_axis.legend(fontsize="small")

        status_axis.text(
            0.0,
            1.0,
            "\n".join(
                (
                    f"UNRESOLVED WELL TRACE: {trace.status.name}",
                    f"b: {trace.b:.10g}",
                    f"field periods scanned: {trace.field_period_count}",
                    f"B residual in: {trace.B_residual_in:.3e}",
                )
            ),
            color="darkred",
            va="top",
            weight="bold",
        )
        figure.suptitle("Well first-return diagnostic: unresolved")
        if output_path is not None:
            figure.savefig(output_path, dpi=160)
        return figure, axes

    profile = sample_well_profile(field, trace, n_samples=n_samples)
    period = 2.0 * np.pi / field.nfp

    field_axis = axes[0, 0]
    field_axis.plot(profile.zeta_unwrapped, profile.B, label=r"$B(\zeta)$")
    field_axis.axhline(trace.b, color="black", linestyle="--", label=r"$b$")
    field_axis.plot(
        profile.zeta_unwrapped[[0, -1]],
        profile.B[[0, -1]],
        "o",
        color="tab:red",
        label="entry / exit",
    )
    for kind, marker, label in ((1, "v", "minimum"), (-1, "^", "maximum")):
        mask = trace.extrema_kind == kind
        if np.any(mask):
            field_axis.plot(
                trace.extrema_zeta_unwrapped[mask],
                trace.extrema_B[mask],
                marker,
                linestyle="none",
                label=label,
            )
    field_axis.set_xlabel(r"unwrapped $\zeta$ [rad]")
    field_axis.set_ylabel(r"$B$")
    field_axis.legend(fontsize="small")
    field_axis.grid(True)

    integrand_axis = axes[0, 1]
    integrand_axis.plot(
        profile.zeta_unwrapped,
        profile.action_integrand,
        label=r"$dA/d|\zeta|$",
    )
    finite_K = np.where(
        np.isfinite(profile.bounce_time_integrand),
        profile.bounce_time_integrand,
        np.nan,
    )
    integrand_axis.plot(profile.zeta_unwrapped, finite_K, label=r"$dK/d|\zeta|$")
    integrand_axis.set_xlabel(r"unwrapped $\zeta$ [rad]")
    integrand_axis.set_ylabel("integrand")
    integrand_axis.legend(fontsize="small")
    integrand_axis.grid(True)

    cumulative_axis = axes[0, 2]
    cumulative_axis.plot(profile.zeta_unwrapped, profile.cumulative_A, label=r"$A$")
    cumulative_axis.plot(profile.zeta_unwrapped, profile.cumulative_K, label=r"$K$")
    cumulative_axis.set_xlabel(r"unwrapped $\zeta$ [rad]")
    cumulative_axis.set_ylabel("cumulative length")
    cumulative_axis.legend(fontsize="small")
    cumulative_axis.grid(True)

    reduced_axis = axes[1, 0]
    reduced_axis.plot(
        np.mod(profile.zeta_unwrapped, period),
        np.mod(profile.theta_unwrapped, 2.0 * np.pi),
    )
    reduced_axis.set_xlabel(r"$\zeta\ \mathrm{mod}\ L_\zeta$ [rad]")
    reduced_axis.set_ylabel(r"$\theta\ \mathrm{mod}\ 2\pi$ [rad]")
    reduced_axis.grid(True)

    unwrapped_axis = axes[1, 1]
    unwrapped_axis.plot(profile.zeta_unwrapped, profile.theta_unwrapped)
    unwrapped_axis.set_xlabel(r"unwrapped $\zeta$ [rad]")
    unwrapped_axis.set_ylabel(r"unwrapped $\theta$ [rad]")
    unwrapped_axis.grid(True)

    status_axis = axes[1, 2]
    status_axis.axis("off")
    status_axis.text(
        0.0,
        1.0,
        "\n".join(
            (
                f"status: {trace.status.name}",
                f"b: {trace.b:.10g}",
                f"field periods: {trace.field_period_count}",
                f"B residual in: {trace.B_residual_in:.3e}",
                f"B residual out: {trace.B_residual_out:.3e}",
                f"A: {trace.action_length:.10g}",
                f"K: {trace.bounce_time_length:.10g}",
                f"A error estimate: {trace.quadrature_error_A:.3e}",
                f"K error estimate: {trace.quadrature_error_K:.3e}",
                f"internal maxima: {trace.n_internal_maxima}",
                f"itinerary: {int(trace.itinerary_hash):016x}",
            )
        ),
        va="top",
        family="monospace",
    )
    figure.suptitle("Regular well first-return diagnostic")
    if output_path is not None:
        figure.savefig(output_path, dpi=160)
    return figure, axes


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


def plot_surface_data(data, *, output_path=None, metadata=None):
    """Plot surface-wide return data and conspicuous failures (§§17.1, 17.3).

    ``A`` and ``K`` use the length units of the Boozer current profiles; exit
    zeta is in radians and the remaining quantities are integer categories.
    Non-regular vertices are overplotted as magenta crosses on every numerical
    map instead of disappearing through ``NaN`` colors. Optional ``metadata``
    is rendered in the title for equilibrium hashes, tolerances, and commits.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    maps = (
        (data.action_length, r"$A$ action length", "viridis"),
        (data.bounce_time_length, r"$K$ bounce-time length", "magma"),
        (data.zeta_out_unwrapped, r"unwrapped $\zeta_+$ [rad]", "twilight"),
        (data.field_period_count, "field-period count", "cividis"),
        (data.n_internal_maxima, "internal maximum count", "plasma"),
        (data.trace_status, "trace status code", "tab10"),
    )
    figure = plt.figure(figsize=(15, 9), constrained_layout=True)
    axes = tuple(
        figure.add_subplot(2, 3, index + 1, projection="3d") for index in range(6)
    )
    points = data.surface.points
    edge_segments = []
    for triangle in data.surface.triangles:
        vertices = points[triangle].copy()
        for index in (1, 2):
            difference = vertices[index, 2] - vertices[0, 2]
            vertices[index, 2] -= data.surface.period * np.round(
                difference / data.surface.period
            )
        edge_segments.extend((vertices[[0, 1]], vertices[[1, 2]], vertices[[2, 0]]))
    failed = ~data.regular
    for axis, (values, title, color_map) in zip(axes, maps):
        if edge_segments:
            axis.add_collection3d(
                Line3DCollection(edge_segments, colors="0.75", linewidths=0.35)
            )
        plotted = (
            np.ones(len(points), dtype=bool)
            if values is data.trace_status
            else data.regular
        )
        plotted_values = np.asarray(values)[plotted]
        if plotted_values.size:
            artist = axis.scatter(
                *points[plotted].T,
                c=plotted_values,
                cmap=color_map,
                s=18,
                depthshade=False,
            )
            figure.colorbar(artist, ax=axis, shrink=0.62, pad=0.08)
        if np.any(failed):
            axis.scatter(
                *points[failed].T,
                color="magenta",
                marker="x",
                s=42,
                linewidths=1.6,
                label="non-regular",
            )
            axis.legend(fontsize="x-small")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel(r"$\zeta$ [rad]")
    context = ""
    if metadata:
        context = "\n" + ", ".join(f"{key}={value}" for key, value in metadata.items())
    figure.suptitle(
        f"B=b incoming surface: b={data.surface.level:.8g}, "
        f"{len(points)} points, {len(data.surface.triangles)} triangles{context}"
    )
    if output_path is not None:
        figure.savefig(output_path, dpi=160)
    return figure, axes


def plot_transition_diagnostics(field, transition, *, output_path=None):
    """Plot matched transition geometry and action diagnostics (§17.5).

    The four panels show ``GAMMA_MAX`` and ``T`` with pointwise connectors,
    the three limiting action curves, the additivity residual, and selected
    lifted well profiles. Field and action values retain their supplied
    units; all angular coordinates are radians. Non-regular transitions are
    labeled explicitly rather than omitted.
    """
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(12, 9), constrained_layout=True)
    geometry_axis = figure.add_subplot(2, 2, 1, projection="3d")
    action_axis = figure.add_subplot(2, 2, 2)
    residual_axis = figure.add_subplot(2, 2, 3)
    profile_axis = figure.add_subplot(2, 2, 4)
    axes = (geometry_axis, action_axis, residual_axis, profile_axis)

    by_role = {port.role: port for port in transition.ports}
    companion = by_role["parent"].points.copy()
    marginal = transition.marginal_points.copy()
    companion[:, 2] = by_role["parent"].zeta_unwrapped
    marginal[:, 2] = transition.event_zeta_unwrapped[:, 1]
    geometry_axis.plot(*companion.T, color="tab:blue", label=r"$T$")
    geometry_axis.plot(*marginal.T, color="tab:red", label=r"$\Gamma_{\max}$")
    for first, second in zip(companion, marginal):
        geometry_axis.plot(
            [first[0], second[0]],
            [first[1], second[1]],
            [first[2], second[2]],
            color="0.6",
            linewidth=0.6,
        )
    geometry_axis.set_xlabel("x")
    geometry_axis.set_ylabel("y")
    geometry_axis.set_zlabel(r"$\zeta$ [rad]")
    geometry_axis.legend()

    labels = {
        "parent": r"$A_W$",
        "child_1": r"$A_1$",
        "child_3": r"$A_3$",
    }
    for port in transition.ports:
        action_axis.plot(
            transition.u,
            port.action_values,
            marker="o",
            markersize=2.5,
            label=labels.get(port.role, port.role),
        )
    action_axis.set_xlabel(r"common curve parameter $u$")
    action_axis.set_ylabel(r"action length $A$ [length]")
    action_axis.legend()

    residual_axis.axhline(0.0, color="black", linewidth=0.7)
    residual_axis.plot(
        transition.u,
        transition.additivity_residual,
        color="tab:purple",
        marker="o",
        markersize=2.5,
    )
    residual_axis.set_xlabel(r"common curve parameter $u$")
    residual_axis.set_ylabel(r"$A_W-A_1-A_3$ [length]")

    finite_events = np.all(np.isfinite(transition.event_zeta_unwrapped), axis=1)
    selected = np.flatnonzero(finite_events)
    if len(selected):
        selected = np.unique(selected[[0, len(selected) // 2, -1]])
        b = None
        for sample in selected:
            s, alpha = transition.field_line_identity[sample]
            zeta_a, zeta_m, zeta_d = transition.event_zeta_unwrapped[sample]
            zeta = np.linspace(zeta_a, zeta_d, 257)
            iota = float(np.asarray(field.iota(s)).reshape(()))
            theta = alpha + iota * zeta
            B = np.asarray(field.B(s, theta, zeta), dtype=float)
            b = float(np.asarray(field.B(s, alpha + iota * zeta_m, zeta_m)).reshape(()))
            profile_axis.plot(zeta, B, label=f"u={transition.u[sample]:.3g}")
            profile_axis.axvline(zeta_m, color="0.75", linewidth=0.5)
        profile_axis.axhline(
            b,
            color="black",
            linestyle="--",
            linewidth=0.8,
            label="b",
        )
        profile_axis.legend(fontsize="small")
    else:
        profile_axis.text(
            0.5,
            0.5,
            f"unresolved: {transition.status.name}",
            color="magenta",
            ha="center",
            va="center",
            transform=profile_axis.transAxes,
        )
    profile_axis.set_xlabel(r"lifted $\zeta$ [rad]")
    profile_axis.set_ylabel("B [field units]")
    figure.suptitle(
        f"Transition {transition.transition_id}: b={transition.b:.8g}, "
        f"{transition.status.name}, "
        f"{len(transition.u)} matched samples"
    )
    if output_path is not None:
        figure.savefig(output_path, dpi=160)
    return figure, axes


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
