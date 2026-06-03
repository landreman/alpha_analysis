import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors
from matplotlib import ticker
from matplotlib.widgets import CheckButtons
from matplotlib.widgets import RangeSlider
from matplotlib.widgets import Slider
import numpy as np
try:
    from numpy import trapezoid
except ImportError:
    # Older versions of numpy don't have trapezoid, so we can use trapz instead.
    from numpy import trapz as trapezoid

from scipy.integrate import quad

from .boozer_field import BoozerSurface
from .boozer_field import BoozerField
from .bounce_points import find_bounce_points
from .bounce_points import _find_well_bounds_from_allowed

# Defaults for plot_J_invariant:
LARGE_PLOT_LAMBDA_N_VALUES = np.round(np.arange(0.05, 1.0, 0.05), 2).tolist()
LAMBDA_N_VALUES = LARGE_PLOT_LAMBDA_N_VALUES[1:]
DEFAULT_N_ALPHA = 120
DEFAULT_N_RHO = 61
DEFAULT_N_PHI = 4001
DEFAULT_CONTOUR_LEVELS = 20

SUPTITLE_FONT_SIZE = 11
PATH_FONT_SIZE = 9
SUBPLOT_TITLE_FONT_SIZE = 10
LABEL_FONT_SIZE = 9
TICK_FONT_SIZE = 8
FOOTER_FONT_SIZE = 6
SUBPLOT_LEFT = 0.06
SUBPLOT_BOTTOM = 0.067
SUBPLOT_RIGHT = 0.985
SUBPLOT_TOP = 0.915
SUBPLOT_WSPACE = 0.55
SUBPLOT_HSPACE = 0.6

def compute_J_invariant(
    surf: BoozerSurface,
    B_bounce: float,
    theta_center: float,
    phi_center: float,
    n_phi: float = 501,
    phi_margin: float = 5.0,
    refine: bool = True,
    clipped_well_nan: bool = True,
) -> dict:
    """Compute the bounce points and J for a given surface and B_bounce value.

    Args:
        surf (BoozerSurface): The Boozer surface object.
        theta_center (float): The center theta value.
        phi_center (float): The center phi value.
        B_bounce (float): The bounce magnetic field value.
        refine (bool, optional): Whether to refine the bounce points. Defaults
        to True.
        n_phi (int, optional): The number of phi points to use for the
        computation. Defaults to 501.
        phi_margin (float, optional): The margin around the center phi value
        to consider for the computation. Defaults to 5.0.
        clipped_well_nan (bool, optional): Whether to set J to NaN if the
        well extends beyond the phi grid. Defaults to True.

    Returns:
        dict: A dictionary containing the computed bounce points and J value.
    """
    data = find_bounce_points(
        surf,
        B_bounce,
        theta_center,
        phi_center,
        n_phi=n_phi,
        phi_margin=phi_margin,
        refine=refine,
    )

    if not np.any(data["allowed"]):
        data["J"] = np.nan
        return data
    
    if data["well_crosses_left_edge"] or data["well_crosses_right_edge"]:
        if clipped_well_nan:
            data["J"] = np.nan
            return data

    def integrand(phi: np.ndarray) -> np.ndarray:
        theta = theta_center + surf.iota * (phi - phi_center)
        B = surf.compute_B([theta], [phi])[0]
        return np.sqrt(np.maximum(0, 1 - B / B_bounce)) / B
    
    # Integrate over the allowed region.
    constant = np.abs(surf.G + surf.I * surf.iota) / (surf.R00 * 2 * np.pi / surf.nfp)
    if refine:
        quad_results = quad(
            integrand,
            data["phi_left"],
            data["phi_right"],
        )
        J = quad_results[0] * constant
    else:
        B = data["B"]
        integrand_on_grid = np.sqrt(np.maximum(0, 1 - B / B_bounce)) / B
        J = trapezoid(np.where(data["well_mask"], integrand_on_grid, 0.0), data["phi"]) * constant

    data["J"] = J
    return data


def _build_coordinate_arrays(ns: int, n_alpha: int, n_rho: int):
    alpha_values = np.linspace(0.0, 2.0 * np.pi, n_alpha, endpoint=False)
    rho_idx = np.unique(np.round(np.linspace(0.0, 1.0, n_rho) ** 2 * (ns - 1)).astype(int))
    rho_values = np.sqrt(rho_idx / (ns - 1))
    s_values = rho_idx / (ns - 1)
    return alpha_values, rho_values, s_values


def _compute_unrefined_j_from_cached_B(
    surf: BoozerSurface,
    B: np.ndarray,
    phi: np.ndarray,
    B_bounce: float,
    clipped_well_nan: bool = True,
    return_data: bool = False,
):
    """Compute J for ``refine=False`` from precomputed ``B(theta, phi)`` samples."""
    allowed = B <= B_bounce
    (
        has_allowed,
        well_crosses_left_edge,
        well_crosses_right_edge,
        left_index,
        right_index,
        well_mask,
    ) = _find_well_bounds_from_allowed(allowed)

    if not has_allowed:
        J = np.nan
    elif clipped_well_nan and (well_crosses_left_edge or well_crosses_right_edge):
        J = np.nan
    else:
        integrand_on_grid = np.sqrt(np.maximum(0, 1 - B / B_bounce)) / B
        constant = np.abs(surf.G + surf.I * surf.iota) / (surf.R00 * 2 * np.pi / surf.nfp)
        J = trapezoid(np.where(well_mask, integrand_on_grid, 0.0), phi) * constant

    if return_data:
        return {
            "J": J,
            "allowed": allowed,
            "well_crosses_left_edge": well_crosses_left_edge,
            "well_crosses_right_edge": well_crosses_right_edge,
            "left_index": left_index,
            "right_index": right_index,
            "well_mask": well_mask,
        }

    return J


def _compute_j_grids(
    booz: BoozerField,
    alpha_values: np.ndarray,
    s_values: np.ndarray,
    refine: bool,
    n_phi: int = DEFAULT_N_PHI,
    lambda_n_values: list[float] | None = None,
    return_b_extrema: bool = False,
):
    if lambda_n_values is None:
        lambda_n_values = LAMBDA_N_VALUES

    j_grids = {}

    if not refine:
        phi_center = np.pi / booz.nfp
        phi, surfaces, b_cache = _build_unrefined_b_cache(
            booz,
            alpha_values,
            s_values,
            phi_center,
            n_phi=n_phi,
        )
        for lambda_n in lambda_n_values:
            j_grids[lambda_n] = _compute_single_j_grid(
                booz,
                alpha_values,
                s_values,
                lambda_n=lambda_n,
                refine=refine,
                n_phi=n_phi,
                phi_center=phi_center,
                phi=phi,
                surfaces=surfaces,
                b_cache=b_cache,
            )
        if return_b_extrema:
            return j_grids, _compute_b_extrema(
                booz,
                alpha_values,
                s_values,
                phi_center,
                b_cache=b_cache,
            )
        return j_grids

    for lambda_n in lambda_n_values:
        j_grids[lambda_n] = _compute_single_j_grid(
            booz,
            alpha_values,
            s_values,
            lambda_n=lambda_n,
            refine=refine,
            n_phi=n_phi,
        )

    if return_b_extrema:
        return j_grids, _compute_b_extrema(
            booz,
            alpha_values,
            s_values,
            np.pi / booz.nfp,
        )

    return j_grids


def _build_unrefined_b_cache(
    booz: BoozerField,
    alpha_values: np.ndarray,
    s_values: np.ndarray,
    phi_center: float,
    n_phi: int = DEFAULT_N_PHI,
):
    phi_margin = 5.0
    phi_field_period = 2.0 * np.pi / booz.nfp
    phi = phi_center + np.linspace(-phi_margin - 0.5, phi_margin + 0.5, n_phi) * phi_field_period

    surfaces = [BoozerSurface(booz, s) for s in s_values]
    b_cache = np.empty((len(alpha_values), len(s_values), n_phi))
    for s_idx, surf in enumerate(surfaces):
        b_cache[:, s_idx, :] = surf.compute_B_tensor_alpha_phi(alpha_values, phi)
        # for a_idx, alpha in enumerate(alpha_values):
        #     theta_center = alpha + surf.iota * phi_center
        #     theta = theta_center + surf.iota * (phi - phi_center)
        #     b_cache[a_idx, s_idx, :] = surf.compute_B(theta, phi)

    return phi, surfaces, b_cache


def _compute_b_extrema(
    booz: BoozerField,
    alpha_values: np.ndarray,
    s_values: np.ndarray,
    phi_center: float,
    b_cache: np.ndarray | None = None,
):
    if b_cache is None:
        _, _, b_cache = _build_unrefined_b_cache(booz, alpha_values, s_values, phi_center)

    return {
        "min": np.min(b_cache, axis=(0, 2)),
        "max": np.max(b_cache, axis=(0, 2)),
    }


def _compute_single_j_grid(
    booz: BoozerField,
    alpha_values: np.ndarray,
    s_values: np.ndarray,
    lambda_n: float,
    refine: bool,
    n_phi: int = DEFAULT_N_PHI,
    phi_center: float | None = None,
    phi: np.ndarray | None = None,
    surfaces: list[BoozerSurface] | None = None,
    b_cache: np.ndarray | None = None,
):
    b_min, b_max = booz.get_min_max()
    if phi_center is None:
        phi_center = np.pi / booz.nfp
    print(f"Processing lambda_n = {lambda_n}")
    b_bounce = b_min + lambda_n * (b_max - b_min)

    if refine:
        j_grid = np.full((len(alpha_values), len(s_values)), np.nan)
        for s_idx, s in enumerate(s_values):
            surf = BoozerSurface(booz, s)
            for a_idx, alpha in enumerate(alpha_values):
                # alpha = theta - iota * phi, so theta_center = alpha + iota * phi_center.
                theta_center = alpha + surf.iota * phi_center
                data = compute_J_invariant(
                    surf,
                    b_bounce,
                    theta_center,
                    phi_center,
                    n_phi=n_phi,
                    refine=refine,
                )
                j_grid[a_idx, s_idx] = data["J"]
        return j_grid

    if phi is None or surfaces is None or b_cache is None:
        phi, surfaces, b_cache = _build_unrefined_b_cache(
            booz,
            alpha_values,
            s_values,
            phi_center,
            n_phi=n_phi,
        )

    j_grid = np.full((len(alpha_values), len(s_values)), np.nan)
    for s_idx, surf in enumerate(surfaces):
        for a_idx in range(len(alpha_values)):
            j_grid[a_idx, s_idx] = _compute_unrefined_j_from_cached_B(
                surf=surf,
                B=b_cache[a_idx, s_idx, :],
                phi=phi,
                B_bounce=b_bounce,
                clipped_well_nan=True,
            )

    return j_grid


def _make_closed_alpha_grid(
    alpha_values: np.ndarray,
    rho_values: np.ndarray,
    j_grid: np.ndarray,
):
    alpha_plot = np.append(alpha_values, 2.0 * np.pi)
    j_plot = np.vstack([j_grid, j_grid[0:1, :]])
    rho_mesh, alpha_mesh = np.meshgrid(rho_values, alpha_plot)
    return rho_mesh, alpha_mesh, j_plot


def _add_footer(fig):
    fig.text(
        0.5,
        0.005,
        os.path.abspath(__file__),
        ha="center",
        fontsize=FOOTER_FONT_SIZE,
        color="grey",
    )


def _add_title_block(fig, title_text: str, boozmn_path: Path):
    fig.suptitle(title_text, fontsize=SUPTITLE_FONT_SIZE, y=0.995)
    fig.text(
        0.5,
        0.965,
        str(boozmn_path),
        ha="center",
        va="top",
        fontsize=PATH_FONT_SIZE,
    )


def _build_figure_grid(count: int):
    ncols = max(1, math.ceil(math.sqrt(count * 16 / 9)))
    nrows = math.ceil(count / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, 8.1), constrained_layout=False)
    fig.subplots_adjust(
        left=SUBPLOT_LEFT,
        bottom=SUBPLOT_BOTTOM,
        right=SUBPLOT_RIGHT,
        top=SUBPLOT_TOP,
        wspace=SUBPLOT_WSPACE,
        hspace=SUBPLOT_HSPACE,
    )
    return fig, np.atleast_1d(axes).ravel()


def _style_axis(ax):
    ax.tick_params(labelsize=TICK_FONT_SIZE)


def _format_colorbar(colorbar):
    colorbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    colorbar.formatter = ticker.FuncFormatter(lambda value, _: f"{value:.2g}")
    colorbar.update_ticks()


def _get_subplot_contour_data(
    j_plot: np.ndarray,
    scale_kind: str,
    contour_levels: int,
    vmin: float | None = None,
    vmax: float | None = None,
):
    finite_values = j_plot[np.isfinite(j_plot)]
    if finite_values.size == 0:
        raise ValueError("Cannot plot an all-NaN J grid.")

    if scale_kind == "log":
        positive_values = finite_values[finite_values > 0]
        if positive_values.size == 0:
            raise ValueError("Cannot use a logarithmic color scale when J has no positive values.")
        vmin = positive_values.min()
        vmax = positive_values.max()
        if math.isclose(vmin, vmax):
            vmax = vmin * 10.0
        levels = np.geomspace(vmin, vmax, contour_levels)
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        plot_data = np.ma.masked_less_equal(j_plot, 0)
    else:
        if vmin is None:
            vmin = finite_values.min()
        if vmax is None:
            vmax = finite_values.max()
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-12
        levels = np.linspace(vmin, vmax, contour_levels)
        norm = None
        plot_data = j_plot

    return plot_data, levels, norm


def _build_single_plot_figure():
    fig = plt.figure(figsize=(9.8, 8.1), constrained_layout=False)
    fig.subplots_adjust(left=0.08, bottom=0.22, right=0.9, top=0.91)
    ax = fig.add_subplot(111)
    return fig, ax


def _style_cartesian_polar_axis(ax):
    tick_values = np.linspace(-1.0, 1.0, 5)
    ax.set_xlabel(r"$x=\rho\cos\alpha$", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(r"$y=\rho\sin\alpha$", fontsize=LABEL_FONT_SIZE)
    ax.set_aspect("equal")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks(tick_values)
    ax.set_yticks(tick_values)
    ax.grid(False)
    _style_axis(ax)


def _plot_single_lambda_gui(
    alpha_values: np.ndarray,
    rho_values: np.ndarray,
    j_grid: np.ndarray,
    contour_levels: int,
    boozmn_path: Path,
    lambda_n: float,
    n_alpha: int,
    n_rho: int,
    refine: bool,
):
    rho_mesh, alpha_mesh, j_plot = _make_closed_alpha_grid(alpha_values, rho_values, j_grid)
    finite_values = j_plot[np.isfinite(j_plot)]
    if finite_values.size == 0:
        raise ValueError("Cannot plot an all-NaN J grid.")

    data_min = float(finite_values.min())
    data_max = float(finite_values.max())
    if math.isclose(data_min, data_max):
        data_max = data_min + 1e-12

    fig, ax = _build_single_plot_figure()
    _add_title_block(
        fig,
        rf"Interactive $J$ polar plot near $\phi=\pi/n_{{fp}}$ ($\lambda_n={lambda_n:.2f}$, n_alpha={n_alpha}, n_rho={n_rho}, refine={refine})",
        boozmn_path,
    )
    _add_footer(fig)

    range_ax = fig.add_axes([0.12, 0.11, 0.62, 0.035])
    contour_ax = fig.add_axes([0.12, 0.055, 0.62, 0.035])
    filled_ax = fig.add_axes([0.84, 0.012, 0.14, 0.08])
    color_range = RangeSlider(
        range_ax,
        "J range",
        data_min,
        data_max,
        valinit=(data_min, data_max),
    )
    contour_slider = Slider(
        contour_ax,
        "Contours",
        valmin=2,
        valmax=60,
        valinit=int(contour_levels),
        valstep=1,
    )
    filled_toggle = CheckButtons(filled_ax, ["Filled"], [True])

    state = {"colorbar": None}
    x_vals = rho_mesh * np.cos(alpha_mesh)
    y_vals = rho_mesh * np.sin(alpha_mesh)

    def _redraw(_=None):
        if state["colorbar"] is not None:
            state["colorbar"].remove()
            state["colorbar"] = None

        ax.clear()
        plot_data, _, _ = _get_subplot_contour_data(
            j_plot,
            "linear",
            int(contour_slider.val),
            vmin=float(color_range.val[0]),
            vmax=float(color_range.val[1]),
        )
        clever_levels = _compute_clever_levels(plot_data, int(contour_slider.val))
        if filled_toggle.get_status()[0]:
            contour = ax.contourf(
                x_vals,
                y_vals,
                plot_data,
                levels=clever_levels,
                colors=_sample_level_colors(clever_levels, filled=True),
                extend="both",
            )
        else:
            contour = ax.contour(
                x_vals,
                y_vals,
                plot_data,
                levels=clever_levels,
                colors=_sample_level_colors(clever_levels, filled=False),
                extend="both",
            )
        _style_cartesian_polar_axis(ax)
        state["colorbar"] = fig.colorbar(contour, ax=ax, pad=0.12, shrink=0.9)
        # _format_colorbar(state["colorbar"])
        fig.canvas.draw_idle()

    color_range.on_changed(_redraw)
    contour_slider.on_changed(_redraw)
    filled_toggle.on_clicked(_redraw)
    fig._single_lambda_widgets = {
        "color_range": color_range,
        "contour_slider": contour_slider,
        "filled_toggle": filled_toggle,
        "state": state,
    }
    _redraw()
    return fig


def _plot_polar_figure(
    alpha_values: np.ndarray,
    rho_values: np.ndarray,
    j_grids: dict,
    lambda_n_values: list[float],
    contour_levels: int,
    scale_kind: str,
    output_path: Path,
    boozmn_path: Path,
    n_alpha: int,
    n_rho: int,
    n_phi: int = DEFAULT_N_PHI,
    refine: bool = True,
):
    fig, axes = _build_figure_grid(len(lambda_n_values))
    for ax, lambda_n in zip(axes, lambda_n_values):
        rho_mesh, alpha_mesh, j_plot = _make_closed_alpha_grid(
            alpha_values,
            rho_values,
            j_grids[lambda_n],
        )
        plot_data, _, _ = _get_subplot_contour_data(
            j_plot,
            scale_kind,
            contour_levels,
        )
        clever_levels = _compute_clever_levels(plot_data, contour_levels)
        x_vals = rho_mesh * np.cos(alpha_mesh)
        y_vals = rho_mesh * np.sin(alpha_mesh)
        contour = ax.contourf(
            x_vals,
            y_vals,
            plot_data,
            levels=clever_levels,
            colors=_sample_level_colors(clever_levels, filled=True),
            extend="both",
        )
        ax.set_title(rf"$\lambda_n={lambda_n:.2f}$", fontsize=SUBPLOT_TITLE_FONT_SIZE)
        _style_cartesian_polar_axis(ax)
        colorbar = fig.colorbar(contour, ax=ax, shrink=0.82, pad=0.02)
        _format_colorbar(colorbar)

    for ax in axes[len(lambda_n_values):]:
        ax.axis("off")

    _add_title_block(
        fig,
        rf"$J$ on polar coordinates near $\phi=\pi/n_{{fp}}$ (n_alpha={n_alpha}, n_rho={n_rho}, n_phi={n_phi}, refine={refine})",
        boozmn_path,
    )
    _add_footer(fig)
    fig.savefig(output_path)
    return fig


def _compute_clever_levels(plot_data: np.ndarray, contour_levels: int) -> np.ndarray:
    finite_values = np.asarray(plot_data)[np.isfinite(plot_data)]
    if finite_values.size == 0:
        raise ValueError("Cannot compute contour levels for an all-NaN J grid.")

    level_quantiles = np.linspace(0.0, 1.0, int(contour_levels))
    clever_levels = np.quantile(finite_values, level_quantiles)
    clever_levels = np.unique(clever_levels)
    if clever_levels.size < 2:
        vmin = float(finite_values.min())
        vmax = float(finite_values.max())
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-12
        clever_levels = np.linspace(vmin, vmax, 2)

    return clever_levels


def _sample_level_colors(levels: np.ndarray, filled: bool, cmap_name: str = "viridis"):
    """Sample colormap colors uniformly by contour level index."""
    n_colors = max(1, len(levels) - 1) if filled else max(1, len(levels))
    cmap = plt.get_cmap(cmap_name)
    if n_colors == 1:
        return [cmap(0.5)]
    return cmap(np.linspace(0.0, 1.0, n_colors))


def _add_large_plot_background_circles(ax):
    g = 0.8
    radii = np.arange(0.0, 1.1, 0.1)
    for radius in radii:
        ax.add_patch(
            plt.Circle(
                (0.0, 0.0),
                radius,
                fill=False,
                lw=0.5,
                color=(g, g, g),
                zorder=0,
            )
        )


def _plot_b_extrema_subplot(
    ax,
    rho_values: np.ndarray,
    b_extrema: dict | None,
    lambda_n_values: list[float],
    b_min: float,
    b_max: float,
):
    if b_extrema is None:
        ax.axis("off")
        return

    ax.plot(rho_values, b_extrema["max"], label=r"$\max(B)$", linewidth=1.8)
    ax.plot(rho_values, b_extrema["min"], label=r"$\min(B)$", linewidth=1.8)
    lambda_norm = colors.Normalize(
        vmin=min(lambda_n_values),
        vmax=max(lambda_n_values) if max(lambda_n_values) > min(lambda_n_values) else min(lambda_n_values) + 1e-12,
    )
    lambda_cmap = plt.get_cmap("jet")
    for lambda_n in reversed(lambda_n_values):
        b_bounce = b_min + lambda_n * (b_max - b_min)
        ax.hlines(
            b_bounce,
            xmin=0.0,
            xmax=1.0,
            linewidth=0.9,
            linestyles="--",
            colors=[lambda_cmap(lambda_norm(lambda_n))],
            label=rf"$\lambda_n={lambda_n:.2f}$",
        )
    ax.set_title(r"$B$ extrema", fontsize=SUBPLOT_TITLE_FONT_SIZE)
    ax.set_xlabel(r"$\rho$", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(r"$B$", fontsize=LABEL_FONT_SIZE)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(
        fontsize=TICK_FONT_SIZE,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )
    _style_axis(ax)


def _plot_large_polar_figures(
    alpha_values: np.ndarray,
    rho_values: np.ndarray,
    j_grids: dict,
    lambda_n_values: list[float],
    contour_levels: int,
    output_base_path: Path,
    boozmn_path: Path,
    n_alpha: int,
    n_rho: int,
    n_phi: int = DEFAULT_N_PHI,
    refine: bool = True,
    b_extrema: dict | None = None,
    b_min: float | None = None,
    b_max: float | None = None,
    n_rows: int = 2,
    n_cols: int = 3,
):
    figures = []
    per_figure = n_rows * n_cols
    for page_idx, start in enumerate(range(0, len(lambda_n_values), per_figure), start=1):
        lambda_chunk = lambda_n_values[start : start + per_figure]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.5, 8.1), constrained_layout=False)
        flat_axes = np.atleast_1d(axes).ravel()
        include_b_extrema = b_extrema is not None and start + per_figure >= len(lambda_n_values)
        plot_axes = flat_axes[:-1] if include_b_extrema else flat_axes

        for ax, lambda_n in zip(plot_axes, lambda_chunk):
            rho_mesh, alpha_mesh, j_plot = _make_closed_alpha_grid(
                alpha_values,
                rho_values,
                j_grids[lambda_n],
            )
            plot_data, _, _ = _get_subplot_contour_data(
                j_plot,
                "linear",
                contour_levels,
            )
            clever_levels = _compute_clever_levels(plot_data, contour_levels)
            x_vals = rho_mesh * np.cos(alpha_mesh)
            y_vals = rho_mesh * np.sin(alpha_mesh)
            _add_large_plot_background_circles(ax)
            contour = ax.contour(
                x_vals,
                y_vals,
                plot_data,
                levels=clever_levels,
                colors=_sample_level_colors(clever_levels, filled=False),
                extend="both",
            )
            ax.set_title(rf"$\lambda_n={lambda_n:.2f}$", fontsize=SUBPLOT_TITLE_FONT_SIZE)
            _style_cartesian_polar_axis(ax)
            colorbar = fig.colorbar(contour, ax=ax, shrink=0.82, pad=0.02)
            colorbar.ax.tick_params(labelsize=TICK_FONT_SIZE)

        if include_b_extrema:
            _plot_b_extrema_subplot(
                flat_axes[-1],
                rho_values,
                b_extrema,
                lambda_n_values,
                float(b_min),
                float(b_max),
            )
            unused_axes = plot_axes[len(lambda_chunk) :]
        else:
            unused_axes = flat_axes[len(lambda_chunk) :]

        for ax in unused_axes:
            ax.axis("off")

        _add_title_block(
            fig,
            rf"$J$ on polar coordinates near $\phi=\pi/n_{{fp}}$ (page {page_idx}, n_alpha={n_alpha}, n_rho={n_rho}, n_phi={n_phi}, refine={refine})",
            boozmn_path,
        )
        _add_footer(fig)
        fig.tight_layout(rect=[0.0, 0.01, 1, 0.98])

        page_output = output_base_path.with_name(
            f"{output_base_path.stem}_page{page_idx:02d}{output_base_path.suffix}"
        )
        fig.savefig(page_output)
        figures.append(fig)

    return figures


def _plot_rho_alpha_figure(
    alpha_values: np.ndarray,
    rho_values: np.ndarray,
    j_grids: dict,
    contour_levels: int,
    scale_kind: str,
    output_path: Path,
    boozmn_path: Path,
    n_alpha: int,
    n_rho: int,
    refine: bool,
):
    fig, axes = _build_figure_grid(len(LAMBDA_N_VALUES))
    y_ticks = [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0, 2.0 * np.pi]
    y_labels = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]

    for ax, lambda_n in zip(axes, LAMBDA_N_VALUES):
        rho_mesh, alpha_mesh, j_plot = _make_closed_alpha_grid(
            alpha_values,
            rho_values,
            j_grids[lambda_n],
        )
        plot_data, levels, _ = _get_subplot_contour_data(
            j_plot,
            scale_kind,
            contour_levels,
        )
        contour = ax.contourf(
            rho_mesh,
            alpha_mesh,
            plot_data,
            levels=levels,
            colors=_sample_level_colors(levels, filled=True),
            extend="both",
        )
        ax.set_title(rf"$\lambda_n={lambda_n:.2f}$", fontsize=SUBPLOT_TITLE_FONT_SIZE)
        ax.set_xlabel(r"$\rho$", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel(r"$\alpha$", fontsize=LABEL_FONT_SIZE)
        ax.set_xlim(rho_values.min(), rho_values.max())
        ax.set_ylim(0.0, 2.0 * np.pi)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        _style_axis(ax)
        colorbar = fig.colorbar(contour, ax=ax, shrink=0.82, pad=0.02)
        _format_colorbar(colorbar)

    for ax in axes[len(LAMBDA_N_VALUES):]:
        ax.axis("off")

    _add_title_block(
        fig,
        rf"$J$ on $(\rho, \alpha)$ coordinates near $\phi=\pi/n_{{fp}}$ ({scale_kind} color scale, n_alpha={n_alpha}, n_rho={n_rho}, refine={refine})",
        boozmn_path,
    )
    _add_footer(fig)
    fig.savefig(output_path)
    return fig


def _save_combined_pdf(figures: list, output_path: Path):
    """Save all generated figures into one multi-page PDF."""
    if not figures:
        return

    with PdfPages(output_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)


def plot_J_invariant(
    boozmn_file: str,
    n_alpha: int = DEFAULT_N_ALPHA,
    n_rho: int = DEFAULT_N_RHO,
    n_phi: int = DEFAULT_N_PHI,
    contour_levels: int = DEFAULT_CONTOUR_LEVELS,
    refine: bool = True,
    show: bool = True,
):
    boozmn_path = Path(boozmn_file).expanduser().resolve()
    booz = BoozerField.from_boozmn(boozmn_path)
    ns = len(booz.s_full)
    b_min, b_max = booz.get_min_max()

    alpha_values, rho_values, s_values = _build_coordinate_arrays(ns, n_alpha, n_rho)

    print(f"Loading {boozmn_path}")
    print("Computing J grids for all lambda_n values")
    all_lambda_n_values = sorted(set(LAMBDA_N_VALUES + LARGE_PLOT_LAMBDA_N_VALUES))
    j_grids, b_extrema = _compute_j_grids(
        booz,
        alpha_values,
        s_values,
        refine=refine,
        n_phi=n_phi,
        lambda_n_values=all_lambda_n_values,
        return_b_extrema=True,
    )

    refine_tag = "true" if refine else "false"
    output_tag = f"_nalpha{n_alpha}_nrho{n_rho}_nphi{n_phi}_refine_{refine_tag}"

    figures = []
    summary_output_path = boozmn_path.with_name(f"{boozmn_path.stem}_J_polar_{output_tag}.pdf")
    print(f"Saving {summary_output_path.name}")
    figures.append(
        _plot_polar_figure(
            alpha_values,
            rho_values,
            j_grids,
            LAMBDA_N_VALUES,
            contour_levels,
            "linear",
            summary_output_path,
            boozmn_path,
            n_alpha,
            n_rho,
            n_phi,
            refine,
        )
    )

    large_output_base_path = boozmn_path.with_name(
        f"{boozmn_path.stem}_J_polar_linear_large{output_tag}.pdf"
    )
    print(
        f"Saving paginated large polar plots (n_rows=2, n_cols=3) with base name {large_output_base_path.name}"
    )
    figures.extend(
        _plot_large_polar_figures(
            alpha_values,
            rho_values,
            j_grids,
            LARGE_PLOT_LAMBDA_N_VALUES,
            contour_levels,
            large_output_base_path,
            boozmn_path,
            n_alpha,
            n_rho,
            n_phi,
            refine,
            b_extrema=b_extrema,
            b_min=b_min,
            b_max=b_max,
            n_rows=2,
            n_cols=3,
        )
    )

    combined_output_path = boozmn_path.with_name(
        f"{boozmn_path.stem}_J_polar_combined{output_tag}.pdf"
    )
    print(f"Saving combined PDF {combined_output_path.name}")
    _save_combined_pdf(figures, combined_output_path)

    if show:
        plt.show()
    for fig in figures:
        plt.close(fig)


def plot_J_invariant_single_lambda(
    boozmn_file: str,
    lambda_n: float,
    n_alpha: int = DEFAULT_N_ALPHA,
    n_rho: int = DEFAULT_N_RHO,
    n_phi: int = DEFAULT_N_PHI,
    contour_levels: int = DEFAULT_CONTOUR_LEVELS,
    refine: bool = True,
    show: bool = True,
):
    boozmn_path = Path(boozmn_file).expanduser().resolve()
    booz = BoozerField.from_boozmn(boozmn_path)
    ns = len(booz.s_full)

    alpha_values, rho_values, s_values = _build_coordinate_arrays(ns, n_alpha, n_rho)

    print(f"Loading {boozmn_path}")
    print(f"Computing J grid for lambda_n = {lambda_n}")
    j_grid = _compute_single_j_grid(
        booz,
        alpha_values,
        s_values,
        lambda_n=lambda_n,
        refine=refine,
        n_phi=n_phi,
    )

    fig = _plot_single_lambda_gui(
        alpha_values,
        rho_values,
        j_grid,
        contour_levels,
        boozmn_path,
        lambda_n,
        n_alpha,
        n_rho,
        refine,
    )

    if show:
        plt.show()
    plt.close(fig)


def plot_J_invariant_cli(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="plot_J_invariant",
        description="Compute J from a boozmn file and save polar subplot grids.",
    )
    parser.add_argument("boozmn_file", help="Path to a boozmn*.nc file")
    parser.add_argument(
        "--n_alpha",
        type=int,
        default=DEFAULT_N_ALPHA,
        help="Number of alpha values",
    )
    parser.add_argument(
        "--n_rho",
        type=int,
        default=DEFAULT_N_RHO,
        help="Number of rho grid values",
    )
    parser.add_argument(
        "--n_phi",
        type=int,
        default=DEFAULT_N_PHI,
        help="Number of phi values for J integration",
    )
    parser.add_argument(
        "--contour_levels",
        type=int,
        default=DEFAULT_CONTOUR_LEVELS,
        help="Number of contour levels",
    )
    parser.add_argument(
        "--refine",
        dest="refine",
        action="store_true",
        default=False,
        help="Refine bounce points with root finding",
    )
    parser.add_argument(
        "--no-refine",
        dest="refine",
        action="store_false",
        help="Disable root refinement (default)",
    )

    args = parser.parse_args(argv)
    plot_J_invariant(
        args.boozmn_file,
        n_alpha=args.n_alpha,
        n_rho=args.n_rho,
        n_phi=args.n_phi,
        contour_levels=args.contour_levels,
        refine=args.refine,
    )
    return 0


def plot_J_invariant_single_lambda_cli(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="plot_J_invariant_single_lambda",
        description="Compute J for one lambda_n value and show an interactive polar contour plot.",
    )
    parser.add_argument("boozmn_file", help="Path to a boozmn*.nc file")
    parser.add_argument(
        "lambda_n",
        type=float,
        help="Normalized bounce-field parameter to plot",
    )
    parser.add_argument(
        "--n_alpha",
        type=int,
        default=120,
        help="Number of alpha values",
    )
    parser.add_argument(
        "--n_rho",
        type=int,
        default=61,
        help="Number of rho grid values",
    )
    parser.add_argument(
        "--n_phi",
        type=int,
        default=DEFAULT_N_PHI,
        help="Number of phi values for J integration",
    )
    parser.add_argument(
        "--contour_levels",
        type=int,
        default=DEFAULT_CONTOUR_LEVELS,
        help="Initial number of contour levels",
    )
    parser.add_argument(
        "--refine",
        dest="refine",
        action="store_true",
        default=False,
        help="Refine bounce points with root finding",
    )
    parser.add_argument(
        "--no-refine",
        dest="refine",
        action="store_false",
        help="Disable root refinement (default)",
    )

    args = parser.parse_args(argv)
    plot_J_invariant_single_lambda(
        args.boozmn_file,
        args.lambda_n,
        n_alpha=args.n_alpha,
        n_rho=args.n_rho,
        n_phi=args.n_phi,
        contour_levels=args.contour_levels,
        refine=args.refine,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(plot_J_invariant_cli())