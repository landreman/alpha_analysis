import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import ticker
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
LAMBDA_N_VALUES = np.arange(0.1, 1.0, 0.05).tolist()
DEFAULT_N_ALPHA = 20
DEFAULT_N_RHO = 11
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
):
    b_min, b_max = booz.get_min_max()
    phi_center = np.pi / booz.nfp
    j_grids = {}

    if refine:
        for lambda_n in LAMBDA_N_VALUES:
            print(f"Processing lambda_n = {lambda_n}")
            b_bounce = b_min + lambda_n * (b_max - b_min)
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
                        refine=refine,
                    )
                    j_grid[a_idx, s_idx] = data["J"]
            j_grids[lambda_n] = j_grid
        return j_grids

    n_phi = 501
    phi_margin = 5.0
    phi_field_period = 2.0 * np.pi / booz.nfp
    phi = phi_center + np.linspace(-phi_margin - 0.5, phi_margin + 0.5, n_phi) * phi_field_period

    surfaces = [BoozerSurface(booz, s) for s in s_values]
    B_cache = np.empty((len(alpha_values), len(s_values), n_phi))
    for s_idx, surf in enumerate(surfaces):
        for a_idx, alpha in enumerate(alpha_values):
            theta_center = alpha + surf.iota * phi_center
            theta = theta_center + surf.iota * (phi - phi_center)
            B_cache[a_idx, s_idx, :] = surf.compute_B(theta, phi)

    for lambda_n in LAMBDA_N_VALUES:
        print(f"Processing lambda_n = {lambda_n}")
        b_bounce = b_min + lambda_n * (b_max - b_min)
        j_grid = np.full((len(alpha_values), len(s_values)), np.nan)
        for s_idx, surf in enumerate(surfaces):
            for a_idx in range(len(alpha_values)):
                j_grid[a_idx, s_idx] = _compute_unrefined_j_from_cached_B(
                    surf=surf,
                    B=B_cache[a_idx, s_idx, :],
                    phi=phi,
                    B_bounce=b_bounce,
                    clipped_well_nan=True,
                )
        j_grids[lambda_n] = j_grid

    return j_grids


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
        vmin = finite_values.min()
        vmax = finite_values.max()
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-12
        levels = np.linspace(vmin, vmax, contour_levels)
        norm = None
        plot_data = j_plot

    return plot_data, levels, norm


def _plot_polar_figure(
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
    for ax, lambda_n in zip(axes, LAMBDA_N_VALUES):
        rho_mesh, alpha_mesh, j_plot = _make_closed_alpha_grid(
            alpha_values,
            rho_values,
            j_grids[lambda_n],
        )
        plot_data, levels, norm = _get_subplot_contour_data(
            j_plot,
            scale_kind,
            contour_levels,
        )
        x_vals = rho_mesh * np.cos(alpha_mesh)
        y_vals = rho_mesh * np.sin(alpha_mesh)
        contour = ax.contourf(
            x_vals,
            y_vals,
            plot_data,
            levels=levels,
            norm=norm,
            cmap="viridis",
            extend="both",
        )
        ax.set_aspect("equal")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(rf"$\lambda_n={lambda_n:.2f}$", fontsize=SUBPLOT_TITLE_FONT_SIZE)
        ax.set_xlabel(r"$\rho\cos(\alpha)$", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel(r"$\rho\sin(\alpha)$", fontsize=LABEL_FONT_SIZE)
        _style_axis(ax)
        colorbar = fig.colorbar(contour, ax=ax, shrink=0.82, pad=0.02)
        _format_colorbar(colorbar)

    for ax in axes[len(LAMBDA_N_VALUES):]:
        ax.axis("off")

    _add_title_block(
        fig,
        rf"$J$ on polar coordinates near $\phi=\pi/n_{{fp}}$ ({scale_kind} color scale, n_alpha={n_alpha}, n_rho={n_rho}, refine={refine})",
        boozmn_path,
    )
    _add_footer(fig)
    fig.savefig(output_path)
    return fig


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
        plot_data, levels, norm = _get_subplot_contour_data(
            j_plot,
            scale_kind,
            contour_levels,
        )
        contour = ax.contourf(
            rho_mesh,
            alpha_mesh,
            plot_data,
            levels=levels,
            norm=norm,
            cmap="viridis",
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


def plot_J_invariant(
    boozmn_file: str,
    n_alpha: int = DEFAULT_N_ALPHA,
    n_rho: int = DEFAULT_N_RHO,
    contour_levels: int = DEFAULT_CONTOUR_LEVELS,
    refine: bool = True,
    show: bool = True,
):
    boozmn_path = Path(boozmn_file).expanduser().resolve()
    booz = BoozerField.from_boozmn(boozmn_path)
    ns = len(booz.s_full)

    alpha_values, rho_values, s_values = _build_coordinate_arrays(ns, n_alpha, n_rho)

    print(f"Loading {boozmn_path}")
    print("Computing J grids for all lambda_n values")
    j_grids = _compute_j_grids(booz, alpha_values, s_values, refine=refine)

    refine_tag = "true" if refine else "false"
    output_tag = f"_nalpha{n_alpha}_nrho{n_rho}_refine_{refine_tag}"

    figure_outputs = [
        (
            _plot_polar_figure,
            "linear",
            boozmn_path.with_name(f"{boozmn_path.stem}_J_polar_linear{output_tag}.pdf"),
        ),
        (
            _plot_polar_figure,
            "log",
            boozmn_path.with_name(f"{boozmn_path.stem}_J_polar_log{output_tag}.pdf"),
        ),
        (
            _plot_rho_alpha_figure,
            "linear",
            boozmn_path.with_name(f"{boozmn_path.stem}_J_rho_alpha_linear{output_tag}.pdf"),
        ),
        (
            _plot_rho_alpha_figure,
            "log",
            boozmn_path.with_name(f"{boozmn_path.stem}_J_rho_alpha_log{output_tag}.pdf"),
        ),
    ]

    figures = []
    for plot_function, scale_kind, output_path in figure_outputs:
        print(f"Saving {output_path.name}")
        figures.append(
            plot_function(
                alpha_values,
                rho_values,
                j_grids,
                contour_levels,
                scale_kind,
                output_path,
                boozmn_path,
                n_alpha,
                n_rho,
                refine,
            )
        )

    if show:
        plt.show()
    for fig in figures:
        plt.close(fig)


def plot_J_invariant_cli(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="plot_J_invariant",
        description="Compute J from a boozmn file and save polar and rho-alpha subplot grids.",
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
        contour_levels=args.contour_levels,
        refine=args.refine,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(plot_J_invariant_cli())