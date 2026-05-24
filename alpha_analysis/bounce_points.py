import os
import argparse
import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

from alpha_analysis.boozer_field import BoozerField, BoozerSurface


def find_bounce_points(
    surf: BoozerSurface,
    B_bounce: float,
    theta_center: float,
    phi_center: float,
    n_phi: float = 501,
    phi_margin: float = 5.0,
    refine: bool = True,
) -> dict:

    args = {
        "surf": surf,
        "B_bounce": B_bounce,
        "theta_center": theta_center,
        "phi_center": phi_center,
        "n_phi": n_phi,
        "phi_margin": phi_margin,
        "refine": refine,
    }

    # Create grid of theta and phi values.
    phi_field_period = 2.0 * np.pi / surf.nfp
    phi = (
        phi_center
        + np.linspace(-phi_margin - 0.5, phi_margin + 0.5, n_phi) * phi_field_period
    )
    dphi = phi[1] - phi[0]
    theta = theta_center + surf.iota * (phi - phi_center)
    all_indices = np.arange(n_phi)

    B = surf.compute_B(theta, phi)
    allowed = B <= B_bounce
    if not np.any(allowed):
        data = {
            "B": B,
            "theta": theta,
            "phi": phi,
            "allowed": allowed,
            "well_crosses_left_edge": np.nan,
            "well_crosses_right_edge": np.nan,
            "left_index": np.nan,
            "right_index": np.nan,
            "well_mask": np.full(n_phi, False),
            "phi_left": np.nan,
            "phi_right": np.nan,
            "theta_left": np.nan,
            "theta_right": np.nan,
        } | args
        return data

    # Find the allowed point closest to the center point.
    center_index = np.argmin(np.abs(phi - phi_center))
    allowed_indices = np.where(allowed)[0]
    interior_index = allowed_indices[np.argmin(np.abs(allowed_indices - center_index))]

    # Find the first non-allowed points to the right of the interior point:
    temp = np.argwhere(np.logical_and(~allowed, all_indices > interior_index))
    well_crosses_right_edge = len(temp) == 0
    if well_crosses_right_edge:
        right_index = len(allowed) - 1
    else:
        right_index = temp[0, 0] - 1

    # Find the first non-allowed points to the left of the interior point:
    temp = np.argwhere(np.logical_and(~allowed, all_indices < interior_index))
    well_crosses_left_edge = len(temp) == 0
    if well_crosses_left_edge:
        left_index = 0
    else:
        left_index = temp[-1, 0] + 1

    well_mask = np.logical_and(all_indices >= left_index, all_indices <= right_index)

    # Function for finding the roots exactly:
    def B_residual(phi_val):
        theta_val = theta_center + surf.iota * (phi_val - phi_center)
        B_val = surf.compute_B([theta_val], [phi_val])[0]
        return B_val - B_bounce

    if well_crosses_left_edge or (not refine):
        phi_left = phi[left_index]
    else:
        root_solution = root_scalar(
            B_residual,
            x0=phi[left_index],
            x1=phi[left_index] - dphi,
        )
        phi_left = root_solution.root

    if well_crosses_right_edge or (not refine):
        phi_right = phi[right_index]
    else:
        root_solution = root_scalar(
            B_residual, x0=phi[right_index], x1=phi[right_index] + dphi
        )
        phi_right = root_solution.root

    theta_left = theta_center + surf.iota * (phi_left - phi_center)
    theta_right = theta_center + surf.iota * (phi_right - phi_center)

    # Update the data dictionary with the well crossing information.
    data = {
        "B": B,
        "theta": theta,
        "phi": phi,
        "allowed": allowed,
        "well_crosses_left_edge": well_crosses_left_edge,
        "well_crosses_right_edge": well_crosses_right_edge,
        "left_index": left_index,
        "right_index": right_index,
        "well_mask": well_mask,
        "phi_left": phi_left,
        "phi_right": phi_right,
        "theta_left": theta_left,
        "theta_right": theta_right,
    } | args
    return data


def _plot_bounce_points(
    surf: BoozerSurface,
    B_bounce: float,
    n_alpha: int,
    phi_center: float,
    n_phi: float = 501,
    phi_margin: float = 5.0,
    refine: bool = True,
    top_text: str = "",
    bottom_text: str = "",
    show: bool = True,
):
    alphas = np.linspace(0.0, 2.0 * np.pi, n_alpha)
    bounce_data = []
    for alpha in alphas:
        data = find_bounce_points(
            surf,
            B_bounce,
            alpha,
            phi_center,
            n_phi=n_phi,
            phi_margin=phi_margin,
            refine=refine,
        )
        bounce_data.append(data)

    ntheta_contours = 180
    nphi_contours = 201
    phi_field_period = 2.0 * np.pi / surf.nfp
    theta_margin = 1.0
    theta = np.linspace(-theta_margin, theta_margin + 2.0 * np.pi, ntheta_contours)
    phi = np.linspace(-0.5 * phi_field_period, 1.5 * phi_field_period, nphi_contours)
    phi2d, theta2d = np.meshgrid(phi, theta)

    B = surf.compute_B(theta2d, phi2d)

    plt.figure(figsize=(8, 8.1))
    plt.contour(phi2d, theta2d, B, levels=50)
    plt.colorbar()
    for alpha, data in zip(alphas, bounce_data):
        phi_left = data["phi_left"]
        phi_right = data["phi_right"]
        theta_left = data["theta_left"]
        theta_right = data["theta_right"]
        plt.plot(
            [phi_left, phi_right],
            [theta_left, theta_right],
            "-",
            color="k",
            linewidth=1.5,
        )
        plt.plot(phi_left, theta_left, "o", color="r", markersize=2)
        plt.plot(phi_right, theta_right, "o", color="b", markersize=2)

    plt.tight_layout()
    plt.figtext(0.5, 0.995, top_text, ha="center", va="top", fontsize=7)
    plt.figtext(0.5, 0.005, bottom_text, ha="center", va="bottom", fontsize=7)
    if show:
        plt.show()


def plot_bounce_points(
    filename: str,
    s: float,
    B_bounce: float,
    n_alpha: int = 50,
    n_phi: float = 501,
    phi_margin: float = 5.0,
    refine: bool = True,
    show: bool = True,
):
    booz = BoozerField.from_boozmn(filename)
    surf = BoozerSurface(booz, s)
    phi_center = np.pi / surf.nfp
    top_text = f"B_bounce={B_bounce:.3f}, s={s:.3f}, n_alpha={n_alpha}, n_phi={n_phi}, phi_margin={phi_margin}, refine={refine}"
    bottom_text = f"{os.path.abspath(filename)}"
    _plot_bounce_points(
        surf,
        B_bounce,
        n_alpha,
        phi_center,
        n_phi=n_phi,
        phi_margin=phi_margin,
        refine=refine,
        top_text=top_text,
        bottom_text=bottom_text,
        show=show,
    )


def plot_bounce_points_cli(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="plot_bounce_points",
        description="Plot trapped-particle bounce points for a boozmn file.",
    )
    parser.add_argument("boozmn_file", help="Path to boozmn netCDF file")
    parser.add_argument(
        "s", type=float, help="Normalized toroidal flux (surface label)"
    )
    parser.add_argument("B_bounce", type=float, help="Bounce-field threshold")
    parser.add_argument(
        "--n_alpha", type=int, default=50, help="Number of alpha values"
    )
    parser.add_argument(
        "--n_phi", type=int, default=501, help="Number of phi grid points"
    )
    parser.add_argument(
        "--phi_margin", type=float, default=5.0, help="Phi margin in field periods"
    )
    parser.add_argument(
        "--refine",
        dest="refine",
        action="store_true",
        default=True,
        help="Refine bounce points with root finding (default)",
    )
    parser.add_argument(
        "--no-refine",
        dest="refine",
        action="store_false",
        help="Disable root refinement",
    )

    args = parser.parse_args(argv)
    plot_bounce_points(
        args.boozmn_file,
        s=args.s,
        B_bounce=args.B_bounce,
        n_alpha=args.n_alpha,
        n_phi=args.n_phi,
        phi_margin=args.phi_margin,
        refine=args.refine,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(plot_bounce_points_cli())
