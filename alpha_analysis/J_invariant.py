import numpy as np
try:
    from numpy import trapezoid
except ImportError:
    # Older versions of numpy don't have trapezoid, so we can use trapz instead.
    from numpy import trapz as trapezoid

from scipy.integrate import quad

from .boozer_field import BoozerSurface
from .bounce_points import find_bounce_points

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
    constant = np.abs(surf.G + surf.I * surf.iota)
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