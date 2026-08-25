"""Static diagnostics for J-connectivity numerical stages (DESIGN.md §17)."""

from __future__ import annotations

import numpy as np

from .denominator import DenominatorConvergence, GlobalBBounds


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
