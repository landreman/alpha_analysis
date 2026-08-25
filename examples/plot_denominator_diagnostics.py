"""Generate Milestone 2 denominator and global-B convergence diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from alpha_analysis.j_connectivity.denominator import (  # noqa: E402
    BoundsConfig,
    DenominatorConfig,
    UniformSourceProfile,
    denominator_convergence,
    find_global_B_bounds,
)
from alpha_analysis.j_connectivity.synthetic_fields import (  # noqa: E402
    SyntheticFourierField,
)
from alpha_analysis.j_connectivity.visualization import (  # noqa: E402
    plot_B_extrema_profiles,
    plot_denominator_convergence,
)


def manufactured_field() -> SyntheticFourierField:
    """Return a positive asymmetric field with analytic Fourier derivatives."""
    return SyntheticFourierField(
        nfp=3,
        m=np.array([0, 0, 1, 2]),
        n=np.array([0, 3, 0, -3]),
        cosine_coefficients=np.array(
            [[2.2, 0.1], [0.12, 0.0], [0.25, -0.05], [0.04, 0.0]]
        ),
        sine_coefficients=np.array(
            [[0.0, 0.0], [0.03, 0.0], [0.08, 0.02], [-0.02, 0.0]]
        ),
        iota_coefficients=np.array([0.7, 0.1]),
        G_coefficients=np.array([4.0, -0.2]),
        I_coefficients=np.array([0.1]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("denominator_diagnostics"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    field = manufactured_field()
    configs = tuple(
        DenominatorConfig(n_s=n_s, n_theta=n_angle, n_zeta=n_angle)
        for n_s, n_angle in ((6, 8), (10, 16), (16, 32), (24, 64))
    )
    convergence = denominator_convergence(field, UniformSourceProfile(), configs)
    bounds = find_global_B_bounds(field, BoundsConfig(25, 48, 48))

    convergence_figure, _ = plot_denominator_convergence(convergence)
    extrema_figure, _ = plot_B_extrema_profiles(bounds)
    convergence_figure.savefig(args.output / "denominator_convergence.png", dpi=160)
    extrema_figure.savefig(args.output / "B_extrema_profiles.png", dpi=160)


if __name__ == "__main__":
    main()
