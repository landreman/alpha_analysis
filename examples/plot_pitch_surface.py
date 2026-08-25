"""Generate the Milestone 5 fixed-``B`` surface diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from alpha_analysis.j_connectivity.background_mesh import (  # noqa: E402
    BackgroundMeshConfig,
    StructuredPrismMeshBackend,
)
from alpha_analysis.j_connectivity.surface_extract import (  # noqa: E402
    MarchingTetrahedraExtractor,
)
from alpha_analysis.j_connectivity.synthetic_fields import (  # noqa: E402
    SyntheticFourierField,
)
from alpha_analysis.j_connectivity.visualization import (  # noqa: E402
    plot_pitch_surface,
)


def manufactured_field() -> SyntheticFourierField:
    """Return an asymmetric field with a closed fixed-``B`` torus."""
    return SyntheticFourierField(
        nfp=3,
        m=np.array([0, 0, 1]),
        n=np.array([0, 3, 3]),
        cosine_coefficients=np.array([[2.0, 0.3], [0.1, 0.0], [0.0, 0.0]]),
        sine_coefficients=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.03]]),
        iota_coefficients=np.array([0.7]),
        G_coefficients=np.array([-2.0]),
        I_coefficients=np.array([0.0]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("pitch_surface.png"))
    parser.add_argument("--b", type=float, default=2.15)
    args = parser.parse_args()

    field = manufactured_field()
    background = StructuredPrismMeshBackend(
        BackgroundMeshConfig(n_radial=10, n_poloidal=20, n_zeta=13)
    ).build(field)
    extraction = MarchingTetrahedraExtractor().extract(background, field, args.b)
    figure, _ = plot_pitch_surface(extraction)
    figure.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
