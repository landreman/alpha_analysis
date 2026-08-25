"""Generate a structured or Gmsh background-mesh diagnostic (Milestones 3–4)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from alpha_analysis.j_connectivity.background_mesh import (  # noqa: E402
    BackgroundMeshConfig,
    GmshBackgroundMeshBackend,
    GmshBackgroundMeshConfig,
    StructuredPrismMeshBackend,
)
from alpha_analysis.j_connectivity.synthetic_fields import (  # noqa: E402
    SyntheticFourierField,
)
from alpha_analysis.j_connectivity.visualization import (  # noqa: E402
    plot_background_mesh,
)


def manufactured_field() -> SyntheticFourierField:
    """Return an asymmetric field for coloring the authoritative mesh arrays."""
    return SyntheticFourierField(
        nfp=3,
        m=np.array([0, 1, 2]),
        n=np.array([0, 3, -3]),
        cosine_coefficients=np.array([[2.2, 0.1], [0.0, 0.2], [0.0, 0.05]]),
        sine_coefficients=np.array([[0.0, 0.0], [0.0, 0.08], [0.0, -0.03]]),
        iota_coefficients=np.array([0.7, 0.1]),
        G_coefficients=np.array([4.0]),
        I_coefficients=np.array([0.0]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("background_mesh.png"))
    parser.add_argument(
        "--backend", choices=("structured", "gmsh"), default="structured"
    )
    args = parser.parse_args()

    field = manufactured_field()
    if args.backend == "gmsh":
        mesh = GmshBackgroundMeshBackend(
            GmshBackgroundMeshConfig(target_size=0.3, axis_size=0.15, axis_radius=0.25)
        ).build(field)
    else:
        mesh = StructuredPrismMeshBackend(BackgroundMeshConfig(4, 16, 4)).build(field)
    figure, _ = plot_background_mesh(mesh)
    figure.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
