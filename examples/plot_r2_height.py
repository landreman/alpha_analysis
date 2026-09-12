"""Reproduce the R2 synthetic local-height contour diagnostic (DESIGN §17.3)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alpha_analysis.j_connectivity.branch_atlas import height_at
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField


def main():
    # At zeta=0 this analytic branch has H(s,alpha)=1.3+0.2s,
    # so H=b=1.4 is the exact line s=0.5 independent of alpha.
    field = SyntheticFourierField(
        1,
        np.zeros(3, dtype=int),
        np.array([0, 1, 2]),
        np.array([[2.0, 0.0], [-1.0, 0.0], [0.3, 0.2]]),
        np.zeros((3, 2)),
        np.array([0.0]),
        np.array([3.0]),
        np.array([0.0]),
    )
    b = 1.4
    s = np.linspace(0, 1, 41)
    alpha = np.linspace(0, 2 * np.pi, 65)
    H = np.array(
        [[height_at(field, float(x), float(a), 0.0).value for a in alpha] for x in s]
    )
    residual = float(np.max(np.abs(H - (1.3 + 0.2 * s[:, None]))))
    assert residual < 1e-12
    figure, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    image = ax.pcolormesh(
        alpha, s, H - b, shading="auto", cmap="coolwarm", vmin=-0.1, vmax=0.1
    )
    ax.contour(alpha, s, H - b, levels=[0], colors="black", linewidths=1.7)
    ax.scatter([0.2], [0.5], c="black", s=25, label="tested generic port")
    ax.set(
        xlabel="chart alpha [rad]",
        ylabel="s",
        xlim=(0, 2 * np.pi),
        ylim=(0, 1),
        title="Local maximum height H-b [B units]\n" "Fourier-model scope; black: H=b",
    )
    ax.legend(loc="upper right")
    figure.colorbar(image, ax=ax, label="H-b [B units]")
    path = Path("docs/validation/r2-atlas-plots/synthetic-height-contour.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"{path}: max analytic height residual {residual:.3e}")


if __name__ == "__main__":
    main()
