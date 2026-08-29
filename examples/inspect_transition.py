"""Generate the milestone-9 synthetic transition diagnostic.

Run from the repository root with ``python examples/inspect_transition.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alpha_analysis.j_connectivity import (
    SurfaceCurveMesh,
    SurfaceMesh,
    extract_critical_curves,
    map_transitions,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField
from alpha_analysis.j_connectivity.visualization import plot_transition_diagnostics

field = SyntheticFourierField(
    nfp=1,
    m=np.array([0, 0, 0]),
    n=np.array([0, 1, 2]),
    cosine_coefficients=np.array([[2.0, 0.0], [-1.0, 0.0], [0.3, 0.2]]),
    sine_coefficients=np.zeros((3, 2)),
    iota_coefficients=np.array([0.4]),
    G_coefficients=np.array([3.0]),
    I_coefficients=np.array([0.0]),
)
b = 1.4
s = 0.5
theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
points = np.column_stack(
    (
        np.sqrt(s) * np.cos(theta),
        np.sqrt(s) * np.sin(theta),
        np.zeros_like(theta),
    )
)
ids = np.arange(len(points))
curve = SurfaceCurveMesh(
    period=2.0 * np.pi,
    points=points,
    segments=np.column_stack((ids, np.roll(ids, -1))),
    B=np.full(len(points), b),
    g=np.zeros(len(points)),
    boundary_tags=np.full(len(points), SurfaceMesh.G_ZERO, dtype=np.int64),
)
transition = map_transitions(field, extract_critical_curves(curve, field, b))[0]
output = Path("transition_diagnostic.png")
figure, _ = plot_transition_diagnostics(field, transition, output_path=output)
plt.close(figure)
print(f"wrote {output} ({transition.status.name})")
