"""Validate 8/10/16/full sampling on the resolved DMercFail reference cut.

Run from the repository root with::

    python examples/validate_transition_sampling.py --output /tmp/sampling.json \
        --plot-dir /tmp/sampling-plots

The four backend/extractor combinations use the ADR 0005 reference at
radially global lambda_n=0.8. Every certified cut must have the same two-sheet
port incidence; every insufficient budget must retain unresolved ports and
insert no cut. Unique vertex traces and identical cut inputs are reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np

from alpha_analysis import BoozerField, DATA_DIR
from alpha_analysis.j_connectivity import (
    BackgroundMeshConfig,
    GmshBackgroundMeshBackend,
    GmshBackgroundMeshConfig,
    MarchingTetrahedraExtractor,
    PyVistaSurfaceExtractor,
    StructuredPrismMeshBackend,
    TransitionMappingConfig,
    TransitionStatus,
    cut_surface_at_transitions,
    extract_critical_curves,
    map_transitions_budget_sweep,
)
from alpha_analysis.j_connectivity.visualization import plot_transition_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path)
    args = parser.parse_args()
    path = Path(DATA_DIR) / (
        "boozmn_20260406-01-262-Ax_nfp4_Garabedian_mpol2_ntor2_minx0_allNfp_"
        "aspect10_DMercFail_m0p3_eval000323_low_resolution.nc"
    )
    field = BoozerField.from_boozmn(path)
    b = 5.040465893072380 + 0.8 * (12.050343540204448 - 5.040465893072380)
    budgets = (8, 10, 16, None)
    controls = TransitionMappingConfig(
        action_quadrature_order=32, max_action_quadrature_order=512
    )
    records = []
    for backend_name, backend in (
        ("structured", StructuredPrismMeshBackend(BackgroundMeshConfig(6, 24, 12))),
        ("gmsh", GmshBackgroundMeshBackend(GmshBackgroundMeshConfig(target_size=0.3))),
    ):
        background = backend.build(field)
        for extractor_name, extractor in (
            ("marching_tetrahedra", MarchingTetrahedraExtractor()),
            ("pyvista", PyVistaSurfaceExtractor()),
        ):
            started = time.perf_counter()
            extraction = extractor.extract(background, field, b)
            critical = extract_critical_curves(extraction, field, b)
            sweep = map_transitions_budget_sweep(field, critical, budgets, controls)
            cut_cache = {}
            for budget, transitions in zip(budgets, sweep):
                if len(transitions) != 1:
                    raise RuntimeError("the reference must have one transition")
                transition = transitions[0]
                key = (
                    transition.status,
                    transition.u.tobytes(),
                    tuple(port.points.tobytes() for port in transition.ports),
                    tuple(port.action_values.tobytes() for port in transition.ports),
                )
                if key not in cut_cache:
                    cut_cache[key] = cut_surface_at_transitions(
                        extraction.incoming,
                        np.full(len(extraction.incoming.points), np.nan),
                        transitions,
                        field=field,
                    )
                cut = cut_cache[key]
                sheet_count = len(np.unique(cut.sheet_ids))
                roles = {port.role: port.sheet_id for port in cut.ports}
                if transition.status is TransitionStatus.BUDGET_INSUFFICIENT:
                    if (
                        not len(cut.unresolved_transition_ids)
                        or len(cut.cut_edges)
                        or any(sheet_id != -1 for sheet_id in roles.values())
                    ):
                        raise RuntimeError("an uncertified transition was cut")
                elif transition.status is TransitionStatus.REGULAR:
                    if (
                        len(cut.unresolved_transition_ids)
                        or sheet_count != 2
                        or roles["parent"] == roles["child_1"]
                        or roles["child_1"] != roles["child_3"]
                    ):
                        raise RuntimeError(
                            "the certified reference sheet graph changed"
                        )
                    for port in cut.ports:
                        np.testing.assert_array_equal(
                            cut.action_values[port.polyline_vertex_ids],
                            port.action_values,
                        )
                else:
                    raise RuntimeError(
                        f"unexpected reference status: {transition.status}"
                    )
                record = {
                    "backend": backend_name,
                    "extractor": extractor_name,
                    "budget": budget,
                    "transition_status": transition.status.name,
                    "samples_used": transition.sampling_samples_used,
                    "authoritative_sample_count": transition.authoritative_sample_count,
                    "sampling_certified": transition.sampling_certified,
                    "sampling_reason": transition.sampling_reason,
                    "sheet_count": sheet_count,
                    "ports": roles,
                    "cut_edge_count": len(cut.cut_edges),
                    "unresolved_reasons": list(cut.unresolved_transition_reasons),
                }
                records.append(record)
                print(json.dumps(record), flush=True)
                if (
                    args.plot_dir is not None
                    and backend_name == "structured"
                    and extractor_name == "marching_tetrahedra"
                    and budget in (8, None)
                ):
                    args.plot_dir.mkdir(parents=True, exist_ok=True)
                    label = "budget8" if budget == 8 else "full"
                    commit = subprocess.check_output(
                        ["git", "rev-parse", "--short", "HEAD"], text=True
                    ).strip()
                    figure, _ = plot_transition_diagnostics(
                        field,
                        transition,
                        output_path=args.plot_dir / f"dmerc-{label}.png",
                        metadata={
                            "equilibrium": "DMercFail, lambda_n=0.8",
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[
                                :12
                            ],
                            "mesh": "structured (6,24,12), MT",
                            "controls": "geom=1e-8+0.2du, action=1e-8+0.02A, cap=128",
                            "commit": commit,
                        },
                    )
                    plt.close(figure)
            print(
                f"{backend_name}/{extractor_name}: "
                f"{time.perf_counter() - started:.2f}s",
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
