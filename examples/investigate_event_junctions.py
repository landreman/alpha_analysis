"""Reproduce milestone-10.2 event-junction diagnostics (accepted ADR 0006).

This diagnostic supplements the independent six-well topology test. Example::

    .venv/bin/python examples/investigate_event_junctions.py --output /tmp/event-junctions

Pass --w7x to exercise the reference curve with four contact occurrences. The
output directory receives JSON evidence, a pickle-free cut snapshot if cutting
completes, and a synthetic diagnostic PNG. Use --resolution for the ADR's
background sweep. A successful diagnostic is not a full convergence claim.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from alpha_analysis import BoozerField
from alpha_analysis.j_connectivity import (
    BackgroundMeshConfig,
    StructuredPrismMeshBackend,
    MarchingTetrahedraExtractor,
    extract_critical_curves,
    map_transitions,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField
from alpha_analysis.j_connectivity.transition_events import (
    ContactLocalizationConfig,
    localize_transition_contacts,
    build_transition_arcs,
)
from alpha_analysis.j_connectivity.mesh_cut import (
    cut_surface_at_transition_arcs,
    save_cut_surface,
)


def synthetic_field():
    return SyntheticFourierField(
        nfp=1,
        m=np.array([0, 0, 0, 1, 1]),
        n=np.array([0, 3, 1, 1, -1]),
        cosine_coefficients=np.array(
            [[2, 0.2], [1, 0], [0.2, 0], [0, 0.025], [0, -0.025]]
        ),
        sine_coefficients=np.zeros((5, 2)),
        iota_coefficients=np.array([0.0]),
        G_coefficients=np.array([3.0]),
        I_coefficients=np.array([0.0]),
    )


def plot_synthetic(source, arrangement, cut, path, resolution):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D

    component = 1
    original = source.points[source.triangles]
    doubled_area = np.cross(
        original[:, 1] - original[:, 0], original[:, 2] - original[:, 0]
    )[:, 2]
    selected = source.component_ids == component
    folded = selected & (doubled_area > 0)
    arc = next(arc for arc in arrangement.arcs if arc.curve.transition_id == 3)
    points = arc.curve.ports[0].points
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), constrained_layout=True)
    for axis, surface, title in (
        (
            axes[0],
            source,
            f"Source component: {np.count_nonzero(folded)} folded triangles",
        ),
        (axes[1], cut.surface, "Conforming cut through the folded chart"),
    ):
        selected_cells = surface.component_ids == component
        triangles = surface.triangles[selected_cells]
        colors = (
            plt.get_cmap("Pastel1")(cut.sheet_ids[selected_cells] % 9)
            if surface is cut.surface
            else "#edf0f2"
        )
        axis.add_collection(
            PolyCollection(
                surface.points[triangles, :2],
                facecolors=colors,
                edgecolors="#a6adb3",
                linewidths=0.35,
            )
        )
        axis.plot(points[:, 0], points[:, 1], color="#1769aa", linewidth=1.8)
        event_points = np.concatenate(
            [event.marginal_points for event in arrangement.events]
        )
        axis.scatter(
            event_points[:, 0],
            event_points[:, 1],
            marker="*",
            s=100,
            color="#e49b00",
            edgecolors="black",
            linewidths=0.4,
            zorder=4,
        )
        axis.set(
            xlim=(-1.05, 1.05),
            ylim=(-0.15, 1.05),
            xlabel="x [logical]",
            ylabel="y [logical]",
            title=title,
            aspect="equal",
        )
    axes[0].add_collection(
        PolyCollection(
            original[folded, :, :2],
            facecolors="#cf4b46",
            edgecolors="#902821",
            linewidths=0.5,
            zorder=3,
        )
    )
    axes[0].legend(
        handles=[
            Line2D(
                [],
                [],
                color="#cf4b46",
                linewidth=6,
                label="folded field-line projection",
            ),
            Line2D([], [], color="#1769aa", label="mapped companion T, arc 3"),
            Line2D(
                [],
                [],
                color="#e49b00",
                marker="*",
                linestyle="none",
                label="localized event",
            ),
        ],
        loc="lower center",
        fontsize=8,
    )
    fig.suptitle(
        "Bounded event-junction insertion — ADR 0006\n"
        f"Synthetic two-barrier field; structured {resolution}; "
        f"{len(np.unique(cut.sheet_ids))} sheets; {len(cut.events)} events; "
        f"{len(cut.unresolved_transition_ids)} unresolved arcs",
        fontsize=12,
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return np.flatnonzero(folded).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--w7x", action="store_true")
    parser.add_argument("--resolution", type=int, nargs=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    if args.w7x:
        equilibrium = (
            repository
            / "data/boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
        )
        field = BoozerField.from_boozmn(equilibrium)
        b = 2.7781394
        resolution = (6, 24, 12)
    else:
        equilibrium = "analytic field in ADR 0006"
        field = synthetic_field()
        b = float(field.B(0.5, 0.0, np.arccos(-np.sqrt(7 / 30))))
        resolution = (4, 16, 36)
    if args.resolution:
        resolution = tuple(args.resolution)
    report = {
        "implementation_status": "bounded insertion diagnostic; no full convergence claim",
        "equilibrium": str(equilibrium),
        "b": b,
        "resolution": resolution,
        "localization_controls": asdict(ContactLocalizationConfig()),
        "source_sha256": {
            name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
            for name in (
                "alpha_analysis/j_connectivity/transition_events.py",
                "alpha_analysis/j_connectivity/mesh_cut.py",
                "alpha_analysis/j_connectivity/critical_curves.py",
            )
        },
    }
    started = time.perf_counter()
    background = StructuredPrismMeshBackend(BackgroundMeshConfig(*resolution)).build(
        field
    )
    extraction = MarchingTetrahedraExtractor().extract(background, field, b)
    critical = extract_critical_curves(extraction, field, b)
    source = map_transitions(field, critical)
    report.update(
        {
            "surface_status": extraction.status.name,
            "critical_status": critical.status.name,
            "source_vertices": len(extraction.incoming.points),
            "source_triangles": len(extraction.incoming.triangles),
            "source_sample_counts": [len(curve.u) for curve in source],
            "source_contact_pairs": [
                curve.contact_sample_pairs.tolist() for curve in source
            ],
            "mapping_seconds": time.perf_counter() - started,
        }
    )
    print("Mapped source curves", flush=True)
    localized = localize_transition_contacts(field, critical, source)
    report["events"] = [
        {
            "event_id": event.event_id,
            "unresolved": event.unresolved,
            "marginal_points": event.marginal_points.tolist(),
            "occurrences": [
                {
                    "source_transition_id": occurrence.source_transition_id,
                    "source_pair": occurrence.source_sample_pair,
                    "u_interval": occurrence.u_interval.tolist(),
                    "localized": bool(occurrence.localized),
                    "reason": occurrence.reason,
                    "retained_traces": len(occurrence.samples),
                }
                for occurrence in event.occurrences
            ],
        }
        for event in localized.events
    ]
    arrangement = build_transition_arcs(field, critical, localized)
    report["arcs"] = [
        {
            "arc_id": arc.curve.transition_id,
            "source_id": arc.source_transition_id,
            "event_endpoints": arc.endpoint_event_ids,
            "status": arc.curve.status.name,
            "samples": len(arc.curve.u),
            "reason": arc.unresolved_reason,
        }
        for arc in arrangement.arcs
    ]
    print("Localized events and built transition arcs", flush=True)
    try:
        cut = cut_surface_at_transition_arcs(
            extraction.incoming,
            np.full(len(extraction.incoming.points), np.nan),
            arrangement,
            field=field,
        )
        save_cut_surface(args.output / "cut.npz", cut)
        report.update(
            {
                "cut_outcome": "completed",
                "sheet_count": len(np.unique(cut.sheet_ids)),
                "unresolved_arc_ids": cut.unresolved_transition_ids.tolist(),
                "unresolved_arc_reasons": cut.unresolved_transition_reasons,
                "port_sheet_ids": [
                    (port.transition_id, port.role, port.sheet_id) for port in cut.ports
                ],
                "corridor_count": cut.corridor_count,
                "max_corridor_faces_used": cut.max_corridor_faces_used,
                "unresolved_event_action_vertex_ids": cut.unresolved_event_action_vertex_ids.tolist(),
                "unresolved_action_flux": cut.unresolved_action_flux(field),
            }
        )
        if not args.w7x:
            report["expected_sheet_count"] = 6
            report["source_folded_triangle_ids"] = plot_synthetic(
                extraction.incoming,
                arrangement,
                cut,
                args.output / "diagnostic.png",
                resolution,
            )
    except Exception as error:
        report["cut_outcome"] = "exception"
        report["cut_error"] = f"{type(error).__name__}: {error}"
    report["total_seconds"] = time.perf_counter() - started
    (args.output / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in report
                if key
                in (
                    "cut_outcome",
                    "cut_error",
                    "sheet_count",
                    "unresolved_arc_ids",
                    "unresolved_arc_reasons",
                    "total_seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
