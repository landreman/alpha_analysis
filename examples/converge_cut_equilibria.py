"""Milestone 10.3: converge the five-equilibrium cut matrix unattended.

Every case runs the failure-directed refinement coordinator
(:func:`alpha_analysis.j_connectivity.refinement.converge_case`) with one
matrix-wide set of escalation ladders — never per-case tuning.  The recorded
outcome of each case is ``resolved`` (every arc cut into sheets with explicit
event hyperedges), ``no_transitions``, or ``unresolved_explicit`` with
per-failure-class counts and terminal reasons.  Every retry the coordinator
made is stored (§21.3).

Run from the repository root with::

    python examples/converge_cut_equilibria.py --output /tmp/milestone103.json

The matrix is 5 files x 2 backends x 2 extractors x 6 radially global
``lambda_n`` values = 120 cases; use the filter options to shard it across
processes and ``--resume`` to continue a checkpoint.
"""

from __future__ import annotations

import argparse
import signal
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from alpha_analysis import BoozerField
from alpha_analysis.j_connectivity import (
    BackgroundMeshConfig,
    BoundsConfig,
    ConstrainedCutConfig,
    ContactLocalizationConfig,
    GmshBackgroundMeshBackend,
    GmshBackgroundMeshConfig,
    MarchingTetrahedraExtractor,
    PyVistaSurfaceExtractor,
    StructuredPrismMeshBackend,
    TransitionMappingConfig,
    find_global_B_bounds,
    load_cut_surface,
    save_cut_surface,
    surface_flux,
)
from alpha_analysis.j_connectivity.refinement import (
    RefinementBudgets,
    converge_case,
)

FILES = (
    "boozmn_20260402-01-038_Ax_PCA_20dofs_allNfp_aspect6_"
    "eval000290_low_resolution.nc",
    "boozmn_20260402-01-178_TURBO_Garabedian_mpol1_xmin0p1_"
    "allNfp_aspect6_eval000155.nc",
    "boozmn_20260406-01-262-Ax_nfp4_Garabedian_mpol2_ntor2_minx0_"
    "allNfp_aspect10_DMercFail_m0p3_eval000323_low_resolution.nc",
    "boozmn_d23p4_tm_ns51_mbooz16_nbooz16.nc",
    "boozmn_n3are_R7.75B5.7_mbooz18_nbooz12.nc",
)
LAMBDA_N = (0.05, 0.1, 0.5, 0.8, 0.9, 0.95)

# One matrix-wide background ladder per backend (level 0 is the milestone
# 9/10 base resolution); the coordinator may climb it only for the recorded
# background failure classes.
STRUCTURED_LEVELS = ((6, 24, 12), (9, 36, 18), (12, 48, 24))
GMSH_TARGET_SIZES = (0.3, 0.2, 0.15)
BUDGETS = RefinementBudgets(
    source_sample_budgets=(8, 16, 32, None),
    max_field_period_caps=(128, 256, 1024),
    localization_bisections=(20, 80, 320),
    background_levels=2,
    local_refinement_rounds=4,
    empty_interval_samples=2,
)


class CaseTimeout(RuntimeError):
    pass


def _write_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _round_trip(cut) -> bool:
    with tempfile.TemporaryDirectory(prefix="milestone103-cut-") as directory:
        path = Path(directory) / "cut.npz"
        save_cut_surface(path, cut)
        restored = load_cut_surface(path)
    arrays = (
        (cut.surface.points, restored.surface.points),
        (cut.surface.triangles, restored.surface.triangles),
        (cut.action_values, restored.action_values),
        (cut.sheet_ids, restored.sheet_ids),
        (cut.cut_edges, restored.cut_edges),
    )
    return all(
        np.array_equal(first, second, equal_nan=True) for first, second in arrays
    )


def _cut_summary(cut, field) -> dict:
    valid_ports = [port for port in cut.ports if port.sheet_id >= 0]
    invalid_incidence = 0
    port_action_error = []
    for port in valid_ports:
        errors = np.abs(
            cut.action_values[port.polyline_vertex_ids] - port.action_values
        )
        port_action_error.extend(errors[np.isfinite(errors)])
        for vertex in port.polyline_vertex_ids:
            incident = np.flatnonzero(
                np.any(cut.surface.triangles == int(vertex), axis=1)
            )
            if not np.any(cut.sheet_ids[incident] == port.sheet_id):
                invalid_incidence += 1
    return {
        "point_count": len(cut.surface.points),
        "triangle_count": len(cut.surface.triangles),
        "sheet_count": int(len(np.unique(cut.sheet_ids))),
        "cut_edge_count": int(len(cut.cut_edges)),
        "valid_port_count": len(valid_ports),
        "port_count": len(cut.ports),
        "event_count": len(cut.events),
        "event_kinds": [
            {"event_id": int(event.event_id), "unresolved": bool(event.unresolved)}
            for event in cut.events
        ],
        "unresolved_transition_ids": cut.unresolved_transition_ids.tolist(),
        "unresolved_transition_reasons": list(cut.unresolved_transition_reasons),
        "invalid_port_incidence_count": invalid_incidence,
        "max_port_action_error": (
            float(np.max(port_action_error)) if port_action_error else None
        ),
        "serialization_round_trip": _round_trip(cut),
        "corridor_count": cut.corridor_count,
        "max_corridor_faces_used": cut.max_corridor_faces_used,
        "unresolved_event_action_vertex_count": int(
            len(cut.unresolved_event_action_vertex_ids)
        ),
        "unresolved_action_triangle_count": int(
            np.count_nonzero(
                np.any(~np.isfinite(cut.action_values[cut.surface.triangles]), axis=1)
            )
        ),
        "cut_flux": surface_flux(cut.surface, field),
        "unresolved_action_flux": cut.unresolved_action_flux(field),
    }


def _sheet_signature(cut) -> list:
    """Label-free sheet/port/event incidence, for budget-invariance records."""
    order: dict[int, int] = {}
    for port in cut.ports:
        order.setdefault(int(port.sheet_id), len(order))
    ports = [
        [int(port.transition_id), port.role, order[int(port.sheet_id)]]
        for port in sorted(cut.ports, key=lambda p: (p.transition_id, p.role))
    ]
    events = [
        sorted(order[int(cut.ports[i].sheet_id)] for i in event.port_indices)
        for event in cut.events
    ]
    return [
        int(len(np.unique(cut.sheet_ids))),
        ports,
        events,
        sorted(map(int, cut.unresolved_transition_ids)),
    ]


def _case_record(resolution, field, elapsed) -> dict:
    record = {
        "outcome": "completed",
        "classification": resolution.classification,
        "resolved": bool(resolution.resolved),
        "background_level": resolution.background_level,
        "failure_class_counts": dict(resolution.failure_class_counts),
        "terminal_reasons": list(resolution.terminal_reasons),
        "attempts": [asdict(record) for record in resolution.attempts],
        "elapsed_seconds": elapsed,
    }
    if resolution.extraction is not None:
        record["surface_status"] = resolution.extraction.status.name
        record["source_flux"] = surface_flux(resolution.extraction.incoming, field)
    if resolution.critical is not None:
        record["critical_status"] = resolution.critical.status.name
    record["transition_count"] = len(resolution.transitions)
    record["transition_status_counts"] = dict(
        Counter(t.status.name for t in resolution.transitions)
    )
    if resolution.arrangement is not None:
        record["arcs"] = [
            {
                "arc_id": arc.curve.transition_id,
                "source_id": arc.source_transition_id,
                "endpoints": list(arc.endpoint_event_ids),
                "status": arc.curve.status.name,
                "certified": bool(arc.curve.sampling_certified),
                "reason": arc.unresolved_reason,
            }
            for arc in resolution.arrangement.arcs
        ]
        record["events"] = [
            {
                "event_id": event.event_id,
                "kind": event.kind,
                "marginal_point_count": int(len(event.marginal_points)),
                "unresolved": bool(event.unresolved),
                "occurrences": [
                    {
                        "source_id": occurrence.source_transition_id,
                        "pair": list(occurrence.source_sample_pair),
                        "localized": bool(occurrence.localized),
                        "reason": occurrence.reason,
                    }
                    for occurrence in event.occurrences
                ],
            }
            for event in resolution.arrangement.events
        ]
    if resolution.cut is not None:
        record["cut"] = _cut_summary(resolution.cut, field)
        record["sheet_signature"] = _sheet_signature(resolution.cut)
    return record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--file-index", type=int, choices=range(len(FILES)), action="append"
    )
    parser.add_argument(
        "--backend", choices=("structured", "gmsh"), action="append", dest="backends"
    )
    parser.add_argument(
        "--extractor",
        choices=("marching_tetrahedra", "pyvista"),
        action="append",
        dest="extractors",
    )
    parser.add_argument(
        "--lambda-n", type=float, choices=LAMBDA_N, action="append", dest="lambda_n"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-case-seconds",
        type=int,
        default=7200,
        help="hard per-case wall guard; a timed-out case is recorded, not lost",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    file_indices = args.file_index or list(range(len(FILES)))
    backends = args.backends or ["structured", "gmsh"]
    extractors = args.extractors or ["marching_tetrahedra", "pyvista"]
    lambda_values = args.lambda_n or list(LAMBDA_N)
    root = Path(__file__).resolve().parents[1]
    controls = {
        "bounds": asdict(BoundsConfig(17, 32, 32)),
        "structured_levels": [list(level) for level in STRUCTURED_LEVELS],
        "gmsh_target_sizes": list(GMSH_TARGET_SIZES),
        "budgets": asdict(BUDGETS),
        "transition": asdict(TransitionMappingConfig()),
        "cut": asdict(ConstrainedCutConfig()),
        "contact_localization": asdict(ContactLocalizationConfig()),
        "implementation_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "alpha_analysis/j_connectivity/refinement.py",
                "alpha_analysis/j_connectivity/mesh_cut.py",
                "alpha_analysis/j_connectivity/transition_events.py",
                "alpha_analysis/j_connectivity/transitions.py",
                "alpha_analysis/j_connectivity/surface_refine.py",
                "alpha_analysis/j_connectivity/critical_curves.py",
            )
        },
    }
    if args.resume and args.output.exists():
        payload = json.loads(args.output.read_text())
        if payload["controls"] != controls:
            raise ValueError("checkpoint controls disagree with this run")
    else:
        payload = {"controls": controls, "files": {}, "cases": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data"

    def timeout_handler(signum, frame):
        raise CaseTimeout("per-case wall guard expired")

    signal.signal(signal.SIGALRM, timeout_handler)

    for file_index in file_indices:
        filename = FILES[file_index]
        print(f"loading [{file_index}] {filename}", flush=True)
        field = BoozerField.from_boozmn(data_dir / filename)
        bounds = find_global_B_bounds(field, BoundsConfig(17, 32, 32))
        payload["files"][filename] = {
            "nfp": field.nfp,
            "refined_B_min": bounds.refined_min,
            "refined_B_max": bounds.refined_max,
        }
        _write_checkpoint(args.output, payload)
        for backend_name in backends:
            backgrounds: dict[int, object] = {}

            def background_factory(level: int):
                if level not in backgrounds:
                    if backend_name == "structured":
                        backend = StructuredPrismMeshBackend(
                            BackgroundMeshConfig(*STRUCTURED_LEVELS[level])
                        )
                    else:
                        backend = GmshBackgroundMeshBackend(
                            GmshBackgroundMeshConfig(
                                target_size=GMSH_TARGET_SIZES[level]
                            )
                        )
                    backgrounds[level] = backend.build(field)
                return backgrounds[level]

            for lambda_n in lambda_values:
                b = bounds.refined_min + lambda_n * (
                    bounds.refined_max - bounds.refined_min
                )
                for extractor_name in extractors:
                    key = f"{file_index}:{lambda_n:g}:{backend_name}:{extractor_name}"
                    if args.resume and key in payload["cases"]:
                        if payload["cases"][key].get("outcome") not in (
                            "timeout",
                            "wall_budget",
                        ):
                            continue
                    print(f"  run {key}", flush=True)
                    started = time.perf_counter()
                    signal.alarm(args.max_case_seconds)
                    try:
                        extractor = (
                            MarchingTetrahedraExtractor()
                            if extractor_name == "marching_tetrahedra"
                            else PyVistaSurfaceExtractor()
                        )
                        resolution = converge_case(
                            field,
                            b,
                            background_factory=background_factory,
                            extractor=extractor,
                            budgets=BUDGETS,
                            transition_config=TransitionMappingConfig(),
                        )
                        result = _case_record(
                            resolution, field, time.perf_counter() - started
                        )
                    except CaseTimeout:
                        # The §23.10.3 goal allows "physics or an explicit
                        # budget" as the terminal state; the per-case wall
                        # guard is such a budget, recorded per case.
                        result = {
                            "outcome": "wall_budget",
                            "classification": "unresolved_explicit",
                            "resolved": False,
                            "failure_class_counts": {"wall_budget": 1},
                            "terminal_reasons": [
                                "per-case wall-clock budget "
                                f"({args.max_case_seconds} s) exhausted before "
                                "the remediation ladders finished; the case "
                                "remains unresolved under an explicit budget"
                            ],
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    except Exception as error:
                        result = {
                            "outcome": "exception",
                            "exception_type": type(error).__name__,
                            "exception_message": str(error),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    finally:
                        signal.alarm(0)
                    result.update(
                        {
                            "file": filename,
                            "file_index": file_index,
                            "lambda_n": lambda_n,
                            "b": b,
                            "backend": backend_name,
                            "extractor": extractor_name,
                        }
                    )
                    print(
                        f"    {result['outcome']}: "
                        f"{result.get('classification', '-')} "
                        f"({result['elapsed_seconds']:.2f}s)",
                        flush=True,
                    )
                    payload["cases"][key] = result
                    _write_checkpoint(args.output, payload)


if __name__ == "__main__":
    main()
