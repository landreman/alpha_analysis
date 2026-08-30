"""Exercise Milestones 10--10.1 cuts on the five required real equilibria.

The matrix and mesh controls match the Milestone 9 validation driver.
Transition mapping uses a bounded adaptive certification budget: uncertified,
failed, or bracketed nongeneric curves are passed through the cut stage and
must remain explicit unresolved hyperedges.
Only cases containing a fully generic transition pay for surface-wide well
traces; those topology-sensitive cases retain the authoritative extracted
mesh. Every case performs a pickle-free topology serialization round trip.

Run from the repository root with::

    python examples/validate_cut_equilibria.py --output /tmp/milestone10-cuts.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
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
    GmshBackgroundMeshBackend,
    GmshBackgroundMeshConfig,
    MarchingTetrahedraExtractor,
    PyVistaSurfaceExtractor,
    StructuredPrismMeshBackend,
    TransitionMappingConfig,
    TransitionStatus,
    cut_surface_at_transitions,
    evaluate_surface_data,
    extract_critical_curves,
    find_global_B_bounds,
    load_cut_surface,
    map_transitions,
    save_cut_surface,
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
LAMBDA_N = (0.05, 0.1, 0.5, 0.9, 0.95)


def _write_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _needs_action_data(transition) -> bool:
    return (
        transition.status is TransitionStatus.REGULAR
        and transition.sampling_certified
        and len(transition.u) >= 2
        and not len(transition.contact_sample_pairs)
        and all(
            status is TransitionStatus.REGULAR for status in transition.sample_status
        )
        and all(
            np.all(np.isfinite(port.points)) and np.all(np.isfinite(port.action_values))
            for port in transition.ports
        )
    )


def _round_trip(cut) -> bool:
    with tempfile.TemporaryDirectory(prefix="milestone10-cut-") as directory:
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


def _cut_summary(cut) -> dict:
    valid_ports = [port for port in cut.ports if port.sheet_id >= 0]
    port_action_error = []
    invalid_incidence = 0
    for port in valid_ports:
        port_action_error.extend(
            np.abs(cut.action_values[port.polyline_vertex_ids] - port.action_values)
        )
        for vertex in port.polyline_vertex_ids:
            incident = np.flatnonzero(
                np.any(cut.surface.triangles == int(vertex), axis=1)
            )
            if not np.any(cut.sheet_ids[incident] == port.sheet_id):
                invalid_incidence += 1
    role_by_transition = {}
    for port in valid_ports:
        role_by_transition.setdefault(port.transition_id, {})[port.role] = port
    triangles_spanning_jump = 0
    for roles in role_by_transition.values():
        if "parent" not in roles or "child_1" not in roles:
            continue
        parent_ids = set(map(int, roles["parent"].polyline_vertex_ids))
        child_ids = set(map(int, roles["child_1"].polyline_vertex_ids))
        for triangle in cut.surface.triangles:
            vertices = set(map(int, triangle))
            triangles_spanning_jump += bool(
                vertices.intersection(parent_ids) and vertices.intersection(child_ids)
            )
    points = cut.surface.points
    return {
        "point_count": len(points),
        "triangle_count": len(cut.surface.triangles),
        "sheet_count": len(np.unique(cut.sheet_ids)),
        "cut_edge_count": len(cut.cut_edges),
        "valid_port_count": len(valid_ports),
        "unresolved_transition_ids": cut.unresolved_transition_ids.tolist(),
        "unresolved_transition_reasons": list(cut.unresolved_transition_reasons),
        "invalid_port_incidence_count": invalid_incidence,
        "triangles_spanning_parent_child_jump": triangles_spanning_jump,
        "max_port_action_error": (
            float(np.max(port_action_error)) if port_action_error else None
        ),
        "serialization_round_trip": _round_trip(cut),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--file-index", type=int, choices=range(len(FILES)), action="append"
    )
    parser.add_argument(
        "--backend",
        choices=("structured", "gmsh"),
        action="append",
        dest="backends",
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
    parser.add_argument(
        "--structured-resolution",
        type=int,
        nargs=3,
        metavar=("N_RADIAL", "N_POLOIDAL", "N_ZETA"),
        default=(6, 24, 12),
    )
    parser.add_argument("--gmsh-target-size", type=float, default=0.3)
    parser.add_argument(
        "--max-curve-samples",
        type=int,
        default=8,
        help="adaptive transition-mapping work budget per critical curve",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    file_indices = args.file_index or list(range(len(FILES)))
    backends = args.backends or ["structured", "gmsh"]
    extractors = args.extractors or ["marching_tetrahedra", "pyvista"]
    lambda_values = args.lambda_n or list(LAMBDA_N)
    structured_config = BackgroundMeshConfig(*args.structured_resolution)
    gmsh_config = GmshBackgroundMeshConfig(target_size=args.gmsh_target_size)
    transition_config = TransitionMappingConfig(
        max_curve_samples=args.max_curve_samples,
        action_quadrature_order=32,
        max_action_quadrature_order=512,
    )
    cut_config = ConstrainedCutConfig()
    controls = {
        "bounds": asdict(BoundsConfig(17, 32, 32)),
        "structured_background": asdict(structured_config),
        "gmsh_background": asdict(gmsh_config),
        "transition": asdict(transition_config),
        "cut": asdict(cut_config),
    }
    if args.resume and args.output.exists():
        payload = json.loads(args.output.read_text())
        if payload["controls"] != controls:
            raise ValueError("checkpoint controls disagree with this run")
    else:
        payload = {"controls": controls, "files": {}, "cases": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_dir = Path(__file__).resolve().parents[1] / "data"

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
            backend = (
                StructuredPrismMeshBackend(structured_config)
                if backend_name == "structured"
                else GmshBackgroundMeshBackend(gmsh_config)
            )
            background = backend.build(field)
            for lambda_n in lambda_values:
                b = bounds.refined_min + lambda_n * (
                    bounds.refined_max - bounds.refined_min
                )
                for extractor_name in extractors:
                    key = f"{file_index}:{lambda_n:g}:{backend_name}:{extractor_name}"
                    if args.resume and key in payload["cases"]:
                        continue
                    print(f"  run {key}", flush=True)
                    started = time.perf_counter()
                    try:
                        extractor = (
                            MarchingTetrahedraExtractor()
                            if extractor_name == "marching_tetrahedra"
                            else PyVistaSurfaceExtractor()
                        )
                        extraction = extractor.extract(background, field, b)
                        critical = extract_critical_curves(extraction, field, b)
                        transitions = map_transitions(
                            field, critical, transition_config
                        )
                        needs_actions = any(
                            _needs_action_data(transition) for transition in transitions
                        )
                        if needs_actions:
                            # A generic transition is the topology-sensitive
                            # case this milestone validates. Keep the
                            # authoritative extracted mesh here: aggressive
                            # optional coarsening can move T outside the local
                            # triangle neighborhood, while coarsening after a
                            # cut is forbidden by DESIGN.md §8.4.
                            surface = extraction.incoming
                            data = evaluate_surface_data(surface, field)
                            action = data.action_length
                            downsampling = None
                            regular_action_count = int(np.count_nonzero(data.regular))
                        else:
                            surface = extraction.incoming
                            action = np.full(len(surface.points), np.nan)
                            downsampling = None
                            regular_action_count = 0
                        cut = cut_surface_at_transitions(
                            surface,
                            action,
                            transitions,
                            field=field,
                            config=cut_config,
                        )
                        result = {
                            "outcome": "completed",
                            "surface_status": extraction.status.name,
                            "critical_status": critical.status.name,
                            "transition_status_counts": dict(
                                Counter(
                                    transition.status.name for transition in transitions
                                )
                            ),
                            "transition_count": len(transitions),
                            "transition_sampling": [
                                {
                                    "transition_id": transition.transition_id,
                                    "certified": transition.sampling_certified,
                                    "samples_used": transition.sampling_samples_used,
                                    "authoritative_sample_count": (
                                        transition.authoritative_sample_count
                                    ),
                                    "unresolved_intervals": (
                                        transition.sampling_unresolved_intervals.tolist()
                                    ),
                                    "reason": transition.sampling_reason,
                                    "max_geometry_error": (
                                        transition.sampling_max_geometry_error
                                    ),
                                    "max_action_error": (
                                        transition.sampling_max_action_error
                                    ),
                                }
                                for transition in transitions
                            ],
                            "needed_surface_actions": needs_actions,
                            "regular_action_count": regular_action_count,
                            "downsampling": downsampling,
                            "cut": _cut_summary(cut),
                        }
                    except Exception as error:
                        result = {
                            "outcome": "exception",
                            "exception_type": type(error).__name__,
                            "exception_message": str(error),
                        }
                    result.update(
                        {
                            "file": filename,
                            "file_index": file_index,
                            "lambda_n": lambda_n,
                            "b": b,
                            "backend": backend_name,
                            "extractor": extractor_name,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    )
                    print(
                        f"    {result['outcome']} ({result['elapsed_seconds']:.2f}s)",
                        flush=True,
                    )
                    payload["cases"][key] = result
                    _write_checkpoint(args.output, payload)


if __name__ == "__main__":
    main()
