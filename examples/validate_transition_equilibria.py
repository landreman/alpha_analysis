"""Exercise milestone-9 transition mapping on the five reference equilibria.

This is the reproducible real-equilibrium validation requested by ``AGENTS.md``.
It runs both background-mesh backends and both fixed-``B`` surface extractors at
the five radially global normalized bounce levels, checkpointing a JSON record
after every case.  The transition summary retains topology, explicit failure
statuses, action-additivity residuals, and lifted field-line identity errors.

Run the complete matrix from the repository root with::

    python examples/validate_transition_equilibria.py \
        --output /tmp/milestone9-real-validation.json

Use ``--file-index`` and the backend/extractor selectors for targeted reruns.
Logical coordinates are dimensionless and angles are in radians.  Actions have
the length units of ``G`` and ``I`` (DESIGN.md §§3.2, 4.2, and 10.2).
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from alpha_analysis import BoozerField
from alpha_analysis.j_connectivity import (
    BackgroundMeshConfig,
    BoundsConfig,
    GmshBackgroundMeshBackend,
    GmshBackgroundMeshConfig,
    MarchingTetrahedraExtractor,
    PyVistaSurfaceExtractor,
    StructuredPrismMeshBackend,
    TransitionMappingConfig,
    extract_critical_curves,
    find_global_B_bounds,
    map_transitions,
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


def _finite_max(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if len(finite) else None


def _finite_min(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if len(finite) else None


def _transition_summary(field, transition) -> dict:
    ports = {port.role: port for port in transition.ports}
    parent = ports["parent"]
    tolerance = (
        transition.controls.additivity_atol
        + transition.controls.additivity_rtol * np.abs(parent.action_values)
    )
    ratio = np.abs(transition.additivity_residual) / tolerance
    quadrature_ratios = []
    for port in transition.ports:
        port_tolerance = (
            transition.controls.action_quadrature_atol
            + transition.controls.action_quadrature_rtol * np.abs(port.action_values)
        )
        quadrature_ratios.append(port.quadrature_error / port_tolerance)
    port_s_errors = []
    port_alpha_errors = []
    for port in transition.ports:
        finite = np.all(np.isfinite(port.points), axis=1) & np.isfinite(
            port.zeta_unwrapped
        )
        if not np.any(finite):
            continue
        points = port.points[finite]
        identity = transition.field_line_identity[finite]
        s = np.sum(points[:, :2] ** 2, axis=1)
        theta = np.arctan2(points[:, 1], points[:, 0])
        iota = np.asarray(field.iota(s), dtype=float)
        alpha = theta - iota * port.zeta_unwrapped[finite]
        alpha_error = alpha - identity[:, 1]
        alpha_error -= 2.0 * np.pi * np.round(alpha_error / (2.0 * np.pi))
        port_s_errors.append(np.max(np.abs(s - identity[:, 0])))
        port_alpha_errors.append(np.max(np.abs(alpha_error)))
    actions = {
        role: {
            "min": _finite_min(port.action_values),
            "max": _finite_max(port.action_values),
            "max_quadrature_error": _finite_max(port.quadrature_error),
        }
        for role, port in ports.items()
    }
    return {
        "transition_id": transition.transition_id,
        "status": transition.status.name,
        "source_critical_status": transition.source_critical_status.name,
        "sample_status_counts": dict(
            Counter(status.name for status in transition.sample_status)
        ),
        "sample_failure_reason_counts": dict(Counter(transition.sample_failure_reason)),
        "sample_count": len(transition.u),
        "total_u_length": transition.total_u_length,
        "s_range": [
            float(np.min(transition.field_line_identity[:, 0])),
            float(np.max(transition.field_line_identity[:, 0])),
        ],
        "actions": actions,
        "max_abs_additivity_residual": _finite_max(
            np.abs(transition.additivity_residual)
        ),
        "max_additivity_tolerance_ratio": _finite_max(ratio),
        "max_quadrature_tolerance_ratio": _finite_max(
            np.concatenate(quadrature_ratios)
        ),
        "max_port_s_identity_error": (
            float(max(port_s_errors)) if port_s_errors else None
        ),
        "max_port_alpha_identity_error": (
            float(max(port_alpha_errors)) if port_alpha_errors else None
        ),
    }


def _case_summary(field, extraction, critical, transitions, timings) -> dict:
    kind_counts = Counter(polyline.kind.name for polyline in critical.polylines)
    length_by_kind = {
        kind: float(
            sum(
                polyline.total_length
                for polyline in critical.polylines
                if polyline.kind.name == kind
            )
        )
        for kind in ("GAMMA_MIN", "GAMMA_MAX", "DEGENERATE")
    }
    transition_summaries = [
        _transition_summary(field, transition) for transition in transitions
    ]
    return {
        "outcome": "completed",
        "timings_seconds": timings,
        "surface": {
            "status": extraction.status.name,
            "unresolved_split_count": extraction.n_unresolved_splits,
            "point_count": len(extraction.full.points),
            "triangle_count": len(extraction.full.triangles),
            "component_count": len(np.unique(extraction.full.component_ids)),
            "g_zero_point_count": len(extraction.g_zero.points),
            "g_zero_segment_count": len(extraction.g_zero.segments),
        },
        "critical": {
            "status": critical.status.name,
            "point_count": len(critical.points),
            "segment_count": len(critical.segments),
            "polyline_counts": {
                kind: int(kind_counts.get(kind, 0))
                for kind in ("GAMMA_MIN", "GAMMA_MAX", "DEGENERATE")
            },
            "polyline_length_by_kind": length_by_kind,
            "report": asdict(critical.report),
        },
        "transition_status_counts": dict(
            Counter(summary["status"] for summary in transition_summaries)
        ),
        "transitions": transition_summaries,
    }


def _write_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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
    parser.add_argument("--max-field-periods", type=int, default=128)
    parser.add_argument("--max-curve-samples", type=int, default=8)
    parser.add_argument("--action-quadrature-order", type=int, default=32)
    parser.add_argument(
        "--structured-resolution",
        type=int,
        nargs=3,
        metavar=("N_RADIAL", "N_POLOIDAL", "N_ZETA"),
        default=(6, 24, 12),
    )
    parser.add_argument("--gmsh-target-size", type=float, default=0.3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    file_indices = args.file_index or list(range(len(FILES)))
    backends = args.backends or ["structured", "gmsh"]
    extractors = args.extractors or ["marching_tetrahedra", "pyvista"]
    lambda_values = args.lambda_n or list(LAMBDA_N)
    transition_config = TransitionMappingConfig(
        max_field_periods=args.max_field_periods,
        max_curve_samples=args.max_curve_samples,
        action_quadrature_order=args.action_quadrature_order,
        max_action_quadrature_order=max(512, 2 * args.action_quadrature_order),
    )
    structured_config = BackgroundMeshConfig(*args.structured_resolution)
    gmsh_config = GmshBackgroundMeshConfig(target_size=args.gmsh_target_size)
    controls = {
        "bounds": asdict(BoundsConfig(17, 32, 32)),
        "structured_background": asdict(structured_config),
        "gmsh_background": asdict(gmsh_config),
        "transition": asdict(transition_config),
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
        bounds_start = time.perf_counter()
        bounds = find_global_B_bounds(field, BoundsConfig(17, 32, 32))
        bounds_seconds = time.perf_counter() - bounds_start
        payload["files"][filename] = {
            "nfp": field.nfp,
            "refined_B_min": bounds.refined_min,
            "refined_B_max": bounds.refined_max,
            "bounds_seconds": bounds_seconds,
        }
        _write_checkpoint(args.output, payload)

        for backend_name in backends:
            build_start = time.perf_counter()
            if backend_name == "structured":
                backend = StructuredPrismMeshBackend(structured_config)
            else:
                backend = GmshBackgroundMeshBackend(gmsh_config)
            background = backend.build(field)
            build_seconds = time.perf_counter() - build_start
            print(
                f"  {backend_name}: {len(background.points)} points, "
                f"{len(background.tetrahedra)} tetrahedra ({build_seconds:.2f}s)",
                flush=True,
            )
            for lambda_n in lambda_values:
                b = bounds.refined_min + lambda_n * (
                    bounds.refined_max - bounds.refined_min
                )
                for extractor_name in extractors:
                    key = f"{file_index}:{lambda_n:g}:{backend_name}:{extractor_name}"
                    if args.resume and key in payload["cases"]:
                        print(f"  skip {key}", flush=True)
                        continue
                    print(f"  run {key}, b={b:.16g}", flush=True)
                    case_start = time.perf_counter()
                    try:
                        extractor = (
                            MarchingTetrahedraExtractor()
                            if extractor_name == "marching_tetrahedra"
                            else PyVistaSurfaceExtractor()
                        )
                        stage_start = time.perf_counter()
                        extraction = extractor.extract(background, field, b)
                        extraction_seconds = time.perf_counter() - stage_start
                        stage_start = time.perf_counter()
                        critical = extract_critical_curves(extraction, field, b)
                        critical_seconds = time.perf_counter() - stage_start
                        stage_start = time.perf_counter()
                        transitions = map_transitions(
                            field, critical, transition_config
                        )
                        transition_seconds = time.perf_counter() - stage_start
                        timings = {
                            "background_build_shared": build_seconds,
                            "surface_extraction": extraction_seconds,
                            "critical_curves": critical_seconds,
                            "transition_mapping": transition_seconds,
                            "case_total": time.perf_counter() - case_start,
                        }
                        result = _case_summary(
                            field, extraction, critical, transitions, timings
                        )
                        print(
                            "    "
                            f"surface={extraction.status.name}, "
                            f"critical={critical.status.name}, "
                            f"maxima={result['critical']['polyline_counts']['GAMMA_MAX']}, "
                            f"transitions={result['transition_status_counts']}, "
                            f"{timings['case_total']:.2f}s",
                            flush=True,
                        )
                    except Exception as error:  # validation must retain every failure
                        result = {
                            "outcome": "exception",
                            "exception_type": type(error).__name__,
                            "exception_message": str(error),
                            "case_total_seconds": time.perf_counter() - case_start,
                        }
                        print(
                            f"    EXCEPTION {type(error).__name__}: {error}", flush=True
                        )
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
                    payload["cases"][key] = result
                    _write_checkpoint(args.output, payload)


if __name__ == "__main__":
    main()
