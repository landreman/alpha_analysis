"""Reproduce R1 five-field, six-pitch forward-scan timing evidence.

Run from the repository root with ``.venv/bin/python``. The pitch support comes
from the R0 *estimate* and is not promoted to a certified field bound.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alpha_analysis import BoozerField, DATA_DIR
from alpha_analysis.j_connectivity import (
    ForwardLineCatalogue,
    ForwardScanConfig,
    WellTraceConfig,
    adaptive_bounce_integral,
    batched_bounce_integrals,
    trace_regular_well,
)
from alpha_analysis.j_connectivity.visualization import plot_forward_catalogue

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "r0-population-ledger.json"
OUT = HERE / "r1-forward-scan-matrix.json"
REPORT = HERE / "r1-forward-scan-matrix.md"
PLOTS = HERE / "r1-forward-scan-plots"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> None:
    payload = json.loads(SOURCE.read_text())
    by_field = {}
    for case in payload["cases"]:
        by_field.setdefault(case["file"], []).append(case)
    PLOTS.mkdir(exist_ok=True)
    baseline = subprocess.check_output(
        ["git", "merge-base", "HEAD", "main"], text=True
    ).strip()
    evidence = {
        "baseline_commit": baseline,
        "source": "h(rho)=1 (UniformSourceProfile); scans and A/K are source independent",
        "pitch_provenance": "R0 sampled radially global extrema estimates; not field bounds",
        "software_scope": "Fourier-model root envelope; A/K n-versus-2n estimates; no field enclosure",
        "hardware": platform.platform(),
        "worker_count": 1,
        "cache_state": "new field objects, warm OS file cache uncontrolled",
        "code_sha256": sha256(
            ROOT / "alpha_analysis/j_connectivity/forward_catalogue.py"
        ),
        "cases": [],
        "files": [],
    }
    for file_index, (name, cases) in enumerate(by_field.items()):
        path = Path(DATA_DIR) / name
        start = time.perf_counter()
        field = BoozerField.from_boozmn(path)
        load_seconds = time.perf_counter() - start
        file_record = {
            "file": name,
            "sha256": sha256(path),
            "load_seconds": load_seconds,
            "line": {"s": 0.5, "theta0": 0.0, "zeta0": 0.0},
        }
        evidence["files"].append(file_record)
        for level, controls in (
            ("coarse", dict(samples_per_period=64, samples_per_wavelength=24)),
            ("fine", dict(samples_per_period=128, samples_per_wavelength=48)),
        ):
            cap_periods = 16
            config = ForwardScanConfig(max_periods=cap_periods, **controls)
            catalogue = ForwardLineCatalogue(field, 0.5, 0.0, 0.0, config)
            scan_start = time.perf_counter()
            catalogue.extend_to(4)
            scan_seconds = time.perf_counter() - scan_start
            initial_queries = {}
            initial_query_seconds = {}
            for case in cases:
                query_start = time.perf_counter()
                initial_queries[case["b"]] = catalogue.query(case["b"])
                initial_query_seconds[case["b"]] = time.perf_counter() - query_start
            for case in cases:
                query = initial_queries[case["b"]]
                query_seconds = initial_query_seconds[case["b"]]
                resume_scan_seconds = 0.0
                stages = [
                    {
                        "periods": 4,
                        "status": query.status.name,
                        "complete_wells": len(query.wells),
                        "open_left": query.open_left,
                        "open_right": query.open_right,
                        "unknown_cells": len(query.unknown_cells),
                    }
                ]
                if query.open_right and catalogue.scanned_periods > 4:
                    query_start = time.perf_counter()
                    query = catalogue.query(case["b"])
                    query_seconds += time.perf_counter() - query_start
                    stages.append(
                        {
                            "periods": catalogue.scanned_periods,
                            "status": query.status.name,
                            "complete_wells": len(query.wells),
                            "open_left": query.open_left,
                            "open_right": query.open_right,
                            "unknown_cells": len(query.unknown_cells),
                        }
                    )
                while query.open_right and catalogue.scanned_periods < cap_periods:
                    next_periods = min(cap_periods, 2 * catalogue.scanned_periods)
                    resume_start = time.perf_counter()
                    catalogue.extend_to(next_periods)
                    resume_scan_seconds += time.perf_counter() - resume_start
                    query_start = time.perf_counter()
                    query = catalogue.query(case["b"])
                    query_seconds += time.perf_counter() - query_start
                    stages.append(
                        {
                            "periods": next_periods,
                            "status": query.status.name,
                            "complete_wells": len(query.wells),
                            "open_left": query.open_left,
                            "open_right": query.open_right,
                            "unknown_cells": len(query.unknown_cells),
                        }
                    )
                batch_start = time.perf_counter()
                integrals = batched_bounce_integrals(catalogue, query.wells)
                batch_seconds = time.perf_counter() - batch_start
                fresh_start = time.perf_counter()
                fresh = ForwardLineCatalogue(field, 0.5, 0.0, 0.0, config)
                fresh.extend_to(query.scanned_periods)
                fresh_query = fresh.query(case["b"])
                fresh_scan_query_seconds = time.perf_counter() - fresh_start
                if len(fresh_query.wells) != len(query.wells):
                    raise AssertionError(
                        "fresh and shared scans disagree on well count"
                    )
                reference = None
                legacy = None
                if query.wells:
                    ref_start = time.perf_counter()
                    ref = adaptive_bounce_integral(catalogue, query.wells[0])
                    reference = {
                        "status": ref.status.name,
                        "A": ref.A if np.isfinite(ref.A) else None,
                        "K": ref.K if np.isfinite(ref.K) else None,
                        "seconds": time.perf_counter() - ref_start,
                    }
                    if level == "coarse":
                        old_start = time.perf_counter()
                        old = trace_regular_well(
                            field,
                            case["b"],
                            query.wells[0].q_in,
                            WellTraceConfig(
                                max_field_periods=cap_periods,
                                quadrature_rtol=1e-7,
                                quadrature_atol=1e-9,
                            ),
                        )
                        legacy = {
                            "status": old.status.name,
                            "seconds": time.perf_counter() - old_start,
                            "A": (
                                old.action_length
                                if np.isfinite(old.action_length)
                                else None
                            ),
                            "K": (
                                old.bounce_time_length
                                if np.isfinite(old.bounce_time_length)
                                else None
                            ),
                        }
                record = {
                    "file": name,
                    "file_index": file_index,
                    "lambda_n": case["lambda_n"],
                    "b": case["b"],
                    "level": level,
                    "controls": dict(
                        controls,
                        initial_periods=4,
                        max_periods=cap_periods,
                        max_cell_subdivisions=12,
                    ),
                    "sample_count": query.scanned_periods * catalogue.steps_per_period
                    + 1,
                    "shared_catalogue_sample_count": catalogue.sample_count,
                    "scan_seconds_shared": scan_seconds,
                    "resume_scan_seconds_shared": resume_scan_seconds,
                    "resume_stages": stages,
                    "final_scanned_periods": query.scanned_periods,
                    "query_seconds": query_seconds,
                    "batch_seconds": batch_seconds,
                    "fresh_scan_query_seconds": fresh_scan_query_seconds,
                    "root_complete_within_window": query.root_complete,
                    "status": query.status.name,
                    "open_left": query.open_left,
                    "open_right": query.open_right,
                    "passing_certified": query.passing_certified,
                    "unknown_cell_count": len(query.unknown_cells),
                    "extrema_unverified_cell_count": len(
                        catalogue.extrema_unverified_cells
                    ),
                    "complete_well_count": len(query.wells),
                    "wells": [
                        {
                            "u_in": well.u_in,
                            "u_out": well.u_out,
                            "A": item.A if np.isfinite(item.A) else None,
                            "K": item.K if np.isfinite(item.K) else None,
                            "A_error_estimate": (
                                item.error_A if np.isfinite(item.error_A) else None
                            ),
                            "K_error_estimate": (
                                item.error_K if np.isfinite(item.error_K) else None
                            ),
                            "integral_status": item.status.name,
                            "method": item.method,
                        }
                        for well, item in zip(query.wells, integrals)
                    ],
                    "adaptive_first_well": reference,
                    "legacy_first_well": legacy,
                    "reason": query.reason,
                }
                evidence["cases"].append(record)
                print(
                    file_index,
                    case["lambda_n"],
                    level,
                    query.status.name,
                    len(query.wells),
                    len(query.unknown_cells),
                    round(query_seconds + batch_seconds, 3),
                    flush=True,
                )
                if level == "fine" and case["lambda_n"] == 0.5:
                    figure, _ = plot_forward_catalogue(
                        catalogue,
                        query,
                        field_label=name,
                        integrals=integrals,
                        output_path=PLOTS / f"field-{file_index}.png",
                    )
                    plt.close(figure)
        OUT.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    if not any(
        row["file_index"] == 4 and row["complete_well_count"] > 0
        for row in evidence["cases"]
    ):
        raise AssertionError("n3are still has no complete well or A/K probe")
    REPORT.write_text(render_report(evidence))


def render_report(evidence) -> str:
    cases = evidence["cases"]
    status_counts = Counter((row["level"], row["status"]) for row in cases)
    shared_time = sum(
        row["scan_seconds_shared"] / 6
        + row["resume_scan_seconds_shared"]
        + row["query_seconds"]
        for row in cases
    )
    fresh_time = sum(row["fresh_scan_query_seconds"] for row in cases)
    resume_time = sum(row["resume_scan_seconds_shared"] for row in cases)
    integral_counts = Counter(
        (row["level"], well["integral_status"])
        for row in cases
        for well in row["wells"]
    )
    unresolved_cells = sum(row["unknown_cell_count"] for row in cases)
    by_key = {}
    for row in cases:
        by_key.setdefault((row["file_index"], row["lambda_n"]), {})[row["level"]] = row
    differences = []
    reference_differences = []
    legacy_differences = []
    for levels in by_key.values():
        coarse, fine = levels["coarse"], levels["fine"]
        if len(coarse["wells"]) != len(fine["wells"]):
            continue
        for first, second in zip(coarse["wells"], fine["wells"]):
            if first["A"] is not None and second["A"] is not None:
                differences.append(abs(first["A"] - second["A"]) / abs(second["A"]))
            if first["K"] is not None and second["K"] is not None:
                differences.append(abs(first["K"] - second["K"]) / abs(second["K"]))
    for row in cases:
        if not row["wells"] or row["adaptive_first_well"]["status"] != "REGULAR":
            continue
        first = row["wells"][0]
        reference = row["adaptive_first_well"]
        if first["integral_status"] == "REGULAR":
            reference_differences.extend(
                abs(first[name] - reference[name]) / abs(reference[name])
                for name in ("A", "K")
            )
            legacy = row["legacy_first_well"]
            if legacy is not None and legacy["status"] == "REGULAR":
                legacy_differences.extend(
                    abs(first[name] - legacy[name]) / abs(legacy[name])
                    for name in ("A", "K")
                )
    summary = []
    for level in ("coarse", "fine"):
        summary.append(
            f"{level}: {status_counts[(level, 'REGULAR')]} complete-window, "
            f"{status_counts[(level, 'MAX_PERIODS')]} censored-window, "
            f"{status_counts[(level, 'NO_WELL')]} analytic passing-line probes, "
            f"{integral_counts[(level, 'REGULAR')]} finite A/K results"
        )
    lines = [
        "# R1 shared forward-scan evidence",
        "",
        "This is a one-line-per-field R1 probe of all 30 physical file/pitch cases,",
        "at two scan resolutions. It does not establish global atlas coverage,",
        "population bounds, accessibility or final f acceptance.",
        "",
        f"Baseline: `{evidence['baseline_commit']}`. Code SHA256: `{evidence['code_sha256']}`.",
        f"Hardware: `{evidence['hardware']}`; one worker; new field objects, warm OS cache uncontrolled.",
        f"Source declaration: `{evidence['source']}`.",
        f"Pitch source: {evidence['pitch_provenance']}.",
        f"Numerical scope: {evidence['software_scope']}.",
        "The scan uses s=0.5, theta0=zeta0=0. Each catalogue begins with four",
        "field periods and resumes 4→8→16 while a queried pitch remains open at",
        "the right boundary. The 16-period cap is a work limit, not passing proof.",
        "One catalogue is shared over all six pitches per field and resolution.",
        "The table reports complete wells only; incomplete roots are not assigned",
        "zero action or classified passing.",
        *summary,
        f"Unresolved B=b root cells across all probes: {unresolved_cells}. "
        f"Up to {max(row['extrema_unverified_cell_count'] for row in cases)} "
        "cells per line still lack an extrema-completeness proof.",
        f"Measured shared scan+query: {shared_time:.3f} s across both grids and all "
        f"pitches; fresh catalogue scan+query per pitch: {fresh_time:.3f} s "
        f"({fresh_time/shared_time:.2f}x cost ratio).",
        f"The shared total includes {resume_time:.3f} s of incremental scan extensions; "
        "the JSON records each 4/8/16-period query snapshot and extension cost.",
        "Initial scan time is repeated in each row for context; count it once",
        "per field/resolution when summing shared wall time.",
        "This comparison uses the same field objects and one line per field;",
        "it excludes quadrature from both sides and is not a whole-equilibrium speed claim.",
        f"Maximum matched coarse/fine relative A or K difference: "
        f"{max(differences) if differences else float('nan'):.3e} "
        "(resolution diagnostic only).",
        f"Complete-well integrals with a non-REGULAR status: "
        f"{sum(row['complete_well_count'] for row in cases) - sum(integral_counts[(level, 'REGULAR')] for level in ('coarse', 'fine'))}. "
        f"Maximum first-well relative difference from independent adaptive "
        f"quadrature: {max(reference_differences) if reference_differences else float('nan'):.3e}; "
        f"from the legacy tracer on coarse probes: "
        f"{max(legacy_differences) if legacy_differences else float('nan'):.3e}.",
        f"The n3are field yielded "
        f"{sum(row['complete_well_count'] for row in cases if row['file_index'] == 4)} "
        "complete wells across both resolutions after resumption; higher-pitch "
        "windows that remain censored are retained as such.",
        "",
        "| field | λn | level | periods | complete wells | status | unknown cells | initial+resume scan s | query+batch s |",
        "| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for case in cases:
        lines.append(
            f"| {case['file_index']} | {case['lambda_n']} | {case['level']} | "
            f"4→{case['final_scanned_periods']} | {case['complete_well_count']} | "
            f"{case['status']} | {case['unknown_cell_count']} | "
            f"{case['scan_seconds_shared'] + case['resume_scan_seconds_shared']:.3f} | "
            f"{case['query_seconds'] + case['batch_seconds']:.3f} |"
        )
    lines += ["", "## Files and diagnostic plots", ""]
    for i, item in enumerate(evidence["files"]):
        lines.append(
            f"- {i}: `{item['file']}` SHA256 `{item['sha256']}`; load {item['load_seconds']:.3f} s"
        )
        lines.append(f"  ![field {i} lifted scan](r1-forward-scan-plots/field-{i}.png)")
    lines += [
        "",
        "The JSON companion retains all controls, root/window flags, unverified",
        "extrema cells, each completed well and its A/K estimate, adaptive reference",
        "and legacy first-well timing. The adaptive and GL error fields are numerical",
        "estimates, not rigorous quadrature or field enclosures.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    run()
