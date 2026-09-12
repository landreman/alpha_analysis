"""Bounded R2 atlas census on the 30 physical reference cases.

The two transverse grids are feasibility probes, not field-level population or
accessibility enclosures. Unknown cells remain unknown even when sampled wells
are visible. The JSON records source, hashes, limits, times and every failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

from alpha_analysis import BoozerField
from alpha_analysis.j_connectivity.branch_atlas import (
    AtlasConfig,
    build_atlas,
    classify_cell,
    plot_atlas_diagnostics,
    refine_atlas_cell,
    transition_at,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)}")


def summarize(atlas):
    return {
        "sample_count": len(atlas.samples),
        "owned_sample_wells": len(atlas.owned_wells),
        "samples_with_censored_or_unknown_scan": sum(
            bool(sample.reason) for sample in atlas.samples
        ),
        "certified_cells": sum(c.multiplicity_upper is not None for c in atlas.cells),
        "unknown_cells": sum(c.multiplicity_upper is None for c in atlas.cells),
        "unknown_geometric_area": atlas.unknown_area,
        "known_multiplicity_area": atlas.known_owned_area,
        "unknown_reasons": sorted(
            {c.unknown_reason for c in atlas.cells if c.unknown_reason}
        ),
        "sampled_actions": sum(np.isfinite(w.action_length) for w in atlas.owned_wells),
        "action_scope": "numerical estimate on sampled complete wells, no K-weight bound",
    }


def dmerc_event(field, b):
    s = 0.8

    def equations(q):
        return [float(field.B(s, q[0], q[1])) - b, float(field.D_B(s, q[0], q[1]))]

    solution = root(equations, [-0.145, 0.0133], tol=1e-11)
    if np.max(np.abs(equations(solution.x))) > 1e-8:
        return None, {"status": "root_failure", "residual": equations(solution.x)}
    theta, zeta = solution.x
    alpha = theta - float(field.iota(s)) * zeta
    event = transition_at(field, b, s, alpha, periods=4)
    return event, {
        "status": event.status,
        "s": s,
        "theta": theta,
        "alpha": alpha,
        "zeta": zeta,
        "field_residual": equations(solution.x),
        "curvature": float(field.D2_B(s, theta, zeta)),
        "port_actions": {port.role: port.action_length for port in event.ports},
        "reason": event.reason,
        "scope": "represented-field event and numerical A estimates; no field enclosure",
    }


def synthetic_diagnostic(plot_dir: Path):
    multicover = SyntheticFourierField(
        1,
        np.zeros(2, dtype=int),
        np.array([0, 2]),
        np.array([[2.0], [-1.0]]),
        np.zeros((2, 1)),
        np.array([0.0]),
        np.array([3.0]),
        np.array([0.0]),
    )
    multicover_atlas = build_atlas(
        multicover, 2.0, AtlasConfig(3, 8, 2, compute_actions=True)
    )
    figure = plot_atlas_diagnostics(multicover_atlas)
    figure.savefig(plot_dir / "synthetic-multicover-atlas.png", dpi=150)
    plt.close(figure)
    field = SyntheticFourierField(
        1,
        np.zeros(3, dtype=int),
        np.array([0, 3, 1]),
        np.array([[2.0], [1.0], [0.2]]),
        np.zeros((3, 1)),
        np.array([0.0]),
        np.array([3.0]),
        np.array([0.0]),
    )
    zeta = np.arccos(-np.sqrt(7 / 30))
    b = float(field.B(0.5, 0, zeta))
    atlas = build_atlas(field, b, AtlasConfig(3, 8, 2, compute_actions=True))
    event = transition_at(field, b, 0.5, 0, periods=2)
    figure = plot_atlas_diagnostics(atlas, event)
    figure.savefig(plot_dir / "synthetic-six-port-atlas.png", dpi=150)
    plt.close(figure)
    return {
        "status": event.status,
        "port_count": len(event.ports),
        "plot": "synthetic-six-port-atlas.png",
        "multicover_plot": "synthetic-multicover-atlas.png",
    }


def run(repository: Path, output: Path, report: Path, plot_dir: Path):
    plot_dir.mkdir(parents=True, exist_ok=True)
    old = json.loads(
        (repository / "docs/validation/r0-population-ledger.json").read_text()
    )
    fields = old["fields"]
    levels = (0.05, 0.1, 0.5, 0.8, 0.9, 0.95)
    coarse = AtlasConfig(3, 4, 2, compute_actions=True)
    fine = AtlasConfig(5, 8, 2)
    cases = []
    start_all = time.perf_counter()
    dmerc_reference = None
    for index, record in enumerate(fields):
        path = repository / "data" / record["file"]
        start_load = time.perf_counter()
        field = BoozerField.from_boozmn(path)
        load_seconds = time.perf_counter() - start_load
        bmin, bmax = record["refined_B_min_estimate"], record["refined_B_max_estimate"]
        for level in levels:
            b = bmin + level * (bmax - bmin)
            coarse_atlas = None
            entry = {
                "file": record["file"],
                "file_index": index,
                "sha256": sha256(path),
                "lambda_n": level,
                "b": b,
                "load_seconds": load_seconds,
                "pitch_source": "R0 radially global extrema estimates, not field bounds",
            }
            for label, config in (("coarse", coarse), ("fine", fine)):
                started = time.perf_counter()
                try:
                    atlas = build_atlas(field, b, config)
                    if label == "coarse":
                        coarse_atlas = atlas
                    entry[label] = {
                        **summarize(atlas),
                        "elapsed_seconds": time.perf_counter() - started,
                        "config": vars(config),
                    }
                    if index == 2 and level == 0.8 and label == "fine":
                        event, dmerc_reference = dmerc_event(field, b)
                        figure = plot_atlas_diagnostics(atlas, event)
                        figure.savefig(plot_dir / "dmerc-0p8-atlas.png", dpi=150)
                        plt.close(figure)
                except Exception as error:
                    entry[label] = {
                        "status": "exception",
                        "reason": repr(error),
                        "elapsed_seconds": time.perf_counter() - started,
                        "config": vars(config),
                    }
            seed = (
                None
                if coarse_atlas is None
                else next(
                    (
                        sample
                        for sample in coarse_atlas.samples
                        if sample.s > 0 and sample.wells
                    ),
                    None,
                )
            )
            if seed is None:
                entry["owned_refinement"] = {
                    "status": "no_complete_seed_on_coarse_grid"
                }
            else:
                radial_index = min(int(seed.s * (coarse.n_s - 1)), coarse.n_s - 2)
                alpha_index = min(
                    int(seed.alpha / (2 * np.pi) * coarse.n_alpha), coarse.n_alpha - 1
                )
                cell_index = radial_index * coarse.n_alpha + alpha_index
                probes_owned = []
                for ds, da in ((1e-4, 1e-3), (1e-5, 1e-4)):
                    s0 = min(seed.s, 1 - ds)
                    a0 = seed.alpha
                    started = time.perf_counter()
                    try:
                        refined = refine_atlas_cell(
                            coarse_atlas, cell_index, (s0, s0 + ds), (a0, a0 + da)
                        )
                        target = next(
                            c
                            for c in refined.cells
                            if c.s_interval == (s0, s0 + ds)
                            and c.alpha_interval == (a0, a0 + da)
                        )
                        probes_owned.append(
                            {
                                "width_s": ds,
                                "width_alpha": da,
                                "target_count": target.multiplicity_upper,
                                "target_reason": target.unknown_reason,
                                "refined_unknown_area": refined.unknown_area,
                                "refined_known_multiplicity_area": refined.known_owned_area,
                                "area_partition_error": sum(
                                    c.area for c in refined.cells
                                )
                                - 2 * np.pi,
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        )
                    except Exception as error:
                        probes_owned.append(
                            {
                                "width_s": ds,
                                "width_alpha": da,
                                "status": "exception",
                                "reason": repr(error),
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        )
                entry["owned_refinement"] = {
                    "status": "probed",
                    "seed_s": seed.s,
                    "seed_alpha": seed.alpha,
                    "coarse_cell_index": cell_index,
                    "probes": probes_owned,
                }
            probes = []
            for width_s, width_alpha in ((0.01, 0.1), (0.001, 0.01)):
                started = time.perf_counter()
                try:
                    cell = classify_cell(
                        field,
                        b,
                        (0.4, 0.4 + width_s),
                        (0, width_alpha),
                        periods=2,
                        subdivisions=12,
                    )
                    probes.append(
                        {
                            "width_s": width_s,
                            "width_alpha": width_alpha,
                            "count": cell.multiplicity_upper,
                            "reason": cell.unknown_reason,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    )
                except Exception as error:
                    probes.append(
                        {
                            "width_s": width_s,
                            "width_alpha": width_alpha,
                            "status": "exception",
                            "reason": repr(error),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    )
            entry["local_refinement_probes"] = probes
            cases.append(entry)
            print(
                f"{index} {level}: coarse={entry['coarse'].get('certified_cells', 'E')} "
                f"fine={entry['fine'].get('certified_cells', 'E')} "
                f"local={[p.get('count') for p in probes]} "
                f"owned={entry['owned_refinement'].get('status')}",
                flush=True,
            )
    synthetic = synthetic_diagnostic(plot_dir)
    result = {
        "milestone": "R2",
        "source": "h(rho)=1 (declared benchmark; atlas geometry is source independent)",
        "scope": "finite Fourier-model atlas probes; no field, K-weight, reachability or f enclosure",
        "hardware": platform.platform(),
        "workers": 1,
        "cache_state": "new field objects; warm OS cache uncontrolled",
        "code_sha256": sha256(
            repository / "alpha_analysis/j_connectivity/branch_atlas.py"
        ),
        "code_hashes": {
            name: sha256(repository / name)
            for name in (
                "alpha_analysis/j_connectivity/branch_atlas.py",
                "alpha_analysis/j_connectivity/forward_catalogue.py",
                "alpha_analysis/j_connectivity/synthetic_fields.py",
                "alpha_analysis/boozer_field.py",
                "examples/validate_r2_atlas.py",
            )
        },
        "git_head": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=repository, text=True
        ).strip(),
        "pitch_bounds_source_sha256": sha256(
            repository / "docs/validation/r0-population-ledger.json"
        ),
        "cases": cases,
        "dmerc_reference": dmerc_reference,
        "synthetic_reference": synthetic,
        "elapsed_seconds": time.perf_counter() - start_all,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False, default=json_scalar) + "\n"
    )
    rows = []
    for case in cases:
        c, f = case["coarse"], case["fine"]
        owned = case["owned_refinement"]
        target_counts = (
            "/".join(str(p.get("target_count")) for p in owned["probes"])
            if owned["status"] == "probed"
            else "no seed"
        )
        rows.append(
            f"| {case['file_index']} | {case['lambda_n']} | {c.get('owned_sample_wells','E')} | "
            f"{c.get('certified_cells','E')}/{c.get('unknown_cells','E')} | "
            f"{f.get('certified_cells','E')}/{f.get('unknown_cells','E')} | "
            f"{case['local_refinement_probes'][0].get('count')} / "
            f"{case['local_refinement_probes'][1].get('count')} | {target_counts} | "
            f"{c['elapsed_seconds']+f['elapsed_seconds']:.2f} |"
        )
    positive_target_cases = sum(
        any(
            (p.get("target_count") or 0) > 0
            for p in case["owned_refinement"].get("probes", [])
        )
        for case in cases
    )
    report.write_text(
        "# R2 root-labelled atlas evidence\n\n"
        "All 30 physical file/pitch cases were probed at 3×4 and 5×8 transverse "
        "grids, two lifted periods, with separate narrow-cell refinements at "
        "s=0.4, α=0. The source declaration is h(ρ)=1. Pitches use the R0 "
        "radially global extrema **estimates**. These are represented-field "
        "geometry and sampled A estimates, not field, K-weight, reachability, "
        "slice or f enclosures. An unknown cell is not given zero weight. "
        "The exact hashes, controls, errors and wall times are in "
        "[the JSON](r2-atlas-matrix.json).\n\n"
        f"Hardware: `{result['hardware']}`; one worker; new field objects, "
        "warm OS cache uncontrolled. "
        f"Total elapsed: {result['elapsed_seconds']:.2f} s.\n\n"
        "| Field | λn | Coarse owned samples | Coarse known/unknown cells | "
        "Fine known/unknown cells | Fixed local counts (wide/narrow) | "
        "Owned target counts (wide/narrow) | "
        "Grid time s |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(rows)
        + "\n\n"
        f"Positive local owned multiplicity was certified in {positive_target_cases}/30 "
        "cases at one of the two narrow target widths. These patches replace "
        "their parent cell with a disjoint partition; the remaining unknown "
        "area and all possible links remain unknown. This does not establish "
        "a useful global population lower bound.\n\n"
        f"Synthetic equal-height event: {synthetic['port_count']} possible limiting "
        f"wells, status `{synthetic['status']}`. "
        f"[Event diagnostic](r2-atlas-plots/{synthetic['plot']}); "
        f"[certified multicover diagnostic](r2-atlas-plots/{synthetic['multicover_plot']}).\n\n"
        f"DMercFail λn=0.8: `{dmerc_reference['status'] if dmerc_reference else 'not found'}` "
        "at s=0.8 from independent B=b and D∥B=0 solve. "
        "[Diagnostic plot](r2-atlas-plots/dmerc-0p8-atlas.png). "
        "The old resolved cut is a geometry comparator, not the expected root or "
        "port action. Unknown coarse cells and narrow-cell failures require "
        "further atlas refinement and field-scope bounds in later milestones.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/validation/r2-atlas-matrix.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/validation/r2-atlas-matrix.md")
    )
    parser.add_argument(
        "--plot-dir", type=Path, default=Path("docs/validation/r2-atlas-plots")
    )
    args = parser.parse_args()
    run(args.repository.resolve(), args.output, args.report, args.plot_dir)
