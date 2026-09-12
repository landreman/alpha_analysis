"""Run the R0 population-only matrix without classifying accessibility.

This script deliberately records estimates, not field-level bounds.  It exercises
all five reference fields and six radially global pitch levels, compares two
spatial quadratures, writes machine-readable provenance, and saves the required
weight-versus-pitch/radius diagnostics.
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

from alpha_analysis import BoozerField
from alpha_analysis.j_connectivity.denominator import (
    BoundsConfig,
    DenominatorConfig,
    UniformSourceProfile,
    compute_denominator,
    find_global_B_bounds,
)
from alpha_analysis.j_connectivity.population import (
    PopulationQuadratureConfig,
    build_population_context,
    compute_pitch_band_estimate,
    compute_population_slice,
)
from alpha_analysis.j_connectivity.visualization import plot_population_diagnostics

FILES = (
    "boozmn_20260402-01-038_Ax_PCA_20dofs_allNfp_aspect6_eval000290_low_resolution.nc",
    "boozmn_20260402-01-178_TURBO_Garabedian_mpol1_xmin0p1_allNfp_aspect6_eval000155.nc",
    "boozmn_20260406-01-262-Ax_nfp4_Garabedian_mpol2_ntor2_minx0_allNfp_aspect10_DMercFail_m0p3_eval000323_low_resolution.nc",
    "boozmn_d23p4_tm_ns51_mbooz16_nbooz16.nc",
    "boozmn_n3are_R7.75B5.7_mbooz18_nbooz12.nc",
)
LAMBDA_N = (0.05, 0.1, 0.5, 0.8, 0.9, 0.95)
COARSE = PopulationQuadratureConfig(6, 24, 24)
FINE = PopulationQuadratureConfig(12, 48, 48)
EXTREMA = BoundsConfig(17, 32, 32, candidate_count=8, safety_factor=2.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git",) + arguments, cwd=repository, text=True
    ).strip()


def _config_dict(config) -> dict[str, int | float]:
    return {name: getattr(config, name) for name in config.__dataclass_fields__}


def run(repository: Path, output: Path, report: Path, plot_directory: Path) -> None:
    source = UniformSourceProfile()
    cases: list[dict] = []
    fields: list[dict] = []
    start_all = time.perf_counter()
    plot_directory.mkdir(parents=True, exist_ok=True)

    for file_index, filename in enumerate(FILES):
        path = repository / "data" / filename
        start = time.perf_counter()
        field = BoozerField.from_boozmn(path)
        bounds = find_global_B_bounds(field, EXTREMA)
        contexts = {}
        denominators = {}
        for label, config in (("coarse", COARSE), ("fine", FINE)):
            contexts[label] = build_population_context(
                field,
                source,
                config,
                source_name="h(rho)=1 (UniformSourceProfile)",
                surface_maximum_scope=(
                    f"sampled angular maximum on {config.n_theta}x{config.n_zeta} grid"
                ),
            )
            denominators[label] = compute_denominator(
                field,
                source,
                DenominatorConfig(config.n_s, config.n_theta, config.n_zeta),
            ).V_h

        fine_slices = []
        for lambda_n in LAMBDA_N:
            b = bounds.refined_min + lambda_n * (
                bounds.refined_max - bounds.refined_min
            )
            estimates = {
                label: compute_population_slice(context, b, dense_line_assumption=True)
                for label, context in contexts.items()
            }
            fine_slices.append(estimates["fine"])
            difference = abs(
                estimates["fine"].total_weight - estimates["coarse"].total_weight
            )
            cases.append(
                {
                    "file": filename,
                    "file_index": file_index,
                    "lambda_n": lambda_n,
                    "b": b,
                    "source": estimates["fine"].source_name,
                    "trapping_scope": estimates["fine"].trapping_scope,
                    "bound_scope": estimates["fine"].bound_scope,
                    "surface_maximum_scope": estimates["fine"].surface_maximum_scope,
                    "coarse_total_weight": estimates["coarse"].total_weight,
                    "fine_total_weight": estimates["fine"].total_weight,
                    "coarse_fine_absolute_change": difference,
                    "fine_uncontrolled_errors": list(
                        estimates["fine"].uncontrolled_errors
                    ),
                    "classification": "population_estimate_only",
                }
            )

        band_estimates = {
            label: compute_pitch_band_estimate(
                context,
                bounds.refined_min,
                bounds.refined_max,
                denominators[label],
                dense_line_assumption=True,
            )
            for label, context in contexts.items()
        }
        plot_path = plot_directory / f"field-{file_index}-population.png"
        figure, _ = plot_population_diagnostics(
            fine_slices,
            field_label=filename,
            source_label="h(rho)=1",
            output_path=plot_path,
        )
        plt.close(figure)
        fields.append(
            {
                "file": filename,
                "file_index": file_index,
                "sha256": _sha256(path),
                "nfp": field.nfp,
                "refined_B_min_estimate": bounds.refined_min,
                "refined_B_max_estimate": bounds.refined_max,
                "safe_B_lower_estimate": bounds.lower,
                "safe_B_upper_estimate": bounds.upper,
                "extrema_interpolation_error_estimate": bounds.interpolation_error,
                "denominator_coarse": denominators["coarse"],
                "denominator_fine": denominators["fine"],
                "full_band_fraction_coarse": band_estimates["coarse"].fraction,
                "full_band_fraction_fine": band_estimates["fine"].fraction,
                "full_band_fraction_absolute_change": abs(
                    band_estimates["fine"].fraction - band_estimates["coarse"].fraction
                ),
                "full_band_bound_scope": band_estimates["fine"].bound_scope,
                "elapsed_seconds": time.perf_counter() - start,
                "plot": str(plot_path.relative_to(repository)),
            }
        )

    population_module = (
        repository / "alpha_analysis" / "j_connectivity" / "population.py"
    )
    payload = {
        "milestone": "R0",
        "result_scope": "ledger-only quadrature estimates; no accessibility classification",
        "source": "h(rho)=1 (UniformSourceProfile)",
        "support_definition": (
            "b=B_min_global+lambda_n*(B_max_global-B_min_global), using the "
            "recorded locally optimized extrema estimates"
        ),
        "assumptions": [
            "dense-line surface-maximum equality is assumed, not certified on rational plateaus",
            "surface maxima at population nodes are sampled-grid estimates, not upper bounds",
            "coarse-fine differences are convergence diagnostics, not error enclosures",
            "field interpolation, extrema, population and denominator quadrature errors are uncontrolled",
        ],
        "controls": {
            "coarse_population": _config_dict(COARSE),
            "fine_population": _config_dict(FINE),
            "global_extrema": _config_dict(EXTREMA),
            "lambda_n": list(LAMBDA_N),
        },
        "provenance": {
            "git_revision": _git_value(repository, "rev-parse", "HEAD"),
            "working_tree_dirty": bool(_git_value(repository, "status", "--porcelain")),
            "population_module_sha256": _sha256(population_module),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "worker_count": 1,
            "cache_state": "cold field load per equilibrium; no persistent numerical cache",
        },
        "fields": fields,
        "cases": cases,
        "elapsed_seconds": time.perf_counter() - start_all,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# R0 independent population-ledger evidence",
        "",
        "This is a ledger-only run on the five reference equilibria and six required",
        "radially global pitch levels. It does **not** classify accessibility and every",
        "number below remains a quadrature **estimate**, not a field-level enclosure.",
        "",
        f"Source: `{payload['source']}`. Revision: `{payload['provenance']['git_revision']}`.",
        f"Total wall time: {payload['elapsed_seconds']:.3f} s on {payload['provenance']['platform']}",
        "with one worker and cold field loads.",
        "",
        "## Assumptions and uncontrolled scope",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["assumptions"])
    lines.extend(
        [
            "",
            "The pitch support uses the locally optimized global extrema estimates. The",
            "population-node surface maxima are sampled on each listed angular grid. Neither",
            "optimization nor coarse/fine agreement is promoted into a rigorous bound.",
            "",
            "## Whole-band estimates",
            "",
            "| field | B min | B max | trapped fraction (fine) | |fine-coarse| | wall s |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in fields:
        lines.append(
            "| {file} | {refined_B_min_estimate:.8g} | {refined_B_max_estimate:.8g} "
            "| {full_band_fraction_fine:.8g} | {full_band_fraction_absolute_change:.3e} "
            "| {elapsed_seconds:.3f} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## Slice estimates",
            "",
            "| field | lambda_n | b | Q total (fine) | |fine-coarse| |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in cases:
        lines.append(
            "| {file} | {lambda_n:.2g} | {b:.8g} | {fine_total_weight:.8g} "
            "| {coarse_fine_absolute_change:.3e} |".format(**item)
        )
    lines.extend(["", "## Diagnostics", ""])
    for item in fields:
        plot_link = Path(item["plot"]).relative_to("docs/validation")
        lines.append(f"![{item['file']} population]({plot_link})")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).parents[1])
    parser.add_argument(
        "--output", type=Path, default=Path("docs/validation/r0-population-ledger.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/validation/r0-population-ledger.md")
    )
    parser.add_argument(
        "--plot-directory",
        type=Path,
        default=Path("docs/validation/r0-population-plots"),
    )
    args = parser.parse_args()
    repository = args.repository.resolve()
    output = args.output if args.output.is_absolute() else repository / args.output
    report = args.report if args.report.is_absolute() else repository / args.report
    plot_directory = (
        args.plot_directory
        if args.plot_directory.is_absolute()
        else repository / args.plot_directory
    )
    run(repository, output, report, plot_directory)


if __name__ == "__main__":
    main()
