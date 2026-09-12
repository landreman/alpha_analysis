"""Reproduce bounded R3 field-contour probes on the 30 prescribed cases.

Run from the repository root with ``.venv/bin/python``. Pitch values are R0
radially global *estimates*, not certified field bounds. Query completion is
feasibility evidence, not a sample of the loss fraction.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alpha_analysis import BoozerField, DATA_DIR
from alpha_analysis.j_connectivity.branch_atlas import transition_at
from alpha_analysis.j_connectivity.contour_trace import (
    ContourConfig,
    DirectContourOracle,
    plot_contour_result,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "r0-population-ledger.json"
OUT = HERE / "r3-contour-matrix.json"
REPORT = HERE / "r3-contour-matrix.md"
PLOTS = HERE / "r3-contour-plots"
DMERC = "boozmn_20260406-01-262-Ax_nfp4_Garabedian_mpol2_ntor2_minx0_allNfp_aspect10_DMercFail_m0p3_eval000323_low_resolution.nc"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(result):
    return {
        "status": result.status.name,
        "reason": result.reason,
        "path_steps": [len(path.points) for path in result.paths],
        "numerically_closed": [path.closed for path in result.paths],
        "edge_intersections": [path.edge_reached for path in result.paths],
        "max_estimated_error_A": max(
            (point.error_A for path in result.paths for point in path.points),
            default=None,
        ),
        "port_outcomes": [
            {"role": role, "status": status.name, "reason": reason}
            for role, status, reason in result.port_outcomes
        ],
        "scope": result.bound_scope,
    }


def synthetic_plot(kind: str):
    if kind == "closed":
        radial, angular, seed = (0.28, -0.8, 0.8), -0.08, (0.6, 0.0)
    else:
        radial, angular, seed = (0.0, 0.15), 0.1, (0.5, 0.0)
    coeff = np.zeros((3, 3))
    coeff[0, 0] = 2
    coeff[0, : len(radial)] += radial
    coeff[1, 0] = -1
    coeff[2, 0] = angular
    field = SyntheticFourierField(
        1,
        np.array([0, 0, 1]),
        np.array([0, 1, 0]),
        coeff,
        np.zeros_like(coeff),
        np.array([0.0]),
        np.array([3.0]),
        np.array([0.0]),
    )
    oracle = DirectContourOracle(field, 2.0)
    result = oracle.query(oracle.seed(*seed))
    fig = plot_contour_result(result, field_label=f"synthetic {kind}", b=2.0)
    fig.savefig(PLOTS / f"synthetic-{kind}.png", dpi=150)
    plt.close(fig)


def run() -> None:
    PLOTS.mkdir(exist_ok=True)
    for kind in ("closed", "edge"):
        synthetic_plot(kind)
    rows = json.loads(SOURCE.read_text())["cases"]
    by_file = {}
    for row in rows:
        by_file.setdefault(row["file"], []).append(row)
    evidence = {
        "milestone": "R3",
        "source": "h(rho)=1 (declared benchmark; contour geometry is source-independent)",
        "pitch_provenance": "R0 sampled radially global extrema estimates; not field enclosures",
        "scope": "represented Fourier/interpolated field, adaptive numerical A with estimated quadrature error, numerical gradient/path, finite-window root topology; no field or f enclosure",
        "hardware": platform.platform(),
        "workers": 1,
        "cache_state": "new field objects; warm OS cache uncontrolled",
        "code_sha256": sha256(ROOT / "alpha_analysis/j_connectivity/contour_trace.py"),
        "field_files": [],
        "seed_choices": [
            [0.3, 0.0],
            [0.5, 0.0],
            [0.8, 0.0],
            [0.5, float(np.pi / 2)],
            [0.8, float(np.pi / 2)],
        ],
        "resolution_controls": {
            "coarse": asdict(ContourConfig(step=0.04, max_steps=80)),
            "fine": asdict(ContourConfig(step=0.02, max_steps=160)),
        },
        "cases": [],
        "dmerc_queries": {},
    }
    started = time.perf_counter()
    for file_index, (name, cases) in enumerate(by_file.items()):
        load_start = time.perf_counter()
        path = Path(DATA_DIR) / name
        field = BoozerField.from_boozmn(path)
        evidence["field_files"].append(
            {
                "file": name,
                "sha256": sha256(path),
                "load_seconds": time.perf_counter() - load_start,
            }
        )
        for case in cases:
            b = case["b"]
            recorded = {
                "file": name,
                "file_index": file_index,
                "lambda_n": case["lambda_n"],
                "b": b,
            }
            choices = (
                (0.3, 0.0),
                (0.5, 0.0),
                (0.8, 0.0),
                (0.5, np.pi / 2),
                (0.8, np.pi / 2),
            )
            probes = []
            for label, config in (
                ("coarse", ContourConfig(step=0.04, max_steps=80)),
                ("fine", ContourConfig(step=0.02, max_steps=160)),
            ):
                oracle = DirectContourOracle(field, b, config)
                t0 = time.perf_counter()
                seed = None
                rejected = []
                for s, alpha in choices:
                    try:
                        seed = oracle.seed(s, alpha)
                        break
                    except (ValueError, RuntimeError) as exc:
                        rejected.append(str(exc))
                if seed is None:
                    probes.append(
                        {
                            "resolution": label,
                            "status": "NO_SEED",
                            "reasons": rejected,
                            "wall_seconds": time.perf_counter() - t0,
                        }
                    )
                    continue
                result = oracle.query(seed)
                item = summarize(result)
                item.update(
                    {
                        "resolution": label,
                        "seed": {
                            "s": seed.s,
                            "alpha": seed.alpha,
                            "zeta_in": seed.zeta_in,
                            "zeta_out": seed.zeta_out,
                            "A": seed.action_length,
                            "error_A": seed.error_A,
                        },
                        "wall_seconds": time.perf_counter() - t0,
                    }
                )
                probes.append(item)
            recorded["probes"] = probes
            evidence["cases"].append(recorded)
            print(
                file_index,
                case["lambda_n"],
                [
                    (p["resolution"], p["status"], round(p["wall_seconds"], 2))
                    for p in probes
                ],
                flush=True,
            )
            if name == DMERC and case["lambda_n"] == 0.8:
                oracle = DirectContourOracle(
                    field, b, ContourConfig(step=0.035, max_steps=96)
                )
                for label, s, alpha in (
                    ("closed_candidate", 0.3, 0.0),
                    ("edge_candidate", 0.95, -0.1577992744671513),
                ):
                    t0 = time.perf_counter()
                    result = oracle.query(oracle.seed(s, alpha))
                    evidence["dmerc_queries"][label] = {
                        **summarize(result),
                        "seed": [s, alpha],
                        "wall_seconds": time.perf_counter() - t0,
                    }
                    fig = plot_contour_result(
                        result, field_label=f"DMercFail {label}", b=b
                    )
                    fig.savefig(PLOTS / f"dmerc-{label}.png", dpi=150)
                    plt.close(fig)
                t0 = time.perf_counter()
                event = transition_at(field, b, 0.8, -0.1577992744671513, periods=2)
                result = oracle.query_event(event, "parent")
                evidence["dmerc_queries"]["transition"] = {
                    **summarize(result),
                    "event_status": event.status,
                    "event_parameter": event.parameter,
                    "event_actions": {
                        port.role: port.action_length for port in event.ports
                    },
                    "wall_seconds": time.perf_counter() - t0,
                }
                fig = plot_contour_result(
                    result, field_label="DMercFail transition", b=b
                )
                fig.savefig(PLOTS / "dmerc-transition.png", dpi=150)
                plt.close(fig)
    evidence["elapsed_seconds"] = time.perf_counter() - started
    OUT.write_text(json.dumps(evidence, indent=2) + "\n")
    counts = {
        label: Counter(
            p["status"]
            for c in evidence["cases"]
            for p in c["probes"]
            if p["resolution"] == label
        )
        for label in ("coarse", "fine")
    }
    completions = {}
    for label in ("coarse", "fine"):
        total = sum(counts[label].values())
        seeded = total - counts[label]["NO_SEED"]
        completed = counts[label]["ACCESSIBLE"] + counts[label]["INACCESSIBLE"]
        completions[label] = (completed, seeded)
    no_seed_both = sum(
        all(p["status"] == "NO_SEED" for p in case["probes"])
        for case in evidence["cases"]
    )
    agreeing = sum(
        case["probes"][0]["status"] == case["probes"][1]["status"]
        for case in evidence["cases"]
    )
    lines = [
        "# R3 direct-contour oracle evidence",
        "",
        "All 30 prescribed field/pitch cases were probed with two contour steps and finite work budgets. The source is declared h(ρ)=1; contours themselves are source-independent. Pitches come from R0 radially global **estimates**. Positive and negative statuses are represented-field numerical queries with finite-window root-pattern checks, not field enclosures or f estimates. Unknown and no-seed outcomes are not classified as unreachable. Exact field hashes, controls, reasons, path lengths and wall times are in [the JSON](r3-contour-matrix.json).",
        "",
        f"Hardware: `{evidence['hardware']}`; one worker; new field objects, warm OS cache uncontrolled. Total elapsed: {evidence['elapsed_seconds']:.2f} s.",
        "",
        f"Coarse (step .04, 80 steps): {dict(counts['coarse'])}. Fine (step .02, 160 steps): {dict(counts['fine'])}. The fixed interior seed search completed {completions['coarse'][0]}/{completions['coarse'][1]} seeded coarse queries and {completions['fine'][0]}/{completions['fine'][1]} seeded fine queries; {no_seed_both}/{len(evidence['cases'])} cases had no seed at either resolution. Statuses agreed across resolutions in {agreeing}/{len(evidence['cases'])} cases. Agreement on unknown is not convergence; completion is feasibility evidence, not an unbiased loss estimate.",
        "",
        "| Field | λn | Coarse | Fine | Coarse wall (s) | Fine wall (s) |",
        "| --- | ---: | --- | --- | ---: | ---: |",
    ]
    for case in evidence["cases"]:
        a, b = case["probes"]
        lines.append(
            f"| {case['file_index']} | {case['lambda_n']} | {a['status']} | {b['status']} | {a['wall_seconds']:.2f} | {b['wall_seconds']:.2f} |"
        )
    lines.extend(["", "DMercFail λn=0.8 probes:", ""])
    for label, item in evidence["dmerc_queries"].items():
        lines.append(
            f"- {label}: `{item['status']}`; {item['reason']}; {item['wall_seconds']:.2f} s. [Plot](r3-contour-plots/dmerc-{label}.png)."
        )
    lines.extend(
        [
            "",
            "Synthetic [closed](r3-contour-plots/synthetic-closed.png) and [edge-reaching](r3-contour-plots/synthetic-edge.png) diagnostic plots show the two classified reference contours. The targeted DMercFail edge witness is certified within the finite represented-field root window; the numerically closed real path remains unknown because its root-pattern certificate fails. Pointwise transition ports are expanded at the same parameter; unresolved incident continuation remains unknown. This partial oracle can challenge R4 where it classifies. The fixed interior matrix's completion rate is a feasibility risk, not an f result.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    run()
