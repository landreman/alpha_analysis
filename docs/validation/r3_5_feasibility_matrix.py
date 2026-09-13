"""Reproduce the bounded R3.5 30-case root/contour/atlas feasibility matrix.

Run from the repository root with ``MPLCONFIGDIR=/private/tmp/mpl-r35
.venv/bin/python docs/validation/r3_5_feasibility_matrix.py``. Each physical
case has a cold 600-second guard. All b values are the saved R0 radially global
*estimates*; no result here is a field or f enclosure.
"""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import signal
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from alpha_analysis import BoozerField, DATA_DIR
from alpha_analysis.j_connectivity.atlas_corridor import (
    certify_contour_path_corridor,
    certify_local_atlas_corridor,
    embed_corridor_in_atlas,
    plot_atlas_corridor,
    plot_path_atlas_corridor,
)
from alpha_analysis.j_connectivity.branch_atlas import AtlasConfig, build_atlas
from alpha_analysis.j_connectivity.contour_trace import (
    ContourConfig,
    DirectContourOracle,
    plot_contour_result,
)
from alpha_analysis.j_connectivity.seed_search import (
    SeedSearchConfig,
    search_contour_seeds,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
R2 = HERE / "r2-atlas-matrix.json"
R3 = HERE / "r3-contour-matrix.json"
OUT = HERE / "r3-5-feasibility-matrix.json"
REPORT = HERE / "r3-5-feasibility-matrix.md"
PLOTS = HERE / "r3-5-feasibility-plots"
PROBE_KEYS = (
    (0, 0.1),
    (1, 0.5),
    (2, 0.05),
    (2, 0.1),
    (2, 0.5),
    (2, 0.8),
    (3, 0.05),
    (3, 0.1),
)
CORRIDOR_KEYS = ((0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5), (4, 0.8), (2, 0.8), (3, 0.1))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    """Render numerical unknowns as JSON null, never as a fabricated zero."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def summarize_contour(result):
    return {
        "status": result.status.name,
        "reason": result.reason,
        "path_steps": [len(path.points) for path in result.paths],
        "closed": [path.closed for path in result.paths],
        "edge_reached": [path.edge_reached for path in result.paths],
        "max_action_drift": max(
            (
                abs(point.action_length - path.points[0].action_length)
                for path in result.paths
                for point in path.points
            ),
            default=None,
        ),
        "max_estimated_error_A": max(
            (point.error_A for path in result.paths for point in path.points),
            default=None,
        ),
        "event_discovery": (
            None
            if result.event_discovery is None
            else {
                **asdict(result.event_discovery),
                "event": (
                    None
                    if result.event_discovery.event is None
                    else asdict(result.event_discovery.event)
                ),
            }
        ),
        "port_outcomes": [
            {"role": role, "status": status.name, "reason": reason}
            for role, status, reason in result.port_outcomes
        ],
        "scope": result.bound_scope,
    }


def summarize_atlas(atlas):
    return {
        "sample_count": len(atlas.samples),
        "certified_positive_cells": sum(
            c.multiplicity_upper is not None and c.multiplicity_upper > 0
            for c in atlas.cells
        ),
        "certified_empty_cells": sum(c.multiplicity_upper == 0 for c in atlas.cells),
        "unknown_cells": sum(c.multiplicity_upper is None for c in atlas.cells),
        "certified_positive_owned_area": atlas.known_owned_area,
        "certified_positive_geometric_area": sum(
            c.area
            for c in atlas.cells
            if c.multiplicity_upper is not None and c.multiplicity_upper > 0
        ),
        "certified_empty_area": sum(
            c.area for c in atlas.cells if c.multiplicity_upper == 0
        ),
        "unknown_area": atlas.unknown_area,
        "fixed_domain_area": sum(c.area for c in atlas.cells),
        "unknown_reasons": dict(
            Counter(c.unknown_reason for c in atlas.cells if c.unknown_reason)
        ),
        "config": asdict(atlas.config),
        "scope": "represented-field finite-window root census; geometric ds d-alpha, no K weight",
    }


class CaseGuard:
    """Cold physical-case wall guard and active 60/300/600 s snapshots."""

    def __init__(self):
        self.started = time.perf_counter()
        self.stage = "cold load"
        self.snapshots = []
        self.thresholds = iter((60, 300, 600))
        self.next_threshold = next(self.thresholds)

    def __enter__(self):
        self.old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._alarm)
        signal.setitimer(signal.ITIMER_REAL, 60)
        return self

    def _alarm(self, signum, frame):
        elapsed = time.perf_counter() - self.started
        self.snapshots.append(
            {
                "at_seconds": self.next_threshold,
                "elapsed_seconds": elapsed,
                "active_stage": self.stage,
            }
        )
        if self.next_threshold == 600:
            raise TimeoutError("600-second cold physical-case guard reached")
        prior = self.next_threshold
        self.next_threshold = next(self.thresholds)
        signal.setitimer(signal.ITIMER_REAL, self.next_threshold - prior)

    def __exit__(self, exc_type, exc, traceback):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.old_handler)


def run_case(old_r2, old_r3, evidence):
    key = (old_r3["file_index"], old_r3["lambda_n"])
    cpu_start = time.process_time()
    guard = CaseGuard()
    record = {
        "file": old_r3["file"],
        "file_index": key[0],
        "lambda_n": key[1],
        "b": old_r3["b"],
        "original_r3_statuses": {
            probe["resolution"]: probe["status"] for probe in old_r3["probes"]
        },
        "original_r3_seeds": {
            probe["resolution"]: probe.get("seed") for probe in old_r3["probes"]
        },
        "original_r2_atlas": {level: old_r2[level] for level in ("coarse", "fine")},
    }
    try:
        with guard:
            record["field_sha256"] = sha256(Path(DATA_DIR) / record["file"])
            field = BoozerField.from_boozmn(Path(DATA_DIR) / record["file"])
            guard.stage = "fixed original contour cohort"
            record["current_original"] = {}
            record["full_path_corridors"] = {}
            for level, step, max_steps in (("coarse", 0.04, 80), ("fine", 0.02, 160)):
                original = next(p for p in old_r3["probes"] if p["resolution"] == level)
                if original.get("seed") is None:
                    record["current_original"][level] = {
                        "status": "NO_SEED",
                        "reason": "fixed R3 four-period five-location cohort found none",
                    }
                    continue
                oracle = DirectContourOracle(
                    field,
                    record["b"],
                    ContourConfig(
                        step=step,
                        max_steps=max_steps,
                        max_certificate_boxes=4096,
                        max_certificate_depth=10,
                    ),
                )
                start = time.perf_counter()
                seed = original["seed"]
                result = oracle.query(
                    oracle.seed(seed["s"], seed["alpha"], seed["zeta_in"])
                )
                record["current_original"][level] = {
                    **summarize_contour(result),
                    "wall_seconds": time.perf_counter() - start,
                }
                if key in ((2, 0.8), (3, 0.1)):
                    guard.stage = f"{level} complete real atlas path corridor"
                    if key == (3, 0.1):
                        path = oracle._trace_one(
                            oracle.seed(seed["s"], seed["alpha"], seed["zeta_in"]), -1
                        )
                    else:
                        path = result.paths[0]
                    if (key == (2, 0.8) and not path.closed) or (
                        key == (3, 0.1) and not path.edge_reached
                    ):
                        raise RuntimeError(
                            "required classified path has no physical terminal"
                        )
                    if not oracle._root_pattern_certified(path):
                        raise RuntimeError(
                            "required classified path lacks an independent root certificate"
                        )
                    corridor_start = time.perf_counter()
                    full = certify_contour_path_corridor(
                        oracle, path, max_depth=12, max_boxes=5000
                    )
                    record["full_path_corridors"][level] = {
                        "status": full.status,
                        "reason": full.reason,
                        "path_terminal": "closed" if path.closed else "edge",
                        "branch_id": full.branch_id,
                        "attempted_boxes": full.attempted_boxes,
                        "max_depth_used": full.max_depth_used,
                        "tile_count": len(full.tiles),
                        "owned_cell_count": len(full.owned_cells),
                        "owned_area": full.owned_area,
                        "tiles": [asdict(tile) for tile in full.tiles],
                        "disjoint_owned_cells": [
                            asdict(cell) for cell in full.owned_cells
                        ],
                        "wall_seconds": time.perf_counter() - corridor_start,
                        "scope": "full numerical contour path, represented-field owned-root cells; ds d-alpha area, not K weight or f",
                    }
                    if full.status == "CERTIFIED":
                        fig = plot_path_atlas_corridor(full)
                        fig.savefig(
                            PLOTS / f"full-path-corridor-{key[0]}-{key[1]}-{level}.png",
                            dpi=150,
                        )
                        import matplotlib.pyplot as plt

                        plt.close(fig)
                    guard.stage = "fixed original contour cohort"
                if key == (2, 0.8) and level == "fine":
                    fig = plot_contour_result(
                        result, field_label="DMercFail 0.8 interior", b=record["b"]
                    )
                    fig.savefig(PLOTS / "dmerc-interior-contour.png", dpi=150)
                    import matplotlib.pyplot as plt

                    plt.close(fig)

            guard.stage = "resumable expanded seed search"
            oracle = DirectContourOracle(field, record["b"])
            search_start = time.perf_counter()
            search = search_contour_seeds(oracle, SeedSearchConfig())
            record["seed_search"] = {
                "seeds": [asdict(seed) for seed in search.seeds],
                "attempts": [asdict(attempt) for attempt in search.attempts],
                "exhausted": search.exhausted,
                "population_empty_proved": search.population_empty_proved,
                "original_attempts": sum(
                    a.cohort == "original" for a in search.attempts
                ),
                "expanded_attempts": sum(
                    a.cohort != "original" for a in search.attempts
                ),
                "wall_seconds": time.perf_counter() - search_start,
            }
            if search.seeds and all(p.get("seed") is None for p in old_r3["probes"]):
                guard.stage = "supplemental contour query"
                record["supplemental_query"] = {}
                for level, step, max_steps in (
                    ("coarse", 0.04, 80),
                    ("fine", 0.02, 160),
                ):
                    query_oracle = DirectContourOracle(
                        field,
                        record["b"],
                        ContourConfig(
                            step=step,
                            max_steps=max_steps,
                            max_certificate_boxes=4096,
                            max_certificate_depth=10,
                        ),
                    )
                    start = time.perf_counter()
                    result = query_oracle.query(search.seeds[0])
                    record["supplemental_query"][level] = {
                        **summarize_contour(result),
                        "wall_seconds": time.perf_counter() - start,
                    }

            guard.stage = "fixed-domain coarse atlas"
            coarse = build_atlas(
                field, record["b"], AtlasConfig(**old_r2["coarse"]["config"])
            )
            record["current_atlas"] = {"coarse": summarize_atlas(coarse)}
            guard.stage = "fixed-domain fine atlas"
            fine = build_atlas(
                field, record["b"], AtlasConfig(**old_r2["fine"]["config"])
            )
            record["current_atlas"]["fine"] = summarize_atlas(fine)
            for level, atlas in (("coarse", coarse), ("fine", fine)):
                full = record["full_path_corridors"].get(level)
                if full is None or full["status"] != "CERTIFIED":
                    continue
                new_positive_area = 0.0
                for owned in full["disjoint_owned_cells"]:
                    for parent in atlas.cells:
                        if parent.multiplicity_upper is not None:
                            continue
                        ds = max(
                            0.0,
                            min(owned["s_interval"][1], parent.s_interval[1])
                            - max(owned["s_interval"][0], parent.s_interval[0]),
                        )
                        da = max(
                            0.0,
                            min(owned["alpha_interval"][1], parent.alpha_interval[1])
                            - max(owned["alpha_interval"][0], parent.alpha_interval[0]),
                        )
                        new_positive_area += ds * da
                full["additional_certified_positive_area_on_fixed_domain"] = (
                    new_positive_area
                )
                full["unknown_geometric_area_after_corridor"] = (
                    atlas.unknown_area - new_positive_area
                )

            guard.stage = "local atlas corridor"
            record["local_corridor_attempt_limit"] = 16
            record["local_corridor_attempts"] = []
            record["local_corridor"] = {
                "status": "NO_SEED",
                "reason": "no located complete ordinary well",
            }
            for seed_index, seed in enumerate(search.seeds[:16]):
                for step in (0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 1e-5):
                    try:
                        corridor = certify_local_atlas_corridor(oracle, seed, step=step)
                    except (ValueError, RuntimeError, ArithmeticError) as error:
                        record["local_corridor_attempts"].append(
                            {
                                "seed_index": seed_index,
                                "step": step,
                                "status": "UNKNOWN",
                                "reason": str(error),
                            }
                        )
                        record["local_corridor"] = {
                            "status": "UNKNOWN",
                            "reason": str(error),
                        }
                        continue
                    record["local_corridor_attempts"].append(
                        {
                            "seed_index": seed_index,
                            "step": step,
                            "status": "CERTIFIED" if corridor.certified else "UNKNOWN",
                            "reason": corridor.reason,
                        }
                    )
                    if corridor.certified:
                        record["local_corridor"] = {
                            "status": "CERTIFIED",
                            "seed": asdict(seed),
                            "step": step,
                            "branch_id": corridor.branch_id,
                            "cells": [asdict(c) for c in corridor.cells],
                            "union": asdict(corridor.union),
                            "scope": "local represented-field geometric corridor; not full-path or weighted coverage",
                        }
                        if key in CORRIDOR_KEYS:
                            fig = plot_atlas_corridor(corridor)
                            fig.savefig(
                                PLOTS / f"corridor-{key[0]}-{key[1]}.png", dpi=150
                            )
                            import matplotlib.pyplot as plt

                            plt.close(fig)
                        embedded = embed_corridor_in_atlas(coarse, corridor)
                        record["fixed_domain_after_local_corridor"] = summarize_atlas(
                            embedded
                        )
                        break
                if record["local_corridor"]["status"] == "CERTIFIED":
                    break
            if key == (2, 0.8):
                guard.stage = "automatic real generic event"
                event_oracle = DirectContourOracle(
                    field,
                    record["b"],
                    ContourConfig(
                        step=0.02,
                        max_steps=160,
                        max_certificate_boxes=4096,
                        max_certificate_depth=10,
                    ),
                )
                event_seed = event_oracle.seed(
                    0.7980685243065248,
                    -0.16013007426726095,
                    zeta_in_hint=-1.3559938733080594,
                )
                event_start = time.perf_counter()
                event_result = event_oracle.query(event_seed)
                record["discovered_event"] = {
                    "seed": asdict(event_seed),
                    **summarize_contour(event_result),
                    "wall_seconds": time.perf_counter() - event_start,
                }
                fig = plot_contour_result(
                    event_result,
                    field_label="DMercFail discovered split",
                    b=record["b"],
                )
                fig.savefig(PLOTS / "dmerc-discovered-event.png", dpi=150)
                import matplotlib.pyplot as plt

                plt.close(fig)
    except (TimeoutError, ValueError, RuntimeError, ArithmeticError) as error:
        record["case_failure"] = f"{type(error).__name__}: {error}"
    record["wall_seconds"] = time.perf_counter() - guard.started
    record["cpu_seconds"] = time.process_time() - cpu_start
    record["snapshots"] = guard.snapshots
    record["peak_rss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record["under_600_second_guard"] = (
        record["wall_seconds"] <= 600 and "case_failure" not in record
    )
    return record


def write_report(evidence):
    cases = evidence["cases"]
    lines = [
        "# R3.5 feasibility matrix (represented-field local evidence)",
        "",
        f"Revision: `{evidence['git_head']}`. Hardware: `{evidence['hardware']}`; one worker; h(ρ)=1 declared. ",
        "Pitches retain R0 sampled radially global extrema estimates. Root/field interpolation and action quadrature errors are not field enclosures.",
        "",
        f"Cases recorded: {len(cases)}/30. Total wall: {evidence['elapsed_seconds']:.2f} s.",
        "The original R2/R3 evidence remains in its separate files. Current results use 4096 certificate boxes, depth 10, and both original contour steps.",
        "",
        "| Field | λn | Original coarse/fine | Current coarse/fine | Expanded seeds | Fine atlas positive/unknown area | Local corridor | Full path coarse/fine | Cold wall (s) |",
        "| --- | ---: | --- | --- | ---: | --- | --- | --- | ---: |",
    ]
    for case in cases:
        old = case["original_r3_statuses"]
        current = case.get("current_original", {})
        new = [
            current.get(level, {}).get("status", "FAIL") for level in ("coarse", "fine")
        ]
        atlas = case.get("current_atlas", {}).get("fine", {})
        seeds = case.get("seed_search", {}).get("seeds", [])
        full = case.get("full_path_corridors", {})
        full_status = "/".join(
            full.get(level, {}).get("status", "—") for level in ("coarse", "fine")
        )
        lines.append(
            f"| {case['file_index']} | {case['lambda_n']} | {old.get('coarse')}/{old.get('fine')} | "
            f"{new[0]}/{new[1]} | {len(seeds)} | "
            f"{atlas.get('certified_positive_owned_area', 0):.3g}/{atlas.get('unknown_area', 0):.3g} | "
            f"{case.get('local_corridor', {}).get('status', 'FAIL')} | {full_status} | {case['wall_seconds']:.2f} |"
        )
    lines += [
        "",
        "The two required classified paths have separate full-corridor records at both contour resolutions. Disjoint owned tile areas are geometric ds d-alpha; remaining unknown area and all K/field errors retain their scope.",
        "No wide or unknown outcome is treated as an f success or an empty trapped population.",
        "",
        "Detailed root lifts, terminal reasons, budgets, field hashes, active snapshots and uncertainty scope are in the JSON.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")


def root_tube_plot():
    """Analytic moving-root diagnostic for the §23 translated-well control."""
    import matplotlib.pyplot as plt

    alpha = np.linspace(-0.05, 0.05, 200)
    zeta = np.linspace(1.35, 4.95, 300)
    values = np.cos(alpha[:, None] - zeta[None, :])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.contourf(
        alpha, zeta, values.T, levels=(-1.01, 0, 1.01), colors=("#a9d7bf", "#f3d2d2")
    )
    ax.plot(alpha, alpha + np.pi / 2, "k-", label="continued incoming root")
    ax.plot(alpha, alpha + 3 * np.pi / 2, "k--", label="first outgoing root")
    ax.axhline(np.pi / 2, color="#ba6b25", lw=0.8, label="fixed scan node")
    ax.axhline(3 * np.pi / 2, color="#ba6b25", lw=0.8)
    ax.set_xlabel("alpha (rad)")
    ax.set_ylabel("lifted zeta (rad)")
    ax.set_title("B=2+cos(alpha−zeta), b=2: roots move through scan nodes")
    ax.legend(fontsize=7, loc="center right")
    fig.savefig(PLOTS / "translated-root-tube.png", dpi=150)
    plt.close(fig)


def main():
    PLOTS.mkdir(exist_ok=True)
    root_tube_plot()
    old_r2 = json.loads(R2.read_text())
    old_r3 = json.loads(R3.read_text())
    r2_by_key = {(c["file_index"], c["lambda_n"]): c for c in old_r2["cases"]}
    code_files = (
        "branch_atlas.py",
        "contour_trace.py",
        "forward_catalogue.py",
        "seed_search.py",
        "atlas_corridor.py",
    )
    evidence = {
        "milestone": "R3.5",
        "source": "h(rho)=1 declared default; contours are source-independent",
        "pitch_scope": "R0 sampled radially global extrema estimates, not field enclosures",
        "scope": "represented Fourier/interpolated field, local root certificates and numerical action estimates; no f/field enclosure",
        "hardware": platform.platform(),
        "workers": 1,
        "cache_state": "fresh field object per physical case; warm OS cache uncontrolled",
        "git_head": __import__("subprocess")
        .check_output(["git", "rev-parse", "HEAD"], text=True)
        .strip(),
        "code_sha256": {
            name: sha256(ROOT / "alpha_analysis/j_connectivity" / name)
            for name in code_files
        },
        "r2_source_sha256": sha256(R2),
        "r3_source_sha256": sha256(R3),
        "original_locations": old_r3["seed_choices"],
        "contour_resolution_controls": {
            "coarse": asdict(
                ContourConfig(
                    step=0.04,
                    max_steps=80,
                    max_certificate_boxes=4096,
                    max_certificate_depth=10,
                )
            ),
            "fine": asdict(
                ContourConfig(
                    step=0.02,
                    max_steps=160,
                    max_certificate_boxes=4096,
                    max_certificate_depth=10,
                )
            ),
        },
        "seed_search_controls": asdict(SeedSearchConfig()),
        "cases": [],
    }
    started = time.perf_counter()
    for old_case in old_r3["cases"]:
        key = (old_case["file_index"], old_case["lambda_n"])
        print("R3.5 case", key, flush=True)
        case = run_case(r2_by_key[key], old_case, evidence)
        evidence["cases"].append(case)
        evidence["elapsed_seconds"] = time.perf_counter() - started
        OUT.write_text(
            json.dumps(json_safe(evidence), indent=2, allow_nan=False) + "\n"
        )
        print(
            "completed",
            key,
            round(case["wall_seconds"], 2),
            "s",
            case.get("case_failure", ""),
            flush=True,
        )
    write_report(evidence)


if __name__ == "__main__":
    main()
