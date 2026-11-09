"""Reproduce R4 finite-model comparisons for all 30 prescribed physical cases.

OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/alpha-r4-mpl .venv/bin/python
    docs/validation/r4_ordinary_bounds.py

No numerical field/action envelope, population bound, or f is claimed. Saved
R3.5 seed identities and b values are reused, not promoted to certificates.
"""

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from alpha_analysis import BoozerField, DATA_DIR
from alpha_analysis.j_connectivity.atlas_accessibility import (
    ordinary_model_from_atlas,
    overlay_certified_cells,
)
from alpha_analysis.j_connectivity.atlas_corridor import (
    certify_local_atlas_corridor,
    embed_corridor_in_atlas,
    _canonical_alpha_boxes,
)
from alpha_analysis.j_connectivity.branch_atlas import (
    AtlasConfig,
    AtlasCell,
    build_atlas,
)
from alpha_analysis.j_connectivity.contour_trace import (
    ContourConfig,
    DirectContourOracle,
)
from alpha_analysis.j_connectivity.ordinary_accessibility import (
    OrdinaryMesh,
    ordinary_bounds,
    reachable_area,
    plot_ordinary_bounds,
)
from r3_5_feasibility_matrix import CaseGuard, json_safe, sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE / "r3-5-feasibility-matrix.json"
OUT = HERE / "r4-ordinary-bounds.json"
PLOTS = HERE / "r4-ordinary-plots"


def run_case(old):
    guard = CaseGuard()
    row = {
        key: old[key] for key in ("file", "file_index", "lambda_n", "b", "field_sha256")
    }
    started_cpu = time.process_time()
    try:
        with guard:
            field = BoozerField.from_boozmn(Path(DATA_DIR) / row["file"])
            assert row["field_sha256"] == sha256(Path(DATA_DIR) / row["file"])
            data = old["original_r3_seeds"]["fine"]
            if data is None:
                seeds = old["seed_search"]["seeds"]
                data = seeds[0] if seeds else None
            oracle = DirectContourOracle(field, row["b"])
            seed = None
            if data is not None:
                try:
                    seed = oracle.seed(data["s"], data["alpha"], data["zeta_in"])
                except RuntimeError as error:
                    row["seed_failure"] = str(error)
            row["seed"] = data
            row["original_classification"] = old["current_original"]["fine"]["status"]
            guard.stage = "certified local root corridor"
            corridor = (
                None if seed is None else certify_local_atlas_corridor(oracle, seed)
            )
            row["corridor_status"] = (
                "NO_SEED"
                if corridor is None
                else ("CERTIFIED" if corridor.certified else "UNKNOWN")
            )
            row["corridor_reason"] = None if corridor is None else corridor.reason
            row["levels"] = {}
            for label, ns, na, bins in (("coarse", 3, 4, 8), ("fine", 5, 8, 16)):
                guard.stage = f"{label} fixed atlas and ordinary model"
                start = time.perf_counter()
                atlas = build_atlas(field, row["b"], AtlasConfig(ns, na, 2))
                if corridor is not None and corridor.certified:
                    atlas = embed_corridor_in_atlas(atlas, corridor)
                saved_local = old.get("local_corridor", {})
                if saved_local.get("status") == "CERTIFIED":
                    from dataclasses import replace

                    raw = saved_local["union"]
                    certified_local = AtlasCell(
                        **{
                            **raw,
                            "s_interval": tuple(raw["s_interval"]),
                            "alpha_interval": tuple(raw["alpha_interval"]),
                            "matched_local_branches": tuple(
                                raw["matched_local_branches"]
                            ),
                        }
                    )
                    local_cells = tuple(
                        replace(
                            certified_local,
                            s_interval=(s0, s1),
                            alpha_interval=(a0, a1),
                            area=(s1 - s0) * (a1 - a0),
                        )
                        for s0, s1, a0, a1 in _canonical_alpha_boxes(certified_local)
                    )
                    atlas = overlay_certified_cells(atlas, local_cells)
                    row["local_corridor_reuse"] = {
                        "step": saved_local["step"],
                        "seed": saved_local["seed"],
                        "cells": len(local_cells),
                        "area": sum(c.area for c in local_cells),
                        "scope": saved_local["scope"],
                    }
                saved_corridor = old.get("full_path_corridors", {}).get(label)
                if saved_corridor is not None:
                    corridor_cells = tuple(
                        AtlasCell(
                            **{
                                **c,
                                "s_interval": tuple(c["s_interval"]),
                                "alpha_interval": tuple(c["alpha_interval"]),
                                "matched_local_branches": tuple(
                                    c["matched_local_branches"]
                                ),
                            }
                        )
                        for c in saved_corridor["disjoint_owned_cells"]
                    )
                    atlas = overlay_certified_cells(atlas, corridor_cells)
                    row.setdefault("full_corridor_reuse", {})[label] = {
                        "cells": len(corridor_cells),
                        "owned_area": sum(c.area for c in corridor_cells),
                        "scope": "inherited R3.5 represented-field cell root census; new corner count/action checks; no true-arc enclosure",
                    }
                model = ordinary_model_from_atlas(atlas)
                level = {
                    "n_s": ns,
                    "n_alpha": na,
                    "bins": bins,
                    "unknown_geometric_area": model.unknown_geometric_area,
                    "failures": model.failures,
                    "scope": model.scope,
                    "fixed_domain_area": sum(c.area for c in atlas.cells),
                }
                if model.mesh is None:
                    level.update(
                        status="UNKNOWN",
                        reason="no certified positive cells with known vertex actions",
                        triangles=0,
                        covered_area=0,
                        possible_covered_area=0,
                    )
                else:
                    mesh = model.mesh
                    span = max(float(np.ptp(mesh.action)), 1e-8)
                    edges = np.linspace(
                        mesh.action.min() - 0.01 * span,
                        mesh.action.max() + 0.01 * span,
                        bins + 1,
                    )
                    if seed is not None and edges[0] < seed.action_length < edges[-1]:
                        edges = np.unique(np.r_[edges, seed.action_length])
                    level["mesh_inputs"] = {
                        "points": mesh.points.tolist(),
                        "triangles": mesh.triangles.tolist(),
                        "action": mesh.action.tolist(),
                        "edge_vertices": mesh.edge_vertices.tolist(),
                        "frontier_vertices": mesh.frontier_vertices.tolist(),
                        "action_error": mesh.action_error.tolist(),
                        "certified": mesh.certified.tolist(),
                        "level_connected": mesh.level_connected.tolist(),
                    }
                    level["bin_edges"] = edges.tolist()
                    level["cell_branches"] = model.cell_branches
                    bound_start = time.perf_counter()
                    bounds = ordinary_bounds(mesh, edges)
                    low, high = reachable_area(mesh, bounds)
                    # A nested bin comparison on *identical* geometry; changing
                    # atlas geometry is recorded separately without asserting nesting.
                    refined = ordinary_bounds(
                        mesh, np.sort(np.r_[edges, (edges[:-1] + edges[1:]) / 2])
                    )
                    low2, high2 = reachable_area(mesh, refined)
                    assert low <= low2 + 1e-12 and high2 <= high + 1e-12
                    level.update(
                        status="FINITE_MODEL_BOUNDS",
                        triangles=len(mesh.triangles),
                        vertices=len(mesh.points),
                        frontier_vertices=len(mesh.frontier_vertices),
                        edge_vertices=len(mesh.edge_vertices),
                        covered_area=float(mesh.areas.sum()),
                        reachable_area=[low, high],
                        nested_reachable_area=[low2, high2],
                        bit_changes=bounds.bit_changes,
                        finite_bit_limit=2 * bounds.lower.size,
                        bound_seconds=time.perf_counter() - bound_start,
                    )
                    assert bounds.bit_changes <= 2 * bounds.lower.size
                    if seed is not None:
                        comparisons = []
                        q = np.array([seed.s, seed.alpha % (2 * np.pi)])
                        for i, ids in enumerate(mesh.triangles):
                            weights = np.linalg.solve(
                                np.vstack((mesh.points[ids].T, np.ones(3))), np.r_[q, 1]
                            )
                            if weights.min() >= -1e-10:
                                action = float(weights @ mesh.action[ids])
                                comparisons.append(
                                    {
                                        "triangle": i,
                                        "PL_action": action,
                                        "direct_action": seed.action_length,
                                        "action_difference_estimate": action
                                        - seed.action_length,
                                        "status": bounds.classify(i, action),
                                    }
                                )
                        level["seed_comparisons"] = comparisons
                    if row["file_index"] == 2 and row["lambda_n"] == 0.8:
                        fig = plot_ordinary_bounds(
                            mesh, bounds, title=f"DMercFail lambda_n=0.8, h=1, {label}"
                        )
                        fig.savefig(PLOTS / f"dmerc-{label}.png", dpi=140)
                        import matplotlib.pyplot as plt

                        plt.close(fig)
                level["wall_seconds"] = time.perf_counter() - start
                row["levels"][label] = level
            if (row["file_index"], row["lambda_n"]) in ((2, 0.8), (3, 0.1)):
                guard.stage = "independent classified ordinary oracle comparisons"
                row["oracle_comparisons"] = {}
                for label, step, steps in (("coarse", 0.04, 80), ("fine", 0.02, 160)):
                    direct = DirectContourOracle(
                        field,
                        row["b"],
                        ContourConfig(
                            step=step, max_steps=steps, max_certificate_boxes=4096
                        ),
                    )
                    result = direct.query(direct.seed(seed.s, seed.alpha, seed.zeta_in))
                    row["oracle_comparisons"][label] = {
                        "status": result.status.name,
                        "reason": result.reason,
                        "scope": result.bound_scope,
                    }
                    for comp in row["levels"][label].get("seed_comparisons", []):
                        if result.status.name == "ACCESSIBLE":
                            assert comp["status"] != "INACCESSIBLE"
                        elif result.status.name == "INACCESSIBLE":
                            assert comp["status"] != "ACCESSIBLE"
    except (TimeoutError, RuntimeError, ValueError, ArithmeticError) as error:
        row["failure"] = str(error)
    row["wall_seconds"] = time.perf_counter() - guard.started
    row["cpu_seconds"] = time.process_time() - started_cpu
    row["snapshots"] = guard.snapshots
    return row


def main():
    PLOTS.mkdir(exist_ok=True)
    points = np.array(
        [(s, a) for s in np.linspace(0.1, 1, 5) for a in np.linspace(0, 1, 7)]
    )
    triangles = []
    for i in range(4):
        for j in range(6):
            k = i * 7 + j
            triangles.extend(((k, k + 7, k + 8), (k, k + 8, k + 1)))
    mesh = OrdinaryMesh(
        points,
        np.array(triangles),
        points.sum(axis=1),
        edge_vertices=np.flatnonzero(points[:, 0] == 1),
    )
    fig = plot_ordinary_bounds(
        mesh,
        ordinary_bounds(mesh, np.linspace(0, 2, 18)),
        title="analytic chart A=s+alpha; EDGE at s=1; unweighted",
    )
    fig.savefig(PLOTS / "analytic-action-bins.png", dpi=140)
    import matplotlib.pyplot as plt

    plt.close(fig)
    start = time.perf_counter()
    evidence = {
        "baseline": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "hardware": platform.platform(),
        "workers": 1,
        "source": "h(rho)=1; geometric ds d-alpha only, no K weight",
        "cache": "cold field objects per case; saved R3.5 root-cell certificates reused and their construction cost excluded; no scan/action cache reused; OS file cache may be warm",
        "pitch_scope": "saved R0 radially global extrema estimates; not field enclosures",
        "r3_5_sha256": sha256(BASE),
        "scope": "finite PL model bounds; field action/interpolation/arc errors unbounded; no physical slice or f acceptance",
        "code_sha256": {
            name: sha256(ROOT / "alpha_analysis/j_connectivity" / name)
            for name in ("ordinary_accessibility.py", "atlas_accessibility.py")
        },
        "cases": [],
    }
    for old in json.loads(BASE.read_text())["cases"]:
        row = run_case(old)
        evidence["cases"].append(row)
        OUT.write_text(json.dumps(json_safe(evidence), indent=2) + "\n")
        print(
            row["file_index"],
            row["lambda_n"],
            round(row["wall_seconds"], 2),
            row.get("failure", "OK"),
            flush=True,
        )
    evidence["total_wall_seconds"] = time.perf_counter() - start
    OUT.write_text(json.dumps(json_safe(evidence), indent=2) + "\n")


if __name__ == "__main__":
    main()
