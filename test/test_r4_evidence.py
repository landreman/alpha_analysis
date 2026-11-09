"""R4 full physical-cohort provenance and no-false-classification checks."""

import hashlib
import json
from pathlib import Path

import numpy as np

from alpha_analysis import DATA_DIR
from alpha_analysis.j_connectivity.ordinary_accessibility import (
    OrdinaryMesh,
    ordinary_bounds,
    reachable_area,
)

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r4_real_matrix_retains_coverage_and_classified_oracle_comparisons():
    path = ROOT / "docs/validation/r4-ordinary-bounds.json"
    evidence = json.loads(path.read_text())
    assert evidence["r3_5_sha256"] == sha(
        ROOT / "docs/validation/r3-5-feasibility-matrix.json"
    )
    for name, digest in evidence["code_sha256"].items():
        assert digest == sha(ROOT / "alpha_analysis/j_connectivity" / name)
    assert evidence["workers"] == 1 and "h(rho)=1" in evidence["source"]
    assert "not field enclosures" in evidence["pitch_scope"]
    rows = evidence["cases"]
    assert len(rows) == 30
    assert {(r["file_index"], r["lambda_n"]) for r in rows} == {
        (i, p) for i in range(5) for p in (0.05, 0.1, 0.5, 0.8, 0.9, 0.95)
    }
    assert {
        row["file_index"]
        for row in rows
        if any(level["triangles"] for level in row["levels"].values())
    } == set(range(5))
    for row in rows:
        assert row["field_sha256"] == sha(Path(DATA_DIR) / row["file"])
        assert "failure" not in row
        assert row["wall_seconds"] < 600
        assert set(row["levels"]) == {"coarse", "fine"}
        for level in row["levels"].values():
            assert "PL" in level["scope"]
            np.testing.assert_allclose(
                level["fixed_domain_area"], 2 * np.pi, atol=1e-12
            )
            assert level["unknown_geometric_area"] > 0
            if level["triangles"]:
                mesh = OrdinaryMesh(**level["mesh_inputs"])
                replayed = ordinary_bounds(mesh, level["bin_edges"])
                assert replayed.bit_changes == level["bit_changes"]
                np.testing.assert_allclose(
                    reachable_area(mesh, replayed),
                    level["reachable_area"],
                    rtol=1e-12,
                    atol=1e-12,
                )
                nested_edges = np.sort(
                    np.r_[
                        level["bin_edges"],
                        (np.array(level["bin_edges"][:-1]) + level["bin_edges"][1:])
                        / 2,
                    ]
                )
                nested = ordinary_bounds(mesh, nested_edges)
                np.testing.assert_allclose(
                    reachable_area(mesh, nested),
                    level["nested_reachable_area"],
                    rtol=1e-12,
                    atol=1e-12,
                )
                for comp in level.get("seed_comparisons", []):
                    assert (
                        replayed.classify(comp["triangle"], comp["PL_action"])
                        == comp["status"]
                    )
                assert level["bit_changes"] <= level["finite_bit_limit"]
                low, high = level["reachable_area"]
                low2, high2 = level["nested_reachable_area"]
                assert -1e-12 <= low <= low2 + 1e-12
                assert low2 <= high2 + 1e-12 <= high + 2e-12
                assert high <= level["covered_area"] + 1e-12
            else:
                assert level["status"] == "UNKNOWN"
    for i, pitch, expected in ((2, 0.8, "INACCESSIBLE"), (3, 0.1, "ACCESSIBLE")):
        row = next(r for r in rows if r["file_index"] == i and r["lambda_n"] == pitch)
        for name in ("coarse", "fine"):
            assert row["full_corridor_reuse"][name]["cells"] > 10
            assert row["full_corridor_reuse"][name]["owned_area"] > 0
            assert row["oracle_comparisons"][name]["status"] == expected
            comparisons = row["levels"][name]["seed_comparisons"]
            assert comparisons
            for comp in comparisons:
                assert comp["status"] != (
                    "INACCESSIBLE" if expected == "ACCESSIBLE" else "ACCESSIBLE"
                )
