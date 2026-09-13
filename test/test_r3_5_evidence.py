"""Saved 30-case R3.5 scientific gate and provenance checks (§23)."""

import hashlib
import json
from pathlib import Path

import numpy as np

from alpha_analysis import DATA_DIR

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "docs/validation"
PITCHES = {0.05, 0.1, 0.5, 0.8, 0.9, 0.95}
BLOCKED = (
    (0, 0.1),
    (1, 0.5),
    (2, 0.05),
    (2, 0.1),
    (2, 0.5),
    (2, 0.8),
    (3, 0.05),
    (3, 0.1),
)
CORRIDORS = ((0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5), (4, 0.8), (2, 0.8), (3, 0.1))


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case(rows, file_index, pitch):
    return next(
        row
        for row in rows
        if row["file_index"] == file_index and row["lambda_n"] == pitch
    )


def _overlap_area(first, second):
    ds = max(
        0,
        min(first["s_interval"][1], second["s_interval"][1])
        - max(first["s_interval"][0], second["s_interval"][0]),
    )
    da = max(
        0,
        min(first["alpha_interval"][1], second["alpha_interval"][1])
        - max(first["alpha_interval"][0], second["alpha_interval"][0]),
    )
    return ds * da


def test_r3_5_feasibility_matrix_preserves_cases_and_gates():
    """The saved case matrix must demonstrate actual R3.5 real gates, not schema."""
    evidence = json.loads((VALIDATION / "r3-5-feasibility-matrix.json").read_text())
    r2_path = VALIDATION / "r2-atlas-matrix.json"
    r3_path = VALIDATION / "r3-contour-matrix.json"
    baseline = json.loads(r3_path.read_text())
    assert evidence["r2_source_sha256"] == _sha(r2_path)
    assert evidence["r3_source_sha256"] == _sha(r3_path)
    assert evidence["workers"] == 1 and "h(rho)=1" in evidence["source"]
    assert "not field enclosures" in evidence["pitch_scope"]
    for name, digest in evidence["code_sha256"].items():
        assert digest == _sha(ROOT / "alpha_analysis/j_connectivity" / name)
    rows = evidence["cases"]
    assert len(rows) == 30
    assert {(row["file_index"], row["lambda_n"]) for row in rows} == {
        (file_index, pitch) for file_index in range(5) for pitch in PITCHES
    }
    for row in rows:
        original = _case(baseline["cases"], row["file_index"], row["lambda_n"])
        assert row["file"] == original["file"] and row["b"] == original["b"]
        assert row["field_sha256"] == _sha(Path(DATA_DIR) / row["file"])
        assert row["under_600_second_guard"] and row["wall_seconds"] <= 600
        assert "case_failure" not in row
        for level in ("coarse", "fine"):
            probe = next(p for p in original["probes"] if p["resolution"] == level)
            assert row["original_r3_statuses"][level] == probe["status"]
            assert row["original_r3_seeds"][level] == probe.get("seed")
            assert row["current_original"][level]["status"] in {
                "NO_SEED",
                "UNKNOWN",
                "ACCESSIBLE",
                "INACCESSIBLE",
            }
            atlas = row["current_atlas"][level]
            np.testing.assert_allclose(
                atlas["fixed_domain_area"], 2 * np.pi, atol=1e-12
            )
            np.testing.assert_allclose(
                atlas["certified_positive_geometric_area"]
                + atlas["certified_empty_area"]
                + atlas["unknown_area"],
                2 * np.pi,
                atol=1e-12,
            )
        search = row["seed_search"]
        assert search["original_attempts"] == 5
        assert not search["population_empty_proved"]
        assert all(a["status"] and a["periods"] >= 0 for a in search["attempts"])
        assert len(search["seeds"]) == sum(a["new_seeds"] for a in search["attempts"])
        for snapshot in row["snapshots"]:
            assert snapshot["at_seconds"] in (60, 300, 600)
            assert row["wall_seconds"] >= snapshot["at_seconds"]

    # The exact eight R3 fine probes blocked by scan-node/root certification
    # now have the physically checked contour terminal at both resolutions.
    for index, pitch in BLOCKED:
        case = _case(rows, index, pitch)
        assert case["original_r3_statuses"]["fine"] == "UNKNOWN"
        expected = "ACCESSIBLE" if (index, pitch) == (3, 0.1) else "INACCESSIBLE"
        for level in ("coarse", "fine"):
            result = case["current_original"][level]
            assert result["status"] == expected, result["reason"]
            assert result["max_estimated_error_A"] is not None
            assert result["max_action_drift"] <= 2e-5
            if expected == "ACCESSIBLE":
                assert any(result["edge_reached"])
            else:
                assert all(result["closed"])

    assert all(
        _case(rows, index, pitch)["local_corridor"]["status"] == "CERTIFIED"
        for index, pitch in CORRIDORS
    )
    assert {
        row["file_index"]
        for row in rows
        if row["local_corridor"]["status"] == "CERTIFIED"
    } == set(range(5))
    assert all(
        row["seed_search"]["seeds"]
        for row in rows
        if row["original_r3_statuses"]["fine"] == "NO_SEED"
    )

    event = _case(rows, 2, 0.8)["discovered_event"]["event_discovery"]
    assert event["status"] == "verified"
    assert event["marginal_residual_B"] < 1e-8
    assert event["action_residual"] < 2e-6
    assert {port["role"] for port in event["event"]["ports"]} == {
        "parent",
        "child_1",
        "child_3",
    }
    assert {role for role, _ in event["one_sided_action_residuals"]} == {
        "parent",
        "child_1",
        "child_3",
    }

    for index, pitch, terminal in ((2, 0.8, "closed"), (3, 0.1, "edge")):
        case = _case(rows, index, pitch)
        for level in ("coarse", "fine"):
            corridor = case["full_path_corridors"][level]
            assert corridor["status"] == "CERTIFIED", corridor["reason"]
            assert corridor["path_terminal"] == terminal
            assert corridor["attempted_boxes"] <= 5000
            assert corridor["max_depth_used"] <= 12
            assert corridor["tile_count"] and corridor["owned_cell_count"]
            assert corridor["owned_area"] > 0
            assert corridor["additional_certified_positive_area_on_fixed_domain"] > 0
            assert (
                corridor["unknown_geometric_area_after_corridor"]
                < case["current_atlas"][level]["unknown_area"]
            )
            owned = corridor["disjoint_owned_cells"]
            np.testing.assert_allclose(
                sum(cell["area"] for cell in owned), corridor["owned_area"], atol=1e-12
            )
            for i, cell in enumerate(owned):
                assert 0 <= cell["s_interval"][0] < cell["s_interval"][1] <= 1
                assert (
                    0
                    <= cell["alpha_interval"][0]
                    < cell["alpha_interval"][1]
                    <= 2 * np.pi
                )
                assert cell["multiplicity_lower"] == cell["multiplicity_upper"] == 1
                assert cell["matched_local_branches"] == [corridor["branch_id"]]
                assert all(_overlap_area(cell, other) <= 1e-14 for other in owned[:i])
