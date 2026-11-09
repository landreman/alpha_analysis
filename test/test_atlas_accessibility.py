"""R4 production adapter checks against real roots and independent contours."""

import json
from pathlib import Path

import numpy as np
import pytest

from alpha_analysis import BoozerField, DATA_DIR
from alpha_analysis.j_connectivity.atlas_accessibility import ordinary_model_from_atlas
from alpha_analysis.j_connectivity.atlas_corridor import (
    certify_local_atlas_corridor,
    embed_corridor_in_atlas,
)
from alpha_analysis.j_connectivity.branch_atlas import AtlasConfig, build_atlas
from alpha_analysis.j_connectivity.contour_trace import (
    ContourConfig,
    DirectContourOracle,
    ContourStatus,
)
from alpha_analysis.j_connectivity.ordinary_accessibility import (
    ordinary_bounds,
    reachable_area,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField


def test_production_bounds_bracket_independent_synthetic_contours():
    # B is independent of s, so each complete alpha line has constant action
    # and reaches s=1. Two distinct wells have overlapping action ranges.
    coeff = np.array([[2.0], [-1.0], [0.03]])
    field = SyntheticFourierField(
        1,
        np.array([0, 0, 1]),
        np.array([0, 2, 0]),
        coeff,
        np.zeros_like(coeff),
        np.array([0.0]),
        np.array([3.0]),
        np.array([0.0]),
    )
    atlas = build_atlas(field, 2, AtlasConfig(3, 5, 2))
    model = ordinary_model_from_atlas(atlas)
    assert model.mesh is not None and not model.failures
    mesh = model.mesh
    bounds = ordinary_bounds(
        mesh, np.linspace(mesh.action.min() - 0.01, mesh.action.max() + 0.01, 17)
    )
    oracle = DirectContourOracle(field, 2, ContourConfig(step=0.04, max_steps=60))
    seed = oracle.seed(0.5, 0.4)
    result = oracle.query(seed)
    assert result.status is ContourStatus.ACCESSIBLE
    assert model.unknown_geometric_area == 0
    # Interior triangle centroids have exact radial PL paths to EDGE.
    assert bounds.lower.any()
    low, high = reachable_area(mesh, bounds)
    assert low > 0 and low <= 4 * np.pi <= high + 1e-12
    # No mesh vertex may belong to different physical well ordinals.
    labels = {}
    for ids, (_, branch) in zip(mesh.triangles, model.cell_branches):
        for vertex in ids:
            assert vertex not in labels or labels[vertex] == branch
            labels[vertex] = branch


@pytest.mark.parametrize("file_index,pitch", [(2, 0.8), (3, 0.1)])
def test_real_ordinary_bounds_compare_classified_oracle_paths(file_index, pitch):
    """Use the two R3.5 classified paths; unknown exterior cannot become closure."""
    evidence = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/validation/r3-5-feasibility-matrix.json"
        ).read_text()
    )
    row = next(
        c
        for c in evidence["cases"]
        if c["file_index"] == file_index and c["lambda_n"] == pitch
    )
    field = BoozerField.from_boozmn(Path(DATA_DIR) / row["file"])
    seed_data = row["original_r3_seeds"]["fine"]
    oracle = DirectContourOracle(field, row["b"])
    seed = oracle.seed(seed_data["s"], seed_data["alpha"], seed_data["zeta_in"])
    corridor = certify_local_atlas_corridor(oracle, seed)
    assert corridor.certified, corridor.reason
    for n_s, n_alpha in ((3, 4), (5, 8)):
        atlas = embed_corridor_in_atlas(
            build_atlas(field, row["b"], AtlasConfig(n_s, n_alpha, 2)), corridor
        )
        model = ordinary_model_from_atlas(atlas)
        assert model.mesh is not None and not model.failures
        mesh = model.mesh
        assert mesh.frontier_vertices.size
        bounds = ordinary_bounds(
            mesh, np.linspace(mesh.action.min() - 1e-5, mesh.action.max() + 1e-5, 17)
        )
        assert row["current_original"]["fine"]["status"] in (
            "ACCESSIBLE",
            "INACCESSIBLE",
        )
        # These local patches do not reach EDGE and cannot certify a positive
        # field witness. Their unknown exterior must keep remote influence alive.
        assert not bounds.lower.any()
        assert bounds.upper.any()
        comparisons = []
        q = np.array([seed.s, seed.alpha % (2 * np.pi)])
        for i, ids in enumerate(mesh.triangles):
            vertices = mesh.points[ids]
            weights = np.linalg.solve(
                np.vstack((vertices.T, np.ones(3))), np.r_[q, 1.0]
            )
            if min(weights) >= -1e-10:
                action = float(np.dot(weights, mesh.action[ids]))
                status = bounds.classify(i, action)
                oracle_status = row["current_original"]["fine"]["status"]
                assert status != (
                    "INACCESSIBLE" if oracle_status == "ACCESSIBLE" else "ACCESSIBLE"
                )
                comparisons.append(status)
        assert comparisons
        lo, hi = reachable_area(mesh, bounds)
        assert lo == 0 < hi
        assert model.unknown_geometric_area > 0
        assert "PL" in model.scope


def test_full_real_corridor_preserves_ownership_and_possible_exterior():
    """The R3.5 closed corridor remains disjoint and open to missing geometry."""
    from alpha_analysis.j_connectivity.atlas_accessibility import (
        overlay_certified_cells,
    )
    from alpha_analysis.j_connectivity.branch_atlas import AtlasCell

    evidence = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/validation/r3-5-feasibility-matrix.json"
        ).read_text()
    )
    row = next(
        c for c in evidence["cases"] if c["file_index"] == 2 and c["lambda_n"] == 0.8
    )
    field = BoozerField.from_boozmn(Path(DATA_DIR) / row["file"])
    saved = row["full_path_corridors"]["coarse"]
    cells = tuple(
        AtlasCell(
            **{
                **c,
                "s_interval": tuple(c["s_interval"]),
                "alpha_interval": tuple(c["alpha_interval"]),
                "matched_local_branches": tuple(c["matched_local_branches"]),
            }
        )
        for c in saved["disjoint_owned_cells"]
    )
    base = build_atlas(field, row["b"], AtlasConfig(3, 4, 2))
    refined = overlay_certified_cells(base, cells)
    np.testing.assert_allclose(
        sum(c.area for c in refined.cells), 2 * np.pi, atol=1e-12
    )
    model = ordinary_model_from_atlas(refined)
    assert not model.failures and model.mesh is not None
    mesh = model.mesh
    np.testing.assert_allclose(sum(mesh.areas), saved["owned_area"], atol=1e-12)
    assert mesh.frontier_vertices.size and not mesh.edge_vertices.size
    bounds = ordinary_bounds(
        mesh, np.linspace(mesh.action.min() - 1e-6, mesh.action.max() + 1e-6, 9)
    )
    assert not bounds.lower.any() and bounds.upper.any()
    with pytest.raises(ValueError, match="disjoint"):
        overlay_certified_cells(base, (cells[0], cells[0]))
