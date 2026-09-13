"""R3.5 local atlas corridors matched to direct ordinary contours."""

import json
from pathlib import Path

import numpy as np
import pytest

from alpha_analysis import BoozerField, DATA_DIR
from alpha_analysis.j_connectivity.atlas_corridor import (
    certify_contour_path_corridor,
    certify_local_atlas_corridor,
    embed_corridor_in_atlas,
)
from alpha_analysis.j_connectivity.branch_atlas import (
    AtlasConfig,
    _line,
    _owned,
    build_atlas,
)
from alpha_analysis.j_connectivity.contour_trace import (
    ContourConfig,
    ContourPath,
    ContourPoint,
    ContourStatus,
    DirectContourOracle,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField


def test_full_path_corridor_does_not_relabel_multiwell_area_as_one():
    """A selected branch cannot claim full geometric ownership of two wells (§8.2)."""
    coefficients = np.array([[2.0], [1.0]])
    field = SyntheticFourierField(
        1,
        np.array([0, 0]),
        np.array([0, 2]),
        coefficients,
        np.zeros_like(coefficients),
        np.array([0.0]),
        np.array([3.0]),
        np.array([0.0]),
    )
    oracle = DirectContourOracle(field, 2.0)
    first = oracle.seed(0.45, 0.0)
    last = oracle.sample(0.55, 0.01, first)
    path = ContourPath((first, last), False, False)
    corridor = certify_contour_path_corridor(oracle, path, max_depth=0)
    assert corridor.status == "UNKNOWN"
    assert "multiple physical wells" in corridor.reason
    assert not corridor.owned_cells


def test_full_path_corridor_preserves_unique_lifted_ownership():
    """A synthetic edge contour has one disjoint atlas owner along its path (§8.2)."""
    coefficients = np.array([[2.0, 0.15], [-1.0, 0.0], [0.1, 0.0]])
    field = SyntheticFourierField(
        1,
        np.array([0, 0, 1]),
        np.array([0, 1, 0]),
        coefficients,
        np.zeros_like(coefficients),
        np.array([0.0]),
        np.array([3.0]),
        np.array([0.0]),
    )
    oracle = DirectContourOracle(field, 2.0, ContourConfig(step=0.025, max_steps=300))
    original = oracle.seed(0.5, 0.0)
    lifted = oracle.sample(
        original.s,
        original.alpha,
        ContourPoint(
            original.s,
            original.alpha,
            original.zeta_in + oracle.period,
            original.zeta_out + oracle.period,
            original.action_length,
        ),
    )
    result = oracle.query(lifted)
    assert result.status is ContourStatus.ACCESSIBLE
    corridor = certify_contour_path_corridor(oracle, result.witness, max_boxes=1000)
    assert corridor.status == "CERTIFIED", corridor.reason
    assert corridor.branch_id == 0
    assert corridor.owned_area > 0
    assert corridor.attempted_boxes <= 1000
    np.testing.assert_allclose(
        sum(cell.area for cell in corridor.owned_cells), corridor.owned_area, atol=1e-14
    )


@pytest.mark.parametrize(
    "file_index,pitch,s,alpha",
    [
        (0, 0.5, 0.5, 0.0),
        (1, 0.5, 0.5, 0.0),
        (2, 0.5, 0.5, 0.0),
        (3, 0.5, 0.5, np.pi / 2),
        (4, 0.8, 0.99, np.pi / 2),
        (2, 0.8, 0.3, 0.0),
        (3, 0.1, 0.3, 0.0),
    ],
)
def test_real_atlas_corridors_match_continued_roots(file_index, pitch, s, alpha):
    """Each real field has adjacent positive cells on a continued A-segment (§23)."""
    rows = json.loads(
        (
            Path(__file__).resolve().parents[1] / "docs/validation/r2-atlas-matrix.json"
        ).read_text()
    )["cases"]
    row = next(
        item
        for item in rows
        if item["file_index"] == file_index and item["lambda_n"] == pitch
    )
    field = BoozerField.from_boozmn(Path(DATA_DIR) / row["file"])
    oracle = DirectContourOracle(field, row["b"])
    wells, _ = _owned(_line(field, s, alpha, 2), row["b"], False)
    assert wells
    well = wells[0]
    seed = oracle.sample(
        s, alpha, ContourPoint(s, alpha, well.zeta_in, well.zeta_out, np.nan)
    )
    corridor = certify_local_atlas_corridor(oracle, seed)
    assert corridor.certified, corridor.reason
    assert corridor.union.multiplicity_upper > 0
    assert all(
        c.multiplicity_upper == corridor.union.multiplicity_upper
        for c in corridor.cells
    )
    np.testing.assert_allclose(
        sum(c.area for c in corridor.cells), corridor.union.area, atol=1e-15, rtol=0
    )
    assert corridor.branch_id in corridor.union.matched_local_branches
    assert all(
        abs(point.action_length - seed.action_length) <= oracle.config.action_atol
        for point in corridor.points
    )
    fixed = build_atlas(field, row["b"], AtlasConfig(3, 4, 2))
    refined = embed_corridor_in_atlas(fixed, corridor)
    assert refined.known_owned_area > fixed.known_owned_area
    np.testing.assert_allclose(
        sum(c.area for c in refined.cells), 2 * np.pi, atol=1e-12
    )
    assert refined.unknown_area < fixed.unknown_area
