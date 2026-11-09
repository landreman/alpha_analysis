"""Root-labelled atlas to ordinary PL reachability (DESIGN §§8, 11, 23 R4).

This adapter preserves unknown boundary incidence. Vertex quadrature is an
estimate; the exact finite PL interpolant is the declared reachability model.
It is not promoted to a uniform action or physical field enclosure.
"""

from dataclasses import dataclass

import numpy as np

from .branch_atlas import BranchAtlas, _line, _owned, classify_cell
from .ordinary_accessibility import OrdinaryMesh


@dataclass(frozen=True)
class AtlasOrdinaryModel:
    """Finite PL model and coverage diagnostics; no population or f bound."""

    mesh: OrdinaryMesh | None
    cell_branches: tuple[tuple[int, int], ...]
    unknown_geometric_area: float
    failures: tuple[str, ...]
    scope: str = (
        "PL interpolant of estimated vertex actions; represented-field root census only"
    )


def ordinary_model_from_atlas(atlas: BranchAtlas) -> AtlasOrdinaryModel:
    """Triangulate certified rectangular branches with unique integration owners.

    Neighbor cells share IDs only after a union root census certifies the same
    local ordinal and every sampled count agrees. Nonconforming interfaces,
    unproved seams, and all remaining exterior boundaries stay upper frontiers.
    This is conservative even when it sacrifices an available lower connection.
    No oracle path or sampled chord is used as a physical coverage certificate.
    """
    samples = {}
    failures = []

    def sample(s, alpha):
        key = (float(s), float(alpha))
        if key not in samples:
            samples[key] = _owned(
                _line(atlas.field, *key, atlas.config.periods), atlas.b, True
            )[0]
        return samples[key]

    patches = []
    unknown_cells = set()
    for index, cell in enumerate(atlas.cells):
        if cell.multiplicity_upper is None:
            unknown_cells.add(index)
            continue
        if cell.multiplicity_upper == 0:
            continue
        s0, s1 = cell.s_interval
        a0, a1 = cell.alpha_interval
        corners = ((s0, a0), (s1, a0), (s1, a1), (s0, a1))
        wells = [sample(*q) for q in corners]
        if any(len(w) != cell.multiplicity_upper for w in wells):
            failures.append(
                f"cell {index}: sampled count disagrees with root certificate"
            )
            unknown_cells.add(index)
            continue
        for branch in cell.matched_local_branches:
            if any(
                branch >= len(w) or not np.isfinite(w[branch].action_length)
                for w in wells
            ):
                failures.append(f"cell {index} branch {branch}: unknown vertex action")
                # Geometric unknown area is not a multiplicity or K-weight bound.
                unknown_cells.add(index)
                continue
            patches.append(
                (index, branch, corners, [w[branch].action_length for w in wells])
            )

    unknown_area = float(sum(atlas.cells[i].area for i in unknown_cells))
    if not patches:
        return AtlasOrdinaryModel(None, (), unknown_area, tuple(failures))
    parent = np.arange(4 * len(patches))

    def root(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    links = set()
    for i, (_, branch, corners, _) in enumerate(patches):
        for j in range(i):
            other_index, other_branch, other, _ = patches[j]
            common = set(corners) & set(other)
            if not common or branch != other_branch:
                continue
            cell = atlas.cells[patches[i][0]]
            neighbor = atlas.cells[other_index]
            union = classify_cell(
                atlas.field,
                atlas.b,
                (
                    min(cell.s_interval[0], neighbor.s_interval[0]),
                    max(cell.s_interval[1], neighbor.s_interval[1]),
                ),
                (
                    min(cell.alpha_interval[0], neighbor.alpha_interval[0]),
                    max(cell.alpha_interval[1], neighbor.alpha_interval[1]),
                ),
                atlas.config.periods,
                atlas.config.cell_subdivisions,
            )
            if (
                union.multiplicity_upper is None
                or branch not in union.matched_local_branches
            ):
                continue
            # Check all corners of both cells against the union census. Matching
            # ordinals have meaning only in this certified common lifted window.
            if any(
                len(sample(*q)) != union.multiplicity_upper
                for q in set(corners) | set(other)
            ):
                continue
            for q in common:
                x, y = 4 * i + corners.index(q), 4 * j + other.index(q)
                parent[root(x)] = root(y)
            if len(common) == 2:
                links.add((i, frozenset(common)))
                links.add((j, frozenset(common)))

    ids, points, actions = {}, [], []
    triangles, owners = [], []
    edge, frontier = set(), set()
    for i, (cell_index, branch, corners, values) in enumerate(patches):
        local = []
        for k, (q, value) in enumerate(zip(corners, values)):
            key = root(4 * i + k)
            if key not in ids:
                ids[key] = len(points)
                points.append(q)
                actions.append(value)
            local.append(ids[key])
        triangles.extend(
            ((local[0], local[1], local[2]), (local[0], local[2], local[3]))
        )
        owners.extend(((cell_index, branch),) * 2)
        for k, l in ((0, 1), (1, 2), (2, 3), (3, 0)):
            if corners[k][0] == corners[l][0] == 1.0:
                edge.update((local[k], local[l]))
            elif (i, frozenset((corners[k], corners[l]))) not in links:
                frontier.update((local[k], local[l]))
    mesh = OrdinaryMesh(
        np.array(points),
        np.array(triangles),
        np.array(actions),
        edge_vertices=np.array(sorted(edge), dtype=int),
        frontier_vertices=np.array(sorted(frontier), dtype=int),
    )
    return AtlasOrdinaryModel(mesh, tuple(owners), unknown_area, tuple(failures))


def overlay_certified_cells(atlas: BranchAtlas, cells) -> BranchAtlas:
    """Overlay disjoint certified root cells without sealing their complement.

    Input cells must be certified for this field, b, ownership window and branch
    convention. Subsets inherit their root census, never an action enclosure.
    Their fixed-domain complement retains the original atlas uncertainty (§8.2).
    """
    from dataclasses import replace
    from .atlas_corridor import _subtract_box

    cells = tuple(cells)
    boxes = []
    for cell in cells:
        s0, s1 = cell.s_interval
        a0, a1 = cell.alpha_interval
        if (
            not (0 <= s0 < s1 <= 1 and 0 <= a0 < a1 <= 2 * np.pi)
            or cell.multiplicity_upper is None
        ):
            raise ValueError("overlay requires canonical certified root cells")
        box = (s0, s1, a0, a1)
        for other in boxes:
            if min(s1, other[1]) > max(s0, other[0]) and min(a1, other[3]) > max(
                a0, other[2]
            ):
                raise ValueError("overlay root cells must have disjoint interiors")
        boxes.append(box)
    result = []
    for original in atlas.cells:
        s0, s1 = original.s_interval
        a0, a1 = original.alpha_interval
        residual = [(s0, s1, a0, a1)]
        for cell, box in zip(cells, boxes):
            low_s, high_s = max(s0, box[0]), min(s1, box[1])
            low_a, high_a = max(a0, box[2]), min(a1, box[3])
            if low_s >= high_s or low_a >= high_a:
                continue
            if (
                original.multiplicity_upper is not None
                and original.multiplicity_upper != cell.multiplicity_upper
            ):
                raise ArithmeticError("overlay disagrees with original root census")
            result.append(
                replace(
                    cell,
                    s_interval=(low_s, high_s),
                    alpha_interval=(low_a, high_a),
                    area=(high_s - low_s) * (high_a - low_a),
                )
            )
            residual = [
                part for remainder in residual for part in _subtract_box(remainder, box)
            ]
        for low_s, high_s, low_a, high_a in residual:
            result.append(
                replace(
                    original,
                    s_interval=(low_s, high_s),
                    alpha_interval=(low_a, high_a),
                    area=(high_s - low_s) * (high_a - low_a),
                )
            )
    if not np.isclose(
        sum(c.area for c in result),
        sum(c.area for c in atlas.cells),
        rtol=0,
        atol=1e-12,
    ):
        raise ArithmeticError("root overlay lost fixed-domain area")
    return BranchAtlas(atlas.field, atlas.b, atlas.samples, tuple(result), atlas.config)
