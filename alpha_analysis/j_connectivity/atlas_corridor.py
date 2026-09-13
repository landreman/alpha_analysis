"""Local matching of direct contour segments to certified atlas cells (§§8.2, 23 R3.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .branch_atlas import (
    AtlasCell,
    BranchAtlas,
    _line,
    _owned,
    classify_cell,
    refine_atlas_cell,
)
from .contour_trace import ContourPath, ContourPoint, DirectContourOracle


@dataclass(frozen=True)
class AtlasCorridor:
    """Two adjacent owned cells and a crossing constant-A segment, or unknown.

    Cell areas use normalized s times alpha radians. ``branch_id`` is valid
    only in the certified union, with lifted zeta roots in radians (§8.2).
    """

    points: tuple[ContourPoint, ContourPoint]
    cells: tuple[AtlasCell, AtlasCell] | None
    union: AtlasCell | None
    branch_id: int | None
    reason: str | None

    @property
    def certified(self) -> bool:
        return self.cells is not None and self.branch_id is not None


@dataclass(frozen=True)
class PathAtlasCorridor:
    """Certified connected tile chain with disjoint fixed-domain ownership (§8.2).

    ``tiles`` cover the lifted numerical contour polyline. ``owned_cells`` are
    their alpha-periodic images after overlap subtraction, each owning exactly
    one continued incoming root. Areas are ds d-alpha, not K-weighted.
    """

    status: str
    reason: str | None
    tiles: tuple[AtlasCell, ...]
    owned_cells: tuple[AtlasCell, ...]
    attempted_boxes: int
    max_depth_used: int
    branch_id: int | None
    path: ContourPath

    @property
    def owned_area(self) -> float:
        return float(sum(cell.area for cell in self.owned_cells))


def _canonical_alpha_boxes(cell: AtlasCell):
    """Split a lifted q rectangle into its [0,2π) periodic images."""
    turn = 2 * np.pi
    first = int(np.floor(cell.alpha_interval[0] / turn))
    last = int(np.floor(cell.alpha_interval[1] / turn))
    for winding in range(first, last + 1):
        low = max(cell.alpha_interval[0], winding * turn)
        high = min(cell.alpha_interval[1], (winding + 1) * turn)
        if high > low:
            yield (
                cell.s_interval[0],
                cell.s_interval[1],
                low - winding * turn,
                high - winding * turn,
            )


def _subtract_box(box, blocker):
    """Return disjoint pieces of box outside one already owned rectangle."""
    s0, s1, a0, a1 = box
    t0, t1, b0, b1 = blocker
    x0, x1 = max(s0, t0), min(s1, t1)
    y0, y1 = max(a0, b0), min(a1, b1)
    if x1 <= x0 or y1 <= y0:
        return [box]
    pieces = [
        (s0, x0, a0, a1),
        (x1, s1, a0, a1),
        (x0, x1, a0, y0),
        (x0, x1, y1, a1),
    ]
    return [piece for piece in pieces if piece[1] > piece[0] and piece[3] > piece[2]]


def _unique_owned_cells(tiles, branch_id):
    owned = []
    for tile in tiles:
        for canonical in _canonical_alpha_boxes(tile):
            residual = [canonical]
            for blocker in owned:
                residual = [
                    part for box in residual for part in _subtract_box(box, blocker)
                ]
                if not residual:
                    break
            owned.extend(residual)
    return tuple(
        AtlasCell(
            (s0, s1),
            (a0, a1),
            (s1 - s0) * (a1 - a0),
            1,
            1,
            None,
            (branch_id,),
        )
        for s0, s1, a0, a1 in owned
    )


def certify_contour_path_corridor(
    oracle: DirectContourOracle,
    path: ContourPath,
    max_depth: int = 12,
    max_boxes: int = 5000,
    periods: int = 2,
    subdivisions: int = 12,
) -> PathAtlasCorridor:
    """Cover a numerical ordinary contour path with matched positive atlas tiles.

    Each accepted tile has a full represented-field root census. Failed tiles
    are subdivided along a corrected A-contour; an exhausted budget, root
    mismatch or corner-count mismatch leaves the entire path corridor unknown.
    Canonical overlap subtraction gives unique integration ownership (§§8, 23).
    """
    if max_depth < 0 or max_boxes < 1 or len(path.points) < 2:
        raise ValueError("invalid finite path-corridor work budget or path")
    attempts = 0
    deepest = 0
    tiles: list[AtlasCell] = []
    endpoints: list[tuple[ContourPoint, ContourPoint]] = []

    def cover(first: ContourPoint, last: ContourPoint, depth: int):
        nonlocal attempts, deepest
        attempts += 1
        deepest = max(deepest, depth)
        if attempts > max_boxes:
            raise RuntimeError("atlas corridor box budget exhausted")
        s0, s1 = sorted((first.s, last.s))
        a0, a1 = sorted((first.alpha, last.alpha))
        if s1 - s0 < 1e-10:
            s0, s1 = max(0.0, s0 - 1e-8), min(1.0, s1 + 1e-8)
        if a1 - a0 < 1e-10:
            a0, a1 = a0 - 1e-8, a1 + 1e-8
        cell = classify_cell(
            oracle.field,
            oracle.b,
            (s0, s1),
            (a0, a1),
            periods,
            subdivisions,
        )
        # The owned-cell output represents the full physical multiplicity on
        # each tile. A tile with multiple wells cannot be relabelled as one
        # merely because this path selects one of its branches.
        if cell.multiplicity_upper == 1:
            tiles.append(cell)
            endpoints.append((first, last))
            return
        if depth >= max_depth:
            reason = (
                "path tile contains multiple physical wells"
                if cell.multiplicity_upper is not None and cell.multiplicity_upper > 1
                else cell.unknown_reason
            )
            raise RuntimeError(f"atlas corridor depth budget exhausted: {reason}")
        predicted = np.array(
            [(first.s + last.s) / 2, (first.alpha + last.alpha) / (4 * np.pi)]
        )
        midpoint = oracle._correct(predicted, first, first.action_length)
        cover(first, midpoint, depth + 1)
        cover(midpoint, last, depth + 1)

    try:
        for first, last in zip(path.points[:-1], path.points[1:]):
            cover(first, last, 0)
        # Verify every tile's independent sampled count at its corners. Cache
        # shared corners and continued path points without trusting ordinates
        # alone to match a physical root lift.
        samples = {}

        def sampled(s, alpha):
            key = (float(s), float(alpha))
            if key not in samples:
                samples[key] = _owned(
                    _line(oracle.field, key[0], key[1], periods), oracle.b, False
                )[0]
            return samples[key]

        branch_id = None
        for cell, (first, last) in zip(tiles, endpoints):
            for s in cell.s_interval:
                for alpha in cell.alpha_interval:
                    if len(sampled(s, alpha)) != cell.multiplicity_upper:
                        raise RuntimeError(
                            "sampled root count disagrees with path tile certificate"
                        )
            for point in (first, last):
                matches = [
                    well.branch_id
                    for well in sampled(point.s, point.alpha)
                    if abs(well.zeta_in - point.zeta_in) <= 2e-6
                    and abs(well.zeta_out - point.zeta_out) <= 2e-6
                ]
                if len(matches) != 1:
                    raise RuntimeError("continued root lacks unique atlas ownership")
                if branch_id is None:
                    branch_id = matches[0]
                elif matches[0] != branch_id:
                    raise RuntimeError("root ordinal changes along atlas corridor")
        owned = _unique_owned_cells(tiles, branch_id)
        if not 0 < sum(c.area for c in owned) <= 2 * np.pi:
            raise RuntimeError("invalid fixed-domain corridor ownership area")
        return PathAtlasCorridor(
            "CERTIFIED", None, tuple(tiles), owned, attempts, deepest, branch_id, path
        )
    except (ValueError, RuntimeError, ArithmeticError) as error:
        return PathAtlasCorridor(
            "UNKNOWN", str(error), tuple(tiles), (), attempts, deepest, None, path
        )


def certify_local_atlas_corridor(
    oracle: DirectContourOracle,
    seed: ContourPoint,
    step: float = 1e-5,
    periods: int = 2,
    subdivisions: int = 12,
) -> AtlasCorridor:
    """Match a direct ordinary A-segment across two positive atlas cells.

    The union root census proves matching ordinal/lift through the shared
    edge. Corner sample counts guard each cell certificate; any mismatch or
    unseen root remains unknown. This is represented-field local coverage,
    not a weighted population or global accessibility enclosure (§8.4).
    """
    if not np.isfinite(step) or step <= 0:
        raise ValueError("corridor step must be positive")
    seed = oracle.sample(seed.s, seed.alpha, seed)
    gradient = oracle.action_gradient(seed)
    scaled = np.array([gradient[0], 2 * np.pi * gradient[1]])
    norm = float(np.linalg.norm(scaled))
    if norm <= oracle.config.gradient_floor:
        return AtlasCorridor((seed, seed), None, None, None, "flat contour tangent")
    tangent = np.array([-scaled[1], scaled[0]]) / norm
    x0 = np.array([seed.s, seed.alpha / (2 * np.pi)])
    try:
        left = oracle._correct(x0 - step * tangent, seed, seed.action_length)
        right = oracle._correct(x0 + step * tangent, seed, seed.action_length)
    except (ValueError, RuntimeError, ArithmeticError) as error:
        return AtlasCorridor((seed, seed), None, None, None, str(error))
    points = (left, right)
    s0, s1 = sorted((left.s, right.s))
    a0, a1 = sorted((left.alpha, right.alpha))
    ds, da = s1 - s0, a1 - a0
    if ds <= 1e-12 and da <= 1e-12:
        return AtlasCorridor(points, None, None, None, "corrected segment did not move")
    s_pad = max(ds * 0.1, 1e-8)
    a_pad = max(da * 0.1, 1e-8)
    s_interval = (max(0.0, s0 - s_pad), min(1.0, s1 + s_pad))
    a_interval = (a0 - a_pad, a1 + a_pad)
    if ds >= da / (2 * np.pi):
        split = (s0 + s1) / 2
        domains = (
            ((s_interval[0], split), a_interval),
            ((split, s_interval[1]), a_interval),
        )
    else:
        split = (a0 + a1) / 2
        domains = (
            (s_interval, (a_interval[0], split)),
            (s_interval, (split, a_interval[1])),
        )
    union = classify_cell(
        oracle.field, oracle.b, s_interval, a_interval, periods, subdivisions
    )
    cells = tuple(
        classify_cell(oracle.field, oracle.b, s, a, periods, subdivisions)
        for s, a in domains
    )
    if (
        union.multiplicity_upper is None
        or union.multiplicity_upper == 0
        or any(c.multiplicity_upper != union.multiplicity_upper for c in cells)
    ):
        return AtlasCorridor(
            points,
            None,
            union,
            None,
            union.unknown_reason
            or "; ".join(c.unknown_reason or "count mismatch" for c in cells),
        )

    # The sampled count guard is deliberately independent of the interval
    # proof; disagreements leave the whole proposed corridor unknown.
    for cell in (*cells, union):
        for s in cell.s_interval:
            for alpha in cell.alpha_interval:
                wells, reason = _owned(
                    _line(oracle.field, s, alpha, periods), oracle.b, False
                )
                if len(wells) != cell.multiplicity_upper:
                    return AtlasCorridor(
                        points,
                        None,
                        union,
                        None,
                        reason or "sampled root count disagrees with cell certificate",
                    )

    branch_ids = []
    for point in points:
        wells, reason = _owned(
            _line(oracle.field, point.s, point.alpha, periods), oracle.b, False
        )
        matches = [
            well.branch_id
            for well in wells
            if abs(well.zeta_in - point.zeta_in) <= 2e-6
            and abs(well.zeta_out - point.zeta_out) <= 2e-6
        ]
        if len(matches) != 1:
            return AtlasCorridor(
                points,
                None,
                union,
                None,
                reason or "continued roots lack unique atlas ownership",
            )
        branch_ids.append(matches[0])
    if branch_ids[0] != branch_ids[1]:
        return AtlasCorridor(
            points, None, union, None, "root ordinal changes across shared cell edge"
        )
    return AtlasCorridor(points, cells, union, branch_ids[0], None)


def embed_corridor_in_atlas(atlas: BranchAtlas, corridor: AtlasCorridor) -> BranchAtlas:
    """Refine the fixed-domain atlas by a certified corridor and its complement.

    Every overlap with an original cell becomes a disjoint adaptive descendant;
    alpha-periodic images are clipped to [0,2π]. Unknown complement is never
    subtracted from a weight estimate without a positive cell certificate (§8.4).
    """
    if not corridor.certified or corridor.union is None:
        raise ValueError("only a certified local corridor can refine the atlas")
    result = atlas
    patch = corridor.union
    original_cells = tuple(atlas.cells)
    for shift in (-2 * np.pi, 0.0, 2 * np.pi):
        a_patch = (patch.alpha_interval[0] + shift, patch.alpha_interval[1] + shift)
        if a_patch[1] <= 0 or a_patch[0] >= 2 * np.pi:
            continue
        for parent in original_cells:
            s0 = max(patch.s_interval[0], parent.s_interval[0])
            s1 = min(patch.s_interval[1], parent.s_interval[1])
            a0 = max(a_patch[0], parent.alpha_interval[0])
            a1 = min(a_patch[1], parent.alpha_interval[1])
            if s1 <= s0 or a1 <= a0:
                continue
            index = next(
                (
                    i
                    for i, cell in enumerate(result.cells)
                    if cell.s_interval == parent.s_interval
                    and cell.alpha_interval == parent.alpha_interval
                ),
                None,
            )
            if index is None:
                raise ArithmeticError("original atlas cell was refined twice")
            result = refine_atlas_cell(result, index, (s0, s1), (a0, a1))
    if not np.isclose(
        sum(cell.area for cell in result.cells),
        sum(cell.area for cell in atlas.cells),
        rtol=0,
        atol=1e-12,
    ):
        raise ArithmeticError("corridor refinement lost fixed-domain area")
    return result


def plot_atlas_corridor(corridor: AtlasCorridor):
    """Diagnostic of known/unknown local coverage and the direct crossing (§17)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(5, 4))
    if corridor.union is not None:
        cell = corridor.union
        ax.add_patch(
            Rectangle(
                (cell.s_interval[0], cell.alpha_interval[0]),
                cell.s_interval[1] - cell.s_interval[0],
                cell.alpha_interval[1] - cell.alpha_interval[0],
                facecolor="0.9" if corridor.certified else "#f4c5c5",
                edgecolor="black",
            )
        )
    if corridor.cells is not None:
        for cell in corridor.cells:
            ax.add_patch(
                Rectangle(
                    (cell.s_interval[0], cell.alpha_interval[0]),
                    cell.s_interval[1] - cell.s_interval[0],
                    cell.alpha_interval[1] - cell.alpha_interval[0],
                    facecolor="#a9d7bf",
                    edgecolor="black",
                    alpha=0.7,
                )
            )
    ax.plot([p.s for p in corridor.points], [p.alpha for p in corridor.points], "ko-")
    ax.set_xlabel("normalized flux s")
    ax.set_ylabel("lifted alpha (rad)")
    ax.set_title(
        "certified root corridor" if corridor.certified else "unknown root corridor"
    )
    ax.autoscale_view()
    return fig


def plot_path_atlas_corridor(corridor: PathAtlasCorridor):
    """Show the disjoint known path tiles against the unknown complement (§17)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_facecolor("#eeeeee")
    for cell in corridor.owned_cells:
        ax.add_patch(
            Rectangle(
                (cell.s_interval[0], cell.alpha_interval[0]),
                cell.s_interval[1] - cell.s_interval[0],
                cell.alpha_interval[1] - cell.alpha_interval[0],
                facecolor="#55aa7a",
                edgecolor="none",
                alpha=0.8,
            )
        )
    s = np.array([point.s for point in corridor.path.points])
    alpha = np.mod([point.alpha for point in corridor.path.points], 2 * np.pi)
    for index in range(len(s) - 1):
        if abs(alpha[index + 1] - alpha[index]) < np.pi:
            ax.plot(s[index : index + 2], alpha[index : index + 2], "k-", lw=1)
    ax.plot(s, alpha, "k.", ms=2)
    ax.set_xlim(max(0, min(s) - 0.03), min(1, max(s) + 0.03))
    ax.set_ylim(0, 2 * np.pi)
    ax.set_xlabel("normalized flux s")
    ax.set_ylabel("alpha mod 2π (rad)")
    ax.set_title(
        f"{corridor.status}: owned area {corridor.owned_area:.4g} of 2π; gray remains unknown"
    )
    return fig
