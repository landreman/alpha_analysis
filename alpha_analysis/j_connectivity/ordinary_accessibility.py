"""Finite ordinary action-contour bounds on an explicit PL atlas (DESIGN §11).

Triangle/vertex IDs encode root-family identity, including any *certified* seam
identifications made by the caller. Coordinates and equal actions never create
adjacency. No transitions are represented. Unknown exterior incidence is an
upper-only EDGE wildcard; it is never a sealed artificial boundary.

The exact, zero-error input defines a finite PL model, not the continuous field.
Nonzero uniform action envelopes need an independent level-topology certificate
for lower propagation; a scalar error bound alone cannot provide it. NaN action or uncertified cells
retain their entire area and possible ordinary influence. Field-level users must
supply complete root-labelled coverage and genuine uniform error enclosures.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class OrdinaryMesh:
    """Owned (s, alpha radians) triangles with half-bounce action A in length units.

    Shared IDs mean certified ordinary continuation, never coordinate matching.
    ``edge_vertices`` identify the physical EDGE subcomplex (all vertices of an
    edge must be marked to seed its full action image). ``frontier_vertices``
    similarly mark possible incidence with omitted geometry connected to EDGE.
    Any omitted branch that might meet a cell must be represented there.

    ``action_error`` bounds |A_true-A_PL| *throughout* each triangle, including
    its boundary; estimates from quadrature or mesh comparisons are insufficient.
    ``certified=False`` means unknown root/geometry, with unrestricted possible
    action and no lower paths. Its unproved incidence invokes a global upper
    EDGE wildcard, since omitted connections cannot be localized. The unweighted
    area remains accounted for.
    ``level_connected`` certifies that the continuous true action has a
    path-connected level set in the entire triangle for every attained value.
    It requires an independent proof, for example a validated regular graph
    chart. A small interpolation residual is insufficient. Without it, positive
    errors give no lower paths. Exact PL triangles need no extra certificate.
    """

    points: np.ndarray
    triangles: np.ndarray
    action: np.ndarray
    edge_vertices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    frontier_vertices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    action_error: np.ndarray | None = None
    certified: np.ndarray | None = None
    level_connected: np.ndarray | None = None

    def __post_init__(self):
        for name, dtype in (
            ("points", float),
            ("triangles", int),
            ("action", float),
            ("edge_vertices", int),
            ("frontier_vertices", int),
        ):
            raw = np.asarray(getattr(self, name))
            if dtype is int and (not np.issubdtype(raw.dtype, np.integer)) and raw.size:
                raise ValueError(f"{name} must contain integer IDs")
            value = np.array(raw, dtype=dtype, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        p, t, a = self.points, self.triangles, self.action
        if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
            raise ValueError("points must be finite (s, alpha) pairs")
        if (
            t.ndim != 2
            or t.shape[1] != 3
            or not len(t)
            or np.any(t < 0)
            or np.any(t >= len(p))
        ):
            raise ValueError("triangles must contain three valid vertex IDs")
        if a.shape != (len(p),) or np.isinf(a).any():
            raise ValueError(
                "action must contain finite values or explicit NaN unknowns"
            )
        for ids in (self.edge_vertices, self.frontier_vertices):
            if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(p)):
                raise ValueError("boundary vertex IDs are invalid")
        if np.any(self.areas <= 0):
            raise ValueError("triangles must have positive area")
        for name, default, dtype in (
            ("action_error", 0.0, float),
            ("certified", True, bool),
            ("level_connected", False, bool),
        ):
            value = getattr(self, name)
            if (
                name != "action_error"
                and value is not None
                and np.asarray(value).dtype != np.dtype(bool)
            ):
                raise ValueError(f"{name} must contain Boolean certificates")
            value = (
                np.full(len(t), default, dtype=dtype)
                if value is None
                else np.array(value, dtype=dtype, copy=True)
            )
            if value.shape != (len(t),):
                raise ValueError(f"{name} must have one entry per triangle")
            if name == "action_error" and (
                not np.isfinite(value).all() or np.any(value < 0)
            ):
                raise ValueError("uniform action errors must be finite and nonnegative")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def areas(self):
        """Unweighted |ds d-alpha|, not hK population (§8.3)."""
        q = self.points[self.triangles]
        u, v = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]
        return abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / 2


@dataclass(frozen=True)
class OrdinaryBounds:
    """Finite bits per triangle/atom; intervals are open and points are separate.

    A point atom preserves finite-area flat cells and exact saddle incidence.
    No epsilon perturbation is used. ``bit_changes`` counts worklist additions,
    at most twice triangle_count*atom_count. Scope is always explicit.
    """

    atoms: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    bit_changes: int
    unknown_triangles: tuple[int, ...]
    bound_scope: str = (
        "finite PL action model with declared uniform envelopes; unweighted"
    )

    def classify(self, triangle: int, action: float) -> str:
        """Classify a known action in a specified local branch/triangle (§11.2)."""
        if not np.isfinite(action):
            return "UNKNOWN"
        matches = np.flatnonzero(
            ((self.atoms[:, 0] == self.atoms[:, 1]) & (self.atoms[:, 0] == action))
            | ((self.atoms[:, 0] < action) & (action < self.atoms[:, 1]))
        )
        if len(matches) != 1:
            raise ValueError("query action lies outside the covered action partition")
        j = matches[0]
        return (
            "ACCESSIBLE"
            if self.lower[triangle, j]
            else ("UNKNOWN" if self.upper[triangle, j] else "INACCESSIBLE")
        )


def _contains(atoms, lo, hi):
    return (lo <= atoms[:, 0]) & (atoms[:, 1] <= hi)


def _intersects(atoms, lo, hi):
    point = atoms[:, 0] == atoms[:, 1]
    return np.where(
        point,
        (lo <= atoms[:, 0]) & (atoms[:, 0] <= hi),
        (lo < atoms[:, 1]) & (atoms[:, 0] < hi),
    )


def _action_image(values, error, inner):
    """Round error-band endpoints inward/outward, preserving exact PL ties."""
    low, high = min(values), max(values)
    if error == 0:
        return low, high
    if inner:
        return np.nextafter(low + error, np.inf), np.nextafter(high - error, -np.inf)
    return np.nextafter(low - error, -np.inf), np.nextafter(high + error, np.inf)


def ordinary_bounds(mesh: OrdinaryMesh, bin_edges) -> OrdinaryBounds:
    """Enclose ordinary reachability by finite inner/outer worklists (§11.2).

    Inner paths transfer the *entire* atom through each proven interface. Outer
    paths transfer any intersecting atom; apparent action drift is confined to
    that atom and disappears under nested refinement on fixed geometry. Flat
    triangle actions split the partition into singleton atoms automatically.
    All shared simplices, including vertex-only saddle contacts, participate.
    """
    edges = np.asarray(bin_edges, dtype=float)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("bin edges must be finite and strictly increasing")
    values = mesh.action[mesh.triangles]
    known = np.isfinite(values).all(axis=1) & mesh.certified
    lo = np.full(len(values), -np.inf)
    hi = np.full(len(values), np.inf)
    for i in np.flatnonzero(known):
        lo[i], hi[i] = _action_image(values[i], mesh.action_error[i], False)
    if np.any(lo[known] < edges[0]) or np.any(hi[known] > edges[-1]):
        raise ValueError("bins must cover all known action envelopes")
    flat = known & (np.ptp(values, axis=1) == 0)
    edges = np.unique(np.r_[edges, values[flat, 0]])
    atoms = np.array([(x, x) for x in edges] + list(zip(edges[:-1], edges[1:])))
    atoms = atoms[np.lexsort((atoms[:, 1], atoms[:, 0]))]
    inner = known & ((mesh.action_error == 0) | mesh.level_connected)
    active_upper = np.array([_intersects(atoms, a, b) for a, b in zip(lo, hi)])
    active_lower = np.array(
        [
            (
                _contains(atoms, *_action_image(v, error, True))
                if ok
                else np.zeros(len(atoms), bool)
            )
            for v, ok, error in zip(values, inner, mesh.action_error)
        ]
    )

    # Vertex stars provide candidate spatial adjacency. Their shared simplex
    # determines the action image; equal action in distinct stars creates no link.
    stars = [[] for _ in mesh.points]
    for i, ids in enumerate(mesh.triangles):
        for v in ids:
            stars[v].append(i)
    pairs = set(pair for star in stars for pair in combinations(star, 2))
    adjacency = [[] for _ in mesh.triangles]
    for first, second in sorted(pairs):
        shared = np.intersect1d(mesh.triangles[first], mesh.triangles[second])
        v = mesh.action[shared]
        error = max(mesh.action_error[first], mesh.action_error[second])
        lower = (
            _contains(atoms, *_action_image(v, error, True))
            if inner[first] and inner[second]
            else np.zeros(len(atoms), bool)
        )
        upper = (
            _intersects(atoms, *_action_image(v, error, False))
            if known[first] and known[second]
            else np.ones(len(atoms), bool)
        )
        adjacency[first].append((second, lower, upper))
        adjacency[second].append((first, lower, upper))

    lower = np.zeros_like(active_lower)
    upper = np.zeros_like(active_upper)
    edge_ids, frontier_ids = set(mesh.edge_vertices), set(mesh.frontier_vertices)
    for i, ids in enumerate(mesh.triangles):
        for boundary, possible_only in ((edge_ids, False), (frontier_ids, True)):
            selected = [v for v in ids if v in boundary]
            if not selected:
                continue
            v = mesh.action[selected]
            if known[i]:
                upper[i] |= _intersects(
                    atoms, *_action_image(v, mesh.action_error[i], False)
                )
            else:
                upper[i] = True
            if inner[i] and not possible_only:
                lower[i] |= _contains(
                    atoms, *_action_image(v, mesh.action_error[i], True)
                )

    # No incidence certificate accompanies an explicitly unknown geometry cell.
    # A global wildcard is deliberately loose: missing branches may connect it
    # to any represented state and EDGE (§11.3), irrespective of local area.
    if not mesh.certified.all():
        upper |= active_upper

    changes = 0
    for bits, active, side in ((lower, active_lower, 1), (upper, active_upper, 2)):
        bits &= active
        queue = deque(map(tuple, np.argwhere(bits)))
        changes += len(queue)
        while queue:
            cell, atom = queue.popleft()
            for link in adjacency[cell]:
                neighbor = link[0]
                if (
                    link[side][atom]
                    and active[neighbor, atom]
                    and not bits[neighbor, atom]
                ):
                    bits[neighbor, atom] = True
                    changes += 1
                    queue.append((neighbor, atom))
    if np.any(lower & ~upper):
        raise ArithmeticError("inner reachability escaped outer reachability")
    for array in (atoms, lower, upper):
        array.setflags(write=False)
    return OrdinaryBounds(atoms, lower, upper, changes, tuple(np.flatnonzero(~known)))


def _clip(poly, threshold, above):
    result = []
    for first, last in zip(poly, poly[1:] + poly[:1]):
        f, g = (
            (first[2] >= threshold, last[2] >= threshold)
            if above
            else (first[2] <= threshold, last[2] <= threshold)
        )
        if f:
            result.append(first)
        if f != g:
            result.append(
                first + (last - first) * ((threshold - first[2]) / (last[2] - first[2]))
            )
    return result


def reachable_area(mesh: OrdinaryMesh, bounds: OrdinaryBounds) -> tuple[float, float]:
    """Integrate disjoint reachable PL polygons in |ds d-alpha| (§12.3).

    This is geometric area only, never K-weight or f. Unknown triangles retain
    full possible area. Uniform action envelopes erode inner and expand outer preimages;
    overlapping expanded intervals are merged before polygon integration.
    """
    totals = []
    for bits, possible in ((bounds.lower, False), (bounds.upper, True)):
        total = 0.0
        for i, ids in enumerate(mesh.triangles):
            indices = np.flatnonzero(bits[i])
            selected = bounds.atoms[indices]
            if not possible and len(indices):
                # Merge only consecutive reached atoms, including their point
                # atoms, before eroding. This preserves nesting with error bands
                # without silently filling an unreached critical singleton.
                runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
                selected = np.array(
                    [
                        (bounds.atoms[run[0], 0], bounds.atoms[run[-1], 1])
                        for run in runs
                    ]
                )
            if not len(selected):
                continue
            a = mesh.action[ids]
            if not mesh.certified[i] or not np.isfinite(a).all():
                if possible:
                    total += mesh.areas[i]
                continue
            if np.ptp(a) == 0:
                error = mesh.action_error[i] if possible else 0
                if any(low - error <= a[0] <= high + error for low, high in selected):
                    total += mesh.areas[i]
                continue
            error = mesh.action_error[i]
            intervals = []
            for low, high in selected:
                low, high = (
                    (low - error, high + error)
                    if possible
                    else (low + error, high - error)
                )
                if error:
                    low = np.nextafter(low, -np.inf if possible else np.inf)
                    high = np.nextafter(high, np.inf if possible else -np.inf)
                if high < low:
                    continue
                if intervals and low <= intervals[-1][1] and (possible or error == 0):
                    intervals[-1][1] = max(intervals[-1][1], high)
                else:
                    intervals.append([low, high])
            for low, high in intervals:
                if high == low:
                    continue
                poly = [row for row in np.c_[mesh.points[ids], a]]
                poly = _clip(_clip(poly, low, True), high, False)
                if len(poly) >= 3:
                    p = np.array(poly)
                    p[:, :2] -= p[0, :2].copy()
                    total += (
                        abs(
                            np.dot(p[:, 0], np.roll(p[:, 1], -1))
                            - np.dot(p[:, 1], np.roll(p[:, 0], -1))
                        )
                        / 2
                    )
        totals.append(float(total))
    return tuple(totals)


def plot_ordinary_bounds(
    mesh: OrdinaryMesh, bounds: OrdinaryBounds, *, title="ordinary PL accessibility"
):
    """Plot inner/outer geometric fractions, separately from field bounds (§17)."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, side in zip(axes, ("lower", "upper")):
        fractions = []
        for i, ids in enumerate(mesh.triangles):
            local = OrdinaryMesh(
                mesh.points,
                ids[None, :],
                mesh.action,
                action_error=mesh.action_error[i : i + 1],
                certified=mesh.certified[i : i + 1],
                level_connected=mesh.level_connected[i : i + 1],
            )
            local_bounds = OrdinaryBounds(
                bounds.atoms, bounds.lower[i : i + 1], bounds.upper[i : i + 1], 0, ()
            )
            fractions.append(
                reachable_area(local, local_bounds)[side == "upper"] / mesh.areas[i]
            )
        collection = PolyCollection(
            mesh.points[mesh.triangles],
            array=np.array(fractions),
            cmap="viridis",
            clim=(0, 1),
            edgecolors="gray",
            linewidths=0.3,
        )
        ax.add_collection(collection)
        ax.autoscale()
        ax.set(
            xlabel="normalized flux s",
            ylabel="alpha (rad)",
            title=f"{side}: geometric area fraction",
        )
        fig.colorbar(collection, ax=ax)
    fig.suptitle(title + "\nfinite PL model; no K weight or physical field enclosure")
    return fig
