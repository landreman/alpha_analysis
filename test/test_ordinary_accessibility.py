"""R4 action-preserving reachability, with analytic level-set expectations."""

import numpy as np
import pytest

from alpha_analysis.j_connectivity.ordinary_accessibility import (
    OrdinaryMesh,
    ordinary_bounds,
    reachable_area,
)


def grid(action, ns=5, na=7, *, open_boundary=False):
    points = np.array(
        [(s, a) for s in np.linspace(0.1, 1, ns) for a in np.linspace(0, 1, na)]
    )
    triangles = []
    for i in range(ns - 1):
        for j in range(na - 1):
            k = i * na + j
            triangles.extend(((k, k + na, k + na + 1), (k, k + na + 1, k + 1)))
    edge = np.flatnonzero(points[:, 0] == 1)
    frontier = (
        np.unique(
            np.concatenate(
                (
                    np.arange(na),
                    np.arange(0, ns * na, na),
                    np.arange(na - 1, ns * na, na),
                )
            )
        )
        if open_boundary
        else np.array([], dtype=int)
    )
    return OrdinaryMesh(
        points,
        np.array(triangles),
        action(points[:, 0], points[:, 1]),
        edge_vertices=edge,
        frontier_vertices=frontier,
    )


def test_connected_reeb_interval_is_not_radial_escape():
    mesh = grid(lambda s, a: s)
    result = ordinary_bounds(mesh, np.linspace(0, 1.1, 12))
    assert result.classify(0, 0.2) == "INACCESSIBLE"
    assert reachable_area(mesh, result) == (0.0, 0.0)
    assert result.bit_changes <= 2 * result.lower.size


def test_disconnected_equal_action_branches_do_not_connect():
    # Coordinate-identical sheets deliberately have different vertex IDs.
    first = grid(lambda s, a: a, ns=2, na=2)
    mesh = OrdinaryMesh(
        np.tile(first.points, (2, 1)),
        np.vstack((first.triangles, first.triangles + 4)),
        np.tile(first.action, 2),
        edge_vertices=first.edge_vertices,
    )
    result = ordinary_bounds(mesh, [0, 0.5, 1])
    assert result.classify(0, 0.25) == "ACCESSIBLE"
    assert result.classify(2, 0.25) == "INACCESSIBLE"
    np.testing.assert_allclose(reachable_area(mesh, result), (0.9, 0.9))


def test_ordinary_bounds_bracket_direct_contours():
    # A=s+alpha: the straight level meets s=1 iff A >= 1.
    mesh = grid(lambda s, a: s + a)
    result = ordinary_bounds(mesh, np.linspace(0, 2, 18))
    for tri, vertices in enumerate(mesh.triangles):
        for weights in ((0.2, 0.3, 0.5), (0.6, 0.2, 0.2)):
            A = np.dot(weights, mesh.action[vertices])
            status = result.classify(tri, A)
            if A >= 1:
                assert status != "INACCESSIBLE"
            else:
                assert status != "ACCESSIBLE"
    low, high = reachable_area(mesh, result)
    assert low <= (1 - 0.1**2) / 2 <= high


def test_nested_action_bins_tighten_fixed_geometry_bounds():
    mesh = grid(lambda s, a: s + a)
    coarse = ordinary_bounds(mesh, np.arange(9) * 0.3)
    fine = ordinary_bounds(mesh, np.arange(33) * 0.075)
    for tri, vertices in enumerate(mesh.triangles):
        for A in np.linspace(
            min(mesh.action[vertices]), max(mesh.action[vertices]), 21
        ):
            before, after = coarse.classify(tri, A), fine.classify(tri, A)
            assert before != "ACCESSIBLE" or after == "ACCESSIBLE"
            assert before != "INACCESSIBLE" or after == "INACCESSIBLE"
    c, f = reachable_area(mesh, coarse), reachable_area(mesh, fine)
    assert c[0] < f[0] <= (1 - 0.1**2) / 2 <= f[1] < c[1]
    assert f[1] - f[0] < c[1] - c[0]


def test_flat_action_and_saddle_cases_preserve_measure():
    flat = grid(lambda s, a: np.ones_like(s))
    result = ordinary_bounds(flat, [0, 2])
    np.testing.assert_allclose(reachable_area(flat, result), (0.9, 0.9))
    assert result.classify(0, 1) == "ACCESSIBLE"
    # Four sectors meeting at a saddle: A=2 has a point junction at vertex 0.
    points = np.array([[0.5, 0.5], [1, 0], [1, 1], [0, 1], [0, 0]])
    saddle = OrdinaryMesh(
        points,
        np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]]),
        np.array([2, 3, 1, 3, 1]),
        edge_vertices=np.array([1, 2]),
    )
    bounds = ordinary_bounds(saddle, [0, 2, 4])
    assert all(bounds.classify(t, 2) == "ACCESSIBLE" for t in range(4))
    np.testing.assert_allclose(reachable_area(saddle, bounds), (0, 0.5))
    refined = ordinary_bounds(saddle, [1, 2, 3])
    np.testing.assert_allclose(reachable_area(saddle, refined), (0.5, 0.5))


def test_unknown_data_retains_area_and_remote_influence():
    mesh = grid(lambda s, a: a, open_boundary=True)
    # No physical EDGE: a missing ordinary corridor can still reach it.
    mesh = OrdinaryMesh(
        mesh.points,
        mesh.triangles,
        mesh.action,
        frontier_vertices=mesh.frontier_vertices,
    )
    bounds = ordinary_bounds(mesh, [0, 0.5, 1])
    assert not bounds.lower.any()
    assert all(
        bounds.classify(i, np.mean(mesh.action[ids])) == "UNKNOWN"
        for i, ids in enumerate(mesh.triangles)
    )
    np.testing.assert_allclose(reachable_area(mesh, bounds), (0, 0.9))
    unknown = OrdinaryMesh(
        mesh.points,
        mesh.triangles,
        np.full(len(mesh.points), np.nan),
        frontier_vertices=mesh.frontier_vertices,
    )
    result = ordinary_bounds(unknown, [0, 1])
    np.testing.assert_allclose(reachable_area(unknown, result), (0, 0.9))
    assert result.unknown_triangles == tuple(range(len(mesh.triangles)))


def test_action_error_does_not_become_a_certified_path():
    mesh = grid(lambda s, a: a)
    uncertain = OrdinaryMesh(
        mesh.points,
        mesh.triangles,
        mesh.action,
        edge_vertices=mesh.edge_vertices,
        action_error=np.full(len(mesh.triangles), 0.1),
    )
    result = ordinary_bounds(uncertain, [-1, 0, 0.5, 1, 2])
    assert not result.lower.any()
    np.testing.assert_allclose(reachable_area(uncertain, result), (0, 0.9))
    with pytest.raises(ValueError, match="cover"):
        ordinary_bounds(uncertain, [0, 1])


def test_certified_level_topology_and_uniform_errors_bracket_affine_field():
    """A_true=1.01s+alpha has connected straight levels and |A_true-A_PL|<=.02."""
    mesh = grid(lambda s, a: s + a)
    enclosed = OrdinaryMesh(
        mesh.points,
        mesh.triangles,
        mesh.action,
        edge_vertices=mesh.edge_vertices,
        action_error=np.full(len(mesh.triangles), 0.02),
        level_connected=np.ones(len(mesh.triangles), bool),
    )
    coarse = ordinary_bounds(enclosed, np.linspace(0, 2.4, 25))
    fine = ordinary_bounds(enclosed, np.linspace(0, 2.4, 49))
    area = 0.9 - 1.01 * 0.9**2 / 2
    low, high = reachable_area(enclosed, coarse)
    low2, high2 = reachable_area(enclosed, fine)
    assert 0 < low <= low2 <= area <= high2 <= high
    # Challenge the two narrow action bands an omitted error could misclassify:
    # below the true EDGE minimum, and above the nominal EDGE maximum.
    for q, accessible in (
        (np.array([0.99, 0.0051]), False),
        (np.array([1.0, 0.995]), True),
    ):
        found = False
        action = 1.01 * q[0] + q[1]
        for tri, ids in enumerate(mesh.triangles):
            weights = np.linalg.solve(
                np.vstack((mesh.points[ids].T, np.ones(3))), np.r_[q, 1]
            )
            if min(weights) >= -1e-10:
                found = True
                assert fine.classify(tri, action) != (
                    "INACCESSIBLE" if accessible else "ACCESSIBLE"
                )
        assert found
    for tri, ids in enumerate(mesh.triangles):
        q = mesh.points[ids].mean(axis=0)
        action = 1.01 * q[0] + q[1]
        if action >= 1.01:
            assert fine.classify(tri, action) != "INACCESSIBLE"
        else:
            assert fine.classify(tri, action) != "ACCESSIBLE"


def test_unknown_geometry_never_assumes_missing_incidence_is_closed():
    """Without an incidence proof, one unknown cell can affect distant states."""
    mesh = grid(lambda s, a: a)
    certified = np.ones(len(mesh.triangles), dtype=bool)
    certified[0] = False
    unknown = OrdinaryMesh(
        mesh.points, mesh.triangles, mesh.action, certified=certified
    )
    result = ordinary_bounds(unknown, [0, 0.5, 1])
    assert not result.lower.any()
    np.testing.assert_allclose(reachable_area(unknown, result), (0, 0.9))
    assert result.classify(len(mesh.triangles) - 1, 0.95) == "UNKNOWN"
