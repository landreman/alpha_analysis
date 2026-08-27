from __future__ import annotations

from collections import Counter

import numpy as np

from alpha_analysis.j_connectivity.background_mesh import (
    BackgroundMeshConfig,
    StructuredPrismMeshBackend,
)
from alpha_analysis.j_connectivity.surface_extract import (
    MarchingTetrahedraExtractor,
    SurfaceMesh,
)
from alpha_analysis.j_connectivity.surface_refine import (
    SurfaceDownsamplingConfig,
    downsample_surface,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField


def _field():
    # B=(s-1/2)^2 + 0.01 cos(3 zeta).  B=0.04 consists of two closed tori.
    return SyntheticFourierField(
        nfp=3,
        m=np.array([0, 0]),
        n=np.array([0, 3]),
        cosine_coefficients=np.array([[0.25, -1.0, 1.0], [0.01, 0.0, 0.0]]),
        sine_coefficients=np.zeros((2, 3)),
        iota_coefficients=np.array([0.7, 0.25]),
        G_coefficients=np.array([1.0]),
        I_coefficients=np.array([0.0]),
    )


def _coordinates(points):
    s = np.sum(points[:, :2] ** 2, axis=1)
    theta = np.arctan2(points[:, 1], points[:, 0])
    theta[s == 0.0] = 0.0
    return s, theta, points[:, 2]


def _topology(surface):
    result = []
    for component_id in np.unique(surface.component_ids):
        triangles = surface.triangles[surface.component_ids == component_id]
        vertices = np.unique(triangles)
        edge_counts = Counter(
            tuple(sorted(map(int, edge)))
            for triangle in triangles
            for edge in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            )
        )
        result.append(
            (
                len(vertices) - len(edge_counts) + len(triangles),
                sum(count == 1 for count in edge_counts.values()),
            )
        )
    return sorted(result)


def _edge_lengths(surface):
    vertices = surface.points[surface.triangles].copy()
    for index in (1, 2):
        difference = vertices[:, index, 2] - vertices[:, 0, 2]
        vertices[:, index, 2] -= surface.period * np.round(difference / surface.period)
    return np.linalg.norm(
        np.concatenate(
            (
                vertices[:, 1] - vertices[:, 0],
                vertices[:, 2] - vertices[:, 1],
                vertices[:, 0] - vertices[:, 2],
            )
        ),
        axis=1,
    )


def test_downsampling_reduces_triangle_count_preserves_topology_and_projects():
    field = _field()
    background = StructuredPrismMeshBackend(
        BackgroundMeshConfig(n_radial=12, n_poloidal=32, n_zeta=24)
    ).build(field)
    original = MarchingTetrahedraExtractor().extract(background, field, b=0.04).full

    reduced = downsample_surface(
        original,
        field,
        SurfaceDownsamplingConfig(target_reduction=0.65),
    )

    assert len(reduced.triangles) < 0.6 * len(original.triangles)
    assert _topology(reduced) == _topology(original) == [(0, 0), (0, 0)]
    s, theta, zeta = _coordinates(reduced.points)
    np.testing.assert_allclose(field.B(s, theta, zeta), 0.04, atol=1.0e-10)
    assert np.any(reduced.triangle_parent_tetrahedra == -1)
    assert np.any(reduced.triangle_parent_tetrahedra >= 0)

    # Shortest-edge collapse removes the extreme small-edge tail instead of
    # merely selecting a random subset of the original triangles.
    original_lengths = _edge_lengths(original)
    reduced_lengths = _edge_lengths(reduced)
    assert np.quantile(reduced_lengths, 0.1) > np.quantile(original_lengths, 0.1)
    assert np.quantile(reduced_lengths, 0.9) / np.quantile(
        reduced_lengths, 0.1
    ) < np.quantile(original_lengths, 0.9) / np.quantile(original_lengths, 0.1)


def test_downsampling_keeps_incoming_boundary_vertices_and_sign():
    field = _field()
    background = StructuredPrismMeshBackend(
        BackgroundMeshConfig(n_radial=10, n_poloidal=28, n_zeta=20)
    ).build(field)
    original = MarchingTetrahedraExtractor().extract(background, field, b=0.04).incoming
    protected = original.boundary_tags != 0
    protected_records = {
        (tuple(point), int(tag))
        for point, tag in zip(
            original.points[protected], original.boundary_tags[protected]
        )
    }

    reduced = downsample_surface(
        original,
        field,
        SurfaceDownsamplingConfig(target_reduction=0.5),
    )
    reduced_records = {
        (tuple(point), int(tag))
        for point, tag in zip(reduced.points, reduced.boundary_tags)
        if tag != 0
    }

    assert len(reduced.triangles) < len(original.triangles)
    assert reduced_records == protected_records
    assert _topology(reduced) == _topology(original)
    assert np.all(reduced.g <= 1.0e-10)


def test_downsampling_is_a_no_op_for_zero_reduction():
    field = _field()
    empty = SurfaceMesh(
        level=0.04,
        period=2.0 * np.pi / field.nfp,
        points=np.empty((0, 3)),
        triangles=np.empty((0, 3), dtype=np.int64),
        B=np.empty(0),
        g=np.empty(0),
        boundary_tags=np.empty(0, dtype=np.int64),
        point_parent_edges=np.empty((0, 2), dtype=np.int64),
        triangle_parent_tetrahedra=np.empty(0, dtype=np.int64),
        component_ids=np.empty(0, dtype=np.int64),
    )

    assert downsample_surface(empty, field) is empty
