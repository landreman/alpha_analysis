from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from alpha_analysis.j_connectivity.background_mesh import (
    BackgroundMeshConfig,
    StructuredPrismMeshBackend,
)
from alpha_analysis.j_connectivity.surface_extract import (
    MarchingTetrahedraExtractor,
    SurfaceMesh,
    surface_flux,
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


def _component_fluxes(surface, field):
    result = []
    for component_id in np.unique(surface.component_ids):
        selected = surface.component_ids == component_id
        component = replace(
            surface,
            triangles=surface.triangles[selected],
            triangle_parent_tetrahedra=surface.triangle_parent_tetrahedra[selected],
            component_ids=np.zeros(np.count_nonzero(selected), dtype=np.int64),
        )
        result.append(surface_flux(component, field))
    return np.sort(result)


def test_downsampling_reduces_triangle_count_preserves_topology_and_projects():
    field = _field()
    background = StructuredPrismMeshBackend(
        BackgroundMeshConfig(n_radial=5, n_poloidal=12, n_zeta=6)
    ).build(field)
    original = MarchingTetrahedraExtractor().extract(background, field, b=0.04).full
    config = SurfaceDownsamplingConfig(target_reduction=0.5)

    result = downsample_surface(original, field, config)
    reduced = result.surface

    assert len(reduced.triangles) <= 0.51 * len(original.triangles)
    assert result.report.reached_target
    assert result.report.achieved_reduction >= 0.49
    assert _topology(reduced) == _topology(original) == [(0, 0), (0, 0)]
    s, theta, zeta = _coordinates(reduced.points)
    np.testing.assert_allclose(field.B(s, theta, zeta), 0.04, atol=1.0e-10)
    assert np.any(reduced.triangle_parent_tetrahedra == -1)
    assert np.any(reduced.triangle_parent_tetrahedra >= 0)
    original_flux = surface_flux(original, field)
    reduced_flux = surface_flux(reduced, field)
    assert (
        abs(reduced_flux - original_flux) / original_flux
        <= config.max_flux_relative_error
    )
    original_component_fluxes = _component_fluxes(original, field)
    reduced_component_fluxes = _component_fluxes(reduced, field)
    np.testing.assert_allclose(
        reduced_component_fluxes,
        original_component_fluxes,
        rtol=config.max_flux_relative_error,
    )

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
        BackgroundMeshConfig(n_radial=5, n_poloidal=12, n_zeta=6)
    ).build(field)
    original = MarchingTetrahedraExtractor().extract(background, field, b=0.04).incoming
    protected = original.boundary_tags != 0
    protected_records = {
        (tuple(point), int(tag))
        for point, tag in zip(
            original.points[protected], original.boundary_tags[protected]
        )
    }

    result = downsample_surface(
        original,
        field,
        SurfaceDownsamplingConfig(target_reduction=0.5),
    )
    reduced = result.surface
    reduced_records = {
        (tuple(point), int(tag))
        for point, tag in zip(reduced.points, reduced.boundary_tags)
        if tag != 0
    }

    assert len(reduced.triangles) < len(original.triangles)
    assert reduced_records == protected_records
    assert _topology(reduced) == _topology(original)
    assert np.all(reduced.g <= 1.0e-10)


class _GuardField:
    """Analytic periodic field that isolates sign and geometric guards."""

    nfp = 1

    def __init__(self, *, hostile_g: bool) -> None:
        self.hostile_g = hostile_g

    @staticmethod
    def _arrays(s, theta, zeta):
        return np.broadcast_arrays(
            np.asarray(s, dtype=float),
            np.asarray(theta, dtype=float),
            np.asarray(zeta, dtype=float),
        )

    def B(self, s, theta, zeta):
        _, theta, zeta = self._arrays(s, theta, zeta)
        result = 2.0 + np.sin(zeta)
        if self.hostile_g:
            result += 0.8 * np.sin(4.0 * theta)
        return result

    def dB_ds(self, s, theta, zeta):
        s, _, _ = self._arrays(s, theta, zeta)
        return np.zeros_like(s)

    def dB_dtheta(self, s, theta, zeta):
        s, theta, _ = self._arrays(s, theta, zeta)
        if self.hostile_g:
            return 3.2 * np.cos(4.0 * theta)
        return np.zeros_like(s)

    def dB_dzeta(self, s, theta, zeta):
        _, _, zeta = self._arrays(s, theta, zeta)
        return np.cos(zeta)

    def D_B(self, s, theta, zeta):
        return self.dB_dtheta(s, theta, zeta) + self.dB_dzeta(s, theta, zeta)

    def D2_B(self, s, theta, zeta):
        _, theta, zeta = self._arrays(s, theta, zeta)
        result = -np.sin(zeta)
        if self.hostile_g:
            result -= 12.8 * np.sin(4.0 * theta)
        return result

    def iota(self, s):
        return np.full_like(np.asarray(s, dtype=float), float(self.hostile_g))

    def G(self, s):
        return np.ones_like(np.asarray(s, dtype=float))

    def I(self, s):
        return np.zeros_like(np.asarray(s, dtype=float))

    def C(self, s):
        return np.ones_like(np.asarray(s, dtype=float))


def _guard_surface(field):
    # Four tagged boundary vertices surround two interior vertices. The only
    # collapsible edge is 4--5; its link is the two triangles 1 and 4.
    points = np.array(
        [
            [-0.6, 0.6, 0.0],
            [0.6, 0.6, 0.0],
            [0.6, -0.6, 0.0],
            [-0.6, -0.6, 0.0],
            [-0.1, 0.0, 0.0],
            [0.1, 0.0, 0.0],
        ]
    )
    if field.hostile_g:
        angles = np.array([np.pi / 6.0, -np.pi / 6.0])
        points[4:, 0] = 0.1 * np.cos(angles)
        points[4:, 1] = 0.1 * np.sin(angles)
        points[4:, 2] = np.arcsin(-0.8 * np.sin(4.0 * angles))
    triangles = np.array(
        [[0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 3, 5], [3, 4, 5], [3, 0, 4]]
    )
    s, theta, zeta = _coordinates(points)
    B = field.B(s, theta, zeta)
    g = B * field.D_B(s, theta, zeta)
    return SurfaceMesh(
        level=2.0,
        period=2.0 * np.pi,
        points=points,
        triangles=triangles,
        B=B,
        g=g,
        boundary_tags=np.array([SurfaceMesh.EDGE] * 4 + [0, 0]),
        point_parent_edges=np.full((6, 2), -1, dtype=np.int64),
        triangle_parent_tetrahedra=np.full(6, -1, dtype=np.int64),
        component_ids=np.zeros(6, dtype=np.int64),
    )


def _permissive_guard_config():
    return SurfaceDownsamplingConfig(
        target_reduction=0.2,
        max_flux_relative_error=10.0,
        max_normal_deviation_degrees=89.0,
        min_triangle_quality=1.0e-6,
    )


def test_downsampling_refuses_a_collapse_that_crosses_g_zero():
    field = _GuardField(hostile_g=True)
    original = _guard_surface(field)
    assert original.g[4] < 0.0 and original.g[5] < 0.0

    result = downsample_surface(original, field, _permissive_guard_config())
    reduced = result.surface

    # The midpoint has g>0, so accepting edge 4--5 would remove two faces.
    assert float(field.D_B(0.0, 0.0, 0.0)) > 0.0
    assert len(reduced.triangles) == len(original.triangles)
    assert not result.report.reached_target
    assert result.report.g_sign_rejections > 0


def test_downsampling_refuses_a_collapse_that_inverts_a_face(monkeypatch):
    import alpha_analysis.j_connectivity.surface_refine as module

    field = _GuardField(hostile_g=False)
    original = _guard_surface(field)
    # Moving vertex 4 above edge 0--1 reverses face 0's normal. This isolates
    # the production-path face guard from midpoint and projection details.
    monkeypatch.setattr(
        module,
        "_project_to_level_near",
        lambda *_args, **_kwargs: np.array([0.6, 0.7, 0.0]),
    )

    result = downsample_surface(original, field, _permissive_guard_config())
    reduced = result.surface

    assert len(reduced.triangles) == len(original.triangles)
    assert result.report.face_validity_rejections > 0


def test_downsampling_is_a_no_op_for_empty_surface():
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

    result = downsample_surface(empty, field)

    assert result.surface is empty
    assert result.report.reached_target
    assert result.report.achieved_reduction == 0.0


def test_downsampling_is_a_no_op_for_zero_reduction():
    field = _GuardField(hostile_g=False)
    original = _guard_surface(field)

    result = downsample_surface(
        original,
        field,
        SurfaceDownsamplingConfig(target_reduction=0.0),
    )

    assert result.surface is original
    assert result.report.input_triangle_count == len(original.triangles)
    assert result.report.target_triangle_count == len(original.triangles)
    assert result.report.output_triangle_count == len(original.triangles)
    assert result.report.reached_target
    assert result.report.achieved_reduction == 0.0
