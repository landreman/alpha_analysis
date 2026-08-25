from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from alpha_analysis.j_connectivity.background_mesh import (
    BackgroundMeshConfig,
    StructuredPrismMeshBackend,
)
from alpha_analysis.j_connectivity.surface_extract import (
    MarchingTetrahedraExtractor,
    PyVistaSurfaceExtractor,
    SurfaceExtractionError,
    SurfaceMesh,
    surface_flux,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField
from alpha_analysis.j_connectivity.types import SurfaceStatus
from alpha_analysis.j_connectivity.visualization import plot_pitch_surface


def _field(
    *,
    radial_coefficients,
    toroidal_cosine,
    nfp=3,
    current_sign=1.0,
    helical_sine=0.0,
):
    radial_coefficients = np.asarray(radial_coefficients, dtype=float)
    width = len(radial_coefficients)
    m = [0, 0]
    n = [0, nfp]
    cosine = [
        radial_coefficients,
        np.pad([toroidal_cosine], (0, width - 1)),
    ]
    sine = [np.zeros(width), np.zeros(width)]
    if helical_sine:
        if width < 2:
            raise ValueError("helical_sine requires a radial polynomial of degree one")
        m.append(1)
        n.append(nfp)
        cosine.append(np.zeros(width))
        sine.append(np.pad([0.0, helical_sine], (0, width - 2)))
    return SyntheticFourierField(
        nfp=nfp,
        m=np.array(m),
        n=np.array(n),
        cosine_coefficients=np.array(cosine),
        sine_coefficients=np.array(sine),
        iota_coefficients=np.array([0.7]),
        G_coefficients=np.array([current_sign]),
        I_coefficients=np.array([0.0]),
    )


def _coordinates(points):
    s = np.sum(points[:, :2] ** 2, axis=1)
    theta = np.arctan2(points[:, 1], points[:, 0])
    theta[s == 0.0] = 0.0
    return s, theta, points[:, 2]


def _component_euler_characteristics(surface):
    characteristics = []
    for component in np.unique(surface.component_ids):
        triangles = surface.triangles[surface.component_ids == component]
        vertices = np.unique(triangles)
        edges = {
            tuple(sorted(edge))
            for triangle in triangles
            for edge in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            )
        }
        characteristics.append(len(vertices) - len(edges) + len(triangles))
    return characteristics


def test_two_closed_level_surface_components_are_periodic_tori_with_polished_roots():
    # B=(s-1/2)^2 + 0.01 cos(3 zeta), B=0.04 has two radial roots
    # for every zeta.  Each root sweeps one torus on the one-period quotient.
    field = _field(radial_coefficients=[0.25, -1.0, 1.0], toroidal_cosine=0.01)
    background = StructuredPrismMeshBackend(
        BackgroundMeshConfig(n_radial=10, n_poloidal=16, n_zeta=12)
    ).build(field)

    extraction = MarchingTetrahedraExtractor().extract(background, field, b=0.04)
    surface = extraction.full
    s, theta, zeta = _coordinates(surface.points)

    assert len(np.unique(surface.component_ids)) == 2
    assert _component_euler_characteristics(surface) == [0, 0]
    edge_counts = Counter(
        tuple(sorted(edge))
        for triangle in surface.triangles
        for edge in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        )
    )
    assert set(edge_counts.values()) == {2}
    assert np.any(surface.boundary_tags & surface.PERIODIC_SEAM)
    assert np.all(zeta < 2.0 * np.pi / field.nfp)
    np.testing.assert_allclose(field.B(s, theta, zeta), 0.04, atol=1.0e-10)
    assert np.all(surface.point_parent_edges >= 0)
    assert np.all(surface.triangle_parent_tetrahedra >= 0)


def test_physical_current_sign_selects_incoming_half_and_flux_balance_converges():
    # B=2+0.3s+0.1cos(3zeta), b=2.15 is a closed torus.  With C=G+iota I<0,
    # incoming g=B*D_B/C<0 is the D_B>0 half, catching a lost current sign.
    field = _field(
        radial_coefficients=[2.0, 0.3],
        toroidal_cosine=0.1,
        current_sign=-2.0,
        helical_sine=0.03,
    )
    imbalances = []
    reconstruction_errors = []
    extractions = []
    for resolution in ((8, 16, 9), (10, 20, 13)):
        background = StructuredPrismMeshBackend(
            BackgroundMeshConfig(*resolution)
        ).build(field)
        extraction = MarchingTetrahedraExtractor().extract(background, field, b=2.15)
        extractions.append(extraction)
        incoming_flux = surface_flux(extraction.incoming, field)
        outgoing_flux = surface_flux(extraction.outgoing, field)
        imbalances.append(
            abs(incoming_flux - outgoing_flux) / (incoming_flux + outgoing_flux)
        )
        reconstruction_errors.append(
            abs(surface_flux(extraction.full, field) - incoming_flux - outgoing_flux)
            / surface_flux(extraction.full, field)
        )

    extraction = extractions[-1]
    incoming_s, incoming_theta, incoming_zeta = _coordinates(extraction.incoming.points)
    outgoing_s, outgoing_theta, outgoing_zeta = _coordinates(extraction.outgoing.points)
    incoming_regular = extraction.incoming.g < -1.0e-9
    outgoing_regular = extraction.outgoing.g > 1.0e-9

    assert np.any(incoming_regular)
    assert np.any(outgoing_regular)
    assert np.all(
        field.D_B(
            incoming_s[incoming_regular],
            incoming_theta[incoming_regular],
            incoming_zeta[incoming_regular],
        )
        > 0.0
    )
    assert np.all(
        field.D_B(
            outgoing_s[outgoing_regular],
            outgoing_theta[outgoing_regular],
            outgoing_zeta[outgoing_regular],
        )
        < 0.0
    )
    assert len(extraction.g_zero.segments) > 0
    assert imbalances[1] < imbalances[0]
    assert imbalances[1] < 2.0e-5
    assert reconstruction_errors[1] < reconstruction_errors[0]
    assert reconstruction_errors[1] < 0.01
    for surface in (
        extraction.full,
        extraction.incoming,
        extraction.outgoing,
    ):
        surface_s, surface_theta, surface_zeta = _coordinates(surface.points)
        np.testing.assert_allclose(
            field.B(surface_s, surface_theta, surface_zeta),
            extraction.b,
            atol=1.0e-10,
        )
    np.testing.assert_allclose(extraction.g_zero.B, extraction.b, atol=1.0e-10)
    np.testing.assert_allclose(extraction.g_zero.g, 0.0, atol=1.0e-10)


def test_surface_flux_matches_independent_ds_wedge_dalpha_determinant():
    field = _field(radial_coefficients=[2.0, 0.3], toroidal_cosine=0.1)
    points = np.array([[0.4, 0.1, 0.2], [0.6, 0.1, 0.3], [0.4, 0.35, 0.45]])
    surface = SurfaceMesh(
        level=2.0,
        period=2.0 * np.pi / field.nfp,
        points=points,
        triangles=np.array([[0, 1, 2]]),
        B=np.full(3, 2.0),
        g=np.ones(3),
        boundary_tags=np.zeros(3, dtype=np.int64),
        point_parent_edges=np.full((3, 2), -1, dtype=np.int64),
        triangle_parent_tetrahedra=np.array([0]),
        component_ids=np.array([0]),
    )

    u = points[1] - points[0]
    v = points[2] - points[0]
    x, y, _ = np.mean(points, axis=0)
    s = x * x + y * y

    def ds(vector):
        return 2.0 * x * vector[0] + 2.0 * y * vector[1]

    def dalpha(vector):
        dtheta = (-y * vector[0] + x * vector[1]) / s
        return dtheta - float(field.iota(s)) * vector[2]

    expected = 0.5 * abs(ds(u) * dalpha(v) - ds(v) * dalpha(u))
    np.testing.assert_allclose(surface_flux(surface, field), expected, rtol=1.0e-14)


def test_undefined_physical_field_direction_has_explicit_surface_status():
    field = _field(
        radial_coefficients=[2.0, 0.3],
        toroidal_cosine=0.1,
        current_sign=0.0,
    )
    background = StructuredPrismMeshBackend(
        BackgroundMeshConfig(n_radial=3, n_poloidal=8, n_zeta=4)
    ).build(field)

    with pytest.raises(SurfaceExtractionError) as caught:
        MarchingTetrahedraExtractor().extract(background, field, b=2.15)

    assert caught.value.status is SurfaceStatus.DEGENERATE


def test_g_zero_projection_rejects_a_root_beyond_the_local_edge(monkeypatch):
    import alpha_analysis.j_connectivity.surface_extract as module

    field = _field(radial_coefficients=[2.0, 0.3], toroidal_cosine=0.1)
    period = 2.0 * np.pi / field.nfp
    first = np.array([0.6, 0.0, 0.2])
    second = np.array([0.0, 0.6, 0.4])
    local_edge_length = np.linalg.norm(second - first)
    monkeypatch.setattr(
        module,
        "root",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=True, x=np.array([0.0, 2.0 * local_edge_length])
        ),
    )
    monkeypatch.setattr(
        module,
        "_evaluate_B",
        lambda _field, points: np.full(len(points), 2.15),
    )
    monkeypatch.setattr(
        module,
        "_physical_g",
        lambda _field, points: np.zeros(len(points)),
    )

    with pytest.raises(SurfaceExtractionError, match="normal_displacement") as caught:
        module._polish_g_crossing(
            first,
            second,
            first_g=-1.0,
            second_g=1.0,
            field=field,
            b=2.15,
            period=period,
            config=module.SurfaceExtractionConfig(),
        )

    assert caught.value.status is SurfaceStatus.ROOT_FAILURE


def test_pyvista_prototype_returns_plain_arrays(monkeypatch):
    field = _field(radial_coefficients=[2.0, 0.3], toroidal_cosine=0.1)
    radius = np.sqrt(0.5)
    angles = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    contour_points = np.column_stack(
        (
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.full(3, 0.5 * np.pi / field.nfp),
        )
    )

    class FakePolyData:
        points = contour_points
        faces = np.array([3, 0, 1, 2])

        def triangulate(self):
            return self

    class FakeGrid:
        def __init__(self, *_args):
            self.point_data = {}

        def contour(self, levels, scalars):
            assert levels == [2.15]
            assert scalars == "B [field units]"
            return FakePolyData()

    class FakePyVista:
        class CellType:
            TETRA = 10

        UnstructuredGrid = FakeGrid

    import alpha_analysis.j_connectivity as package

    monkeypatch.setattr(
        package, "optional_import", lambda *_args, **_kwargs: FakePyVista
    )
    background = StructuredPrismMeshBackend(
        BackgroundMeshConfig(n_radial=1, n_poloidal=3, n_zeta=1)
    ).build(field)

    extraction = PyVistaSurfaceExtractor().extract(background, field, b=2.15)

    np.testing.assert_allclose(extraction.full.points, contour_points)
    np.testing.assert_array_equal(np.sort(extraction.full.triangles), [[0, 1, 2]])
    assert np.all(extraction.full.point_parent_edges == -1)
    assert np.all(extraction.full.triangle_parent_tetrahedra == -1)
    assert isinstance(extraction.full.points, np.ndarray)


def test_pitch_surface_diagnostic_is_headless_and_saveable(tmp_path):
    field = _field(radial_coefficients=[2.0, 0.3], toroidal_cosine=0.1)
    background = StructuredPrismMeshBackend(
        BackgroundMeshConfig(n_radial=5, n_poloidal=12, n_zeta=8)
    ).build(field)
    extraction = MarchingTetrahedraExtractor().extract(background, field, b=2.15)

    figure, axes = plot_pitch_surface(extraction)
    output = tmp_path / "pitch_surface.png"
    figure.savefig(output)

    assert output.stat().st_size > 0
    assert len(axes) == 2
    assert "component" in axes[0].get_title().lower()
    assert "incoming" in axes[1].get_title().lower()
    plt.close(figure)
