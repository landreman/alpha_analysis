from __future__ import annotations

import os
from collections import Counter
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from alpha_analysis import DATA_DIR, BoozerField

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
        iota_coefficients=np.array([0.7, 0.25]),
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


@pytest.mark.parametrize(
    "extractor_type",
    [MarchingTetrahedraExtractor, PyVistaSurfaceExtractor],
    ids=["marching-tetrahedra", "pyvista"],
)
def test_extractors_return_two_closed_periodic_tori_with_polished_roots(
    extractor_type,
):
    # B=(s-1/2)^2 + 0.01 cos(3 zeta), B=0.04 has two radial roots
    # for every zeta.  Each root sweeps one torus on the one-period quotient.
    field = _field(radial_coefficients=[0.25, -1.0, 1.0], toroidal_cosine=0.01)
    background = StructuredPrismMeshBackend(
        BackgroundMeshConfig(n_radial=10, n_poloidal=16, n_zeta=12)
    ).build(field)

    extraction = extractor_type().extract(background, field, b=0.04)
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
    directed_edges = Counter(
        edge
        for triangle in surface.triangles
        for edge in zip(triangle, np.roll(triangle, -1))
    )
    assert set(directed_edges.values()) == {1}
    assert all(directed_edges[(second, first)] == 1 for first, second in directed_edges)
    assert np.any(surface.boundary_tags & surface.PERIODIC_SEAM)
    assert np.all(zeta < 2.0 * np.pi / field.nfp)
    np.testing.assert_allclose(field.B(s, theta, zeta), 0.04, atol=1.0e-10)
    if extractor_type is MarchingTetrahedraExtractor:
        assert np.all(surface.point_parent_edges >= 0)
        assert np.all(surface.triangle_parent_tetrahedra >= 0)
    else:
        assert np.all(surface.point_parent_edges == -1)
        assert np.all(surface.triangle_parent_tetrahedra == -1)


@pytest.mark.parametrize(
    "extractor_type,resolutions,max_imbalance,expected_seam_points",
    [
        pytest.param(
            MarchingTetrahedraExtractor,
            ((10, 20, 13), (18, 36, 29)),
            2.0e-5,
            72,
            id="marching-tetrahedra",
        ),
        pytest.param(
            PyVistaSurfaceExtractor,
            ((6, 12, 7), (22, 44, 37)),
            5.0e-4,
            90,
            id="pyvista",
        ),
    ],
)
def test_extractors_use_physical_current_sign_and_flux_balance_converges(
    extractor_type, resolutions, max_imbalance, expected_seam_points
):
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
    for resolution in resolutions:
        background = StructuredPrismMeshBackend(
            BackgroundMeshConfig(*resolution)
        ).build(field)
        extraction = extractor_type().extract(background, field, b=2.15)
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
    assert (
        np.count_nonzero(extraction.full.boundary_tags & SurfaceMesh.PERIODIC_SEAM)
        == expected_seam_points
    )
    assert imbalances[1] < imbalances[0]
    assert imbalances[1] < max_imbalance
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


@pytest.mark.parametrize(
    "extractor_type",
    [MarchingTetrahedraExtractor, PyVistaSurfaceExtractor],
    ids=["marching-tetrahedra", "pyvista"],
)
@pytest.mark.parametrize(
    "radial_coefficients,b,resolution,expect_seam",
    [
        pytest.param([2.0, -0.3], 1.75, (10, 20, 13), False, id="edge-only"),
        pytest.param([2.0, 0.3], 2.35, (8, 16, 5), True, id="edge-and-seam"),
    ],
)
def test_extractors_preserve_boundary_provenance_inside_plasma_domain(
    extractor_type, radial_coefficients, b, resolution, expect_seam
):
    # These analytic levels reach the physical edge; the second also crosses
    # the periodic seam, so EDGE recovery must not absorb seam-only vertices.
    field = _field(radial_coefficients=radial_coefficients, toroidal_cosine=0.1)
    background = StructuredPrismMeshBackend(BackgroundMeshConfig(*resolution)).build(
        field
    )

    surface = extractor_type().extract(background, field, b=b).full
    radii_squared = np.sum(surface.points[:, :2] ** 2, axis=1)
    edge = (surface.boundary_tags & SurfaceMesh.EDGE) != 0
    seam = (surface.boundary_tags & SurfaceMesh.PERIODIC_SEAM) != 0
    edge_counts = Counter(
        tuple(sorted(segment))
        for triangle in surface.triangles
        for segment in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        )
    )
    boundary_vertices = np.unique(
        [segment for segment, count in edge_counts.items() if count == 1]
    )

    assert len(boundary_vertices) > 0
    np.testing.assert_array_equal(np.flatnonzero(edge), boundary_vertices)
    assert bool(np.any(seam)) is expect_seam
    assert np.all(radii_squared <= 1.0 + 32.0 * np.finfo(float).eps)


@pytest.mark.parametrize(
    "extractor_type",
    [MarchingTetrahedraExtractor, PyVistaSurfaceExtractor],
    ids=["marching-tetrahedra", "pyvista"],
)
def test_extractors_return_empty_mesh_for_absent_level(extractor_type):
    field = _field(radial_coefficients=[2.0, 0.3], toroidal_cosine=0.1)
    background = StructuredPrismMeshBackend(
        BackgroundMeshConfig(n_radial=3, n_poloidal=8, n_zeta=4)
    ).build(field)

    extraction = extractor_type().extract(background, field, b=3.0)

    for surface in (extraction.full, extraction.incoming, extraction.outgoing):
        assert surface.points.shape == (0, 3)
        assert surface.triangles.shape == (0, 3)
    assert extraction.g_zero.points.shape == (0, 3)
    assert extraction.g_zero.segments.shape == (0, 2)


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


@pytest.mark.parametrize(
    "extractor_type",
    [MarchingTetrahedraExtractor, PyVistaSurfaceExtractor],
    ids=["marching-tetrahedra", "pyvista"],
)
def test_extractors_report_undefined_physical_field_direction(extractor_type):
    field = _field(
        radial_coefficients=[2.0, 0.3],
        toroidal_cosine=0.1,
        current_sign=0.0,
    )
    background = StructuredPrismMeshBackend(
        BackgroundMeshConfig(n_radial=3, n_poloidal=8, n_zeta=4)
    ).build(field)

    with pytest.raises(SurfaceExtractionError) as caught:
        extractor_type().extract(background, field, b=2.15)

    assert caught.value.status is SurfaceStatus.DEGENERATE


def test_g_zero_planar_runaway_root_is_replaced_by_a_local_bracketed_root(
    monkeypatch,
):
    # The planar Newton solve claims a root two edge lengths off the edge;
    # the locality check must discard it and the bracketed fallback must
    # return a point on the edge instead of the distant impostor.
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

    point, resolved = module._polish_g_crossing(
        first,
        second,
        first_g=-1.0,
        second_g=1.0,
        field=field,
        b=2.15,
        period=period,
        config=module.SurfaceExtractionConfig(),
    )

    assert resolved
    assert module._distance_to_segment(point, first, second) < 1.0e-9


def test_g_zero_polish_reports_an_unresolved_split_when_no_local_root_exists(
    monkeypatch,
):
    # g pretends to be +1 everywhere while the endpoint bracket claims a sign
    # change, so no g=0 point exists anywhere: no stage may fabricate a
    # resolved split point.  The polish must instead report the split as
    # unresolved with a local, on-surface point (the g sign discontinuity).
    import alpha_analysis.j_connectivity.surface_extract as module

    field = _field(radial_coefficients=[2.0, 0.3], toroidal_cosine=0.1)
    period = 2.0 * np.pi / field.nfp
    first = np.array([0.6, 0.0, 0.2])
    second = np.array([0.0, 0.6, 0.4])
    monkeypatch.setattr(
        module,
        "_evaluate_B",
        lambda _field, points: np.full(len(points), 2.15),
    )
    monkeypatch.setattr(
        module,
        "_physical_g",
        lambda _field, points: np.ones(len(points)),
    )

    point, resolved = module._polish_g_crossing(
        first,
        second,
        first_g=-1.0,
        second_g=1.0,
        field=field,
        b=2.15,
        period=period,
        config=module.SurfaceExtractionConfig(),
    )

    assert not resolved
    assert module._distance_to_segment(point, first, second) <= np.linalg.norm(
        second - first
    )


def _record_split_point_locality(monkeypatch):
    """Wrap ``_polish_g_crossing`` to record how local each split point is.

    Returns two lists that fill during extraction: the number of planar-solve
    rejections (the fallback exercised), and per split point the pair
    ``(distance to the parent edge / edge length, x^2 + y^2)``.
    """
    import alpha_analysis.j_connectivity.surface_extract as module

    planar = module._polish_g_crossing_planar
    rejections = []

    def counting_planar(*args, **kwargs):
        result = planar(*args, **kwargs)
        if result is None:
            rejections.append(1)
        return result

    polish = module._polish_g_crossing
    locality = []

    def recording_polish(
        first, second, first_g, second_g, field, b, period, config, **kwargs
    ):
        point, resolved = polish(
            first, second, first_g, second_g, field, b, period, config, **kwargs
        )
        second_local = module._unwrap_point_relative(second, first, period)
        distance = module._distance_to_segment(
            module._unwrap_point_relative(point, first, period), first, second_local
        )
        edge_length = float(np.linalg.norm(second_local - first))
        locality.append((distance, edge_length, float(np.sum(point[:2] ** 2))))
        return point, resolved

    monkeypatch.setattr(module, "_polish_g_crossing_planar", counting_planar)
    monkeypatch.setattr(module, "_polish_g_crossing", recording_polish)
    return rejections, locality


def _assert_thin_tube_extraction_is_sound(
    extraction, field, b, rejections, locality, expect_unresolved=False
):
    # The scenario must actually exercise the fallback, or it tests nothing.
    assert rejections
    if expect_unresolved:
        assert extraction.status is SurfaceStatus.UNRESOLVED
        assert extraction.n_unresolved_splits > 0
    else:
        assert extraction.status is SurfaceStatus.REGULAR
        assert extraction.n_unresolved_splits == 0
    assert len(extraction.incoming.triangles) > 0
    assert len(extraction.outgoing.triangles) > 0
    assert len(extraction.g_zero.segments) > 0
    for surface in (extraction.full, extraction.incoming, extraction.outgoing):
        s, theta, zeta = _coordinates(surface.points)
        np.testing.assert_allclose(field.B(s, theta, zeta), b, atol=1.0e-10)
    np.testing.assert_allclose(extraction.g_zero.B, b, atol=1.0e-10)
    np.testing.assert_allclose(extraction.g_zero.g, 0.0, atol=1.0e-10)
    # Split points must be local to their parent edge and inside the plasma
    # domain: a polished root far from its edge or beyond x^2+y^2=1 is a
    # spurious solution of the simultaneous system, not the local crossing.
    # The locality scale is the acceptance radius of the bracketed fallback:
    # four edge lengths, or twice the background patch scale for marching
    # edges much shorter than their background cell.
    import alpha_analysis.j_connectivity.surface_extract as module

    patch_scale = module._max_triangle_edge_length(extraction.full)
    for distance, edge_length, radius_squared in locality:
        assert distance <= max(4.0 * edge_length, 2.0 * patch_scale) + 1.0e-6
        assert radius_squared <= 1.0 + 1.0e-9


@pytest.mark.parametrize(
    "b,resolution",
    [
        pytest.param(-0.009, (6, 12, 8), id="deep-thin-tube"),
        pytest.param(-0.005, (6, 12, 8), id="mid-thin-tube"),
        pytest.param(0.001, (6, 8, 6), id="coarse-mesh"),
    ],
)
def test_split_points_survive_thin_tube_sheet_jumps_near_min_B(
    monkeypatch, b, resolution
):
    # B = (s-1/2)^2 + 0.01 cos(3 zeta) + helical m=1: near min(B) the level
    # set is a thin helical tube crossed by background edges, the planar
    # Newton solve escapes to distant roots, and the projected chord solve is
    # multi-sheeted.  The bracketed fallbacks must still split every edge
    # with points that are local and inside the unit disk.
    field = _field(
        radial_coefficients=[0.25, -1.0, 1.0],
        toroidal_cosine=0.01,
        helical_sine=0.01,
    )
    background = StructuredPrismMeshBackend(BackgroundMeshConfig(*resolution)).build(
        field
    )
    rejections, locality = _record_split_point_locality(monkeypatch)

    extraction = MarchingTetrahedraExtractor().extract(background, field, b)

    _assert_thin_tube_extraction_is_sound(extraction, field, b, rejections, locality)


@pytest.mark.parametrize(
    "b,resolution",
    [
        # 2.44736 is one of the originally reported ROOT_FAILURE levels on
        # the example script's default background mesh.
        pytest.param(2.44736, (6, 24, 12), id="reported-root-failure"),
        # At 2.34 the first crossing met by a fixed-orientation plane trace
        # sits more than ten edge lengths away; the nearest crossing must be
        # returned instead.
        pytest.param(2.34, (6, 24, 12), id="far-first-crossing"),
        # At 2.38 on the coarse mesh the gradient-seeded cutting plane meets
        # the thin tube in two disconnected curves and its crossings leave
        # the unit disk; the rotated cutting planes must find the local
        # in-disk crossing.
        pytest.param(2.38, (4, 16, 8), id="disconnected-plane-cut"),
    ],
)
def test_w7x_split_points_polish_on_the_thin_tube_near_min_B(
    monkeypatch, b, resolution
):
    # Regression for the W7-X boozmn levels near min(B): extraction must
    # succeed, and the split points must stay local to their parent edges
    # and inside the unit disk (no wild vertices).
    field = BoozerField.from_boozmn(
        os.path.join(
            DATA_DIR, "boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
        )
    )
    background = StructuredPrismMeshBackend(BackgroundMeshConfig(*resolution)).build(
        field
    )
    rejections, locality = _record_split_point_locality(monkeypatch)

    extraction = MarchingTetrahedraExtractor().extract(background, field, b)

    _assert_thin_tube_extraction_is_sound(extraction, field, b, rejections, locality)


def test_sheet_bridging_edge_yields_an_unresolved_g_jump_split(monkeypatch):
    # Regression for a gmsh-mesh edge (target_size=0.25, b=2.37689) whose
    # marching triangle bridges two sheets of the under-resolved thin tube:
    # walking on the surface from one endpoint can never reach the other,
    # the genuine g=0 crossings are more than a background cell away, and
    # the nearby portion of the g=0 curve lies outside the unit disk.  The
    # polish must classify the split as unresolved and place the vertex at
    # the local g sign discontinuity instead of failing or returning a
    # distant root.
    import alpha_analysis.j_connectivity.surface_extract as module

    field = BoozerField.from_boozmn(
        os.path.join(
            DATA_DIR, "boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
        )
    )
    b = 2.37689
    first = np.array([-0.8233028, -0.24078709, 0.55283671])
    second = np.array([-0.85337398, -0.29026496, 0.70184775])

    point, resolved = module._polish_g_crossing(
        first,
        second,
        first_g=-0.09975090828317087,
        second_g=0.09779409743075868,
        field=field,
        b=b,
        period=2.0 * np.pi / field.nfp,
        config=module.SurfaceExtractionConfig(),
        patch_scale=0.4605287527600239,
    )

    assert not resolved
    # The point is on the surface, inside the domain, and local to the edge.
    np.testing.assert_allclose(
        module._evaluate_B(field, point[np.newaxis, :]), b, atol=1.0e-10
    )
    assert float(np.sum(point[:2] ** 2)) <= 1.0 + 1.0e-9
    assert module._distance_to_segment(point, first, second) <= np.linalg.norm(
        second - first
    )


@pytest.mark.slow
def test_gmsh_extraction_reports_unresolved_sheet_bridging_splits(monkeypatch):
    # End-to-end version: the level with sheet-bridging triangles extracts,
    # reports UNRESOLVED with a G_JUMP count, keeps every G_JUMP vertex out
    # of the g=0 curve, and the curve's own points still satisfy g=0.
    from alpha_analysis.j_connectivity.background_mesh import (
        GmshBackgroundMeshBackend,
        GmshBackgroundMeshConfig,
    )

    field = BoozerField.from_boozmn(
        os.path.join(
            DATA_DIR, "boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
        )
    )
    background = GmshBackgroundMeshBackend(
        GmshBackgroundMeshConfig(target_size=0.25)
    ).build(field)
    rejections, locality = _record_split_point_locality(monkeypatch)

    b = 2.40339
    extraction = MarchingTetrahedraExtractor().extract(background, field, b)

    _assert_thin_tube_extraction_is_sound(
        extraction, field, b, rejections, locality, expect_unresolved=True
    )
    for half in (extraction.incoming, extraction.outgoing):
        assert np.any((half.boundary_tags & SurfaceMesh.G_JUMP) != 0)
    assert np.all((extraction.g_zero.boundary_tags & SurfaceMesh.G_JUMP) == 0)


@pytest.mark.slow
def test_gmsh_extraction_polishes_boundary_exit_split_points(monkeypatch):
    # End-to-end version of the boundary-exit regression: the reported
    # failing configuration (gmsh backend, target_size=0.16, the lowest
    # level of the example script) must extract, with the boundary-exit
    # split point on x^2+y^2=1 carrying both EDGE and G_ZERO provenance.
    from alpha_analysis.j_connectivity.background_mesh import (
        GmshBackgroundMeshBackend,
        GmshBackgroundMeshConfig,
    )

    field = BoozerField.from_boozmn(
        os.path.join(
            DATA_DIR, "boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
        )
    )
    background = GmshBackgroundMeshBackend(
        GmshBackgroundMeshConfig(target_size=0.16)
    ).build(field)
    rejections, locality = _record_split_point_locality(monkeypatch)

    b = 2.45077049
    extraction = MarchingTetrahedraExtractor().extract(background, field, b)

    _assert_thin_tube_extraction_is_sound(extraction, field, b, rejections, locality)
    curve = extraction.g_zero
    radii = np.linalg.norm(curve.points[:, :2], axis=1)
    on_boundary = np.abs(radii - 1.0) <= 1.0e-9
    assert np.any(on_boundary)
    expected = SurfaceMesh.EDGE | SurfaceMesh.G_ZERO
    assert np.all((curve.boundary_tags[on_boundary] & expected) == expected)


def test_g_zero_crossing_exits_through_the_outer_boundary(monkeypatch):
    # Regression for a gmsh-mesh edge (target_size=0.16, b=2.45077049) whose
    # surface patch has no interior g=0 crossing inside the unit disk: the
    # g=0 curve leaves the domain, every in-plane trace exits the disk
    # without a sign change, and the split point must be found on the
    # boundary curve B=b, x^2+y^2=1 instead of failing or fabricating an
    # out-of-domain root.
    import alpha_analysis.j_connectivity.surface_extract as module

    field = BoozerField.from_boozmn(
        os.path.join(
            DATA_DIR, "boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
        )
    )
    b = 2.45077049
    first = np.array([0.8563976980785168, -0.04927986920210465, 0.683797253974709])
    second = first + np.array(
        [0.14360230192148316, 0.04927986920210441, -0.14139055613177587]
    )
    boundary_stage = module._polish_g_zero_on_outer_boundary
    calls = []

    def counting_boundary_stage(*args, **kwargs):
        calls.append(1)
        return boundary_stage(*args, **kwargs)

    monkeypatch.setattr(
        module, "_polish_g_zero_on_outer_boundary", counting_boundary_stage
    )

    point, resolved = module._polish_g_crossing(
        first,
        second,
        first_g=0.09250636086144219,
        second_g=-0.17835418567150624,
        field=field,
        b=b,
        period=2.0 * np.pi / field.nfp,
        config=module.SurfaceExtractionConfig(),
        patch_scale=0.28858997612245285,
    )

    # The interior stages must have given way to the boundary stage, and the
    # result must sit exactly on the domain boundary with both residuals met.
    assert calls
    assert resolved
    np.testing.assert_allclose(np.linalg.norm(point[:2]), 1.0, rtol=0.0, atol=1.0e-9)
    np.testing.assert_allclose(
        module._evaluate_B(field, point[np.newaxis, :]), b, atol=1.0e-10
    )
    np.testing.assert_allclose(
        module._physical_g(field, point[np.newaxis, :]), 0.0, atol=1.0e-10
    )
    # Locality: the boundary exit point stays within the acceptance radius.
    assert module._distance_to_segment(point, first, second) <= max(
        4.0 * np.linalg.norm(second - first), 2.0 * 0.28858997612245285
    )


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
        point_data = {
            "boundary tag [bit mask]": np.full(3, 2, dtype=np.int64),
            "background OUTER boundary indicator": np.ones(3),
        }

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


def test_chandrupatla_roots_match_the_scalar_brentq_contract():
    # The batched bracketed solver replacing the per-edge brentq calls must
    # land on the same roots at the same parameter tolerance, across a
    # family of brackets of very different stiffness solved simultaneously.
    import alpha_analysis.j_connectivity.surface_extract as module

    rng = np.random.default_rng(3)
    n = 50
    true_roots = rng.uniform(0.02, 0.98, n)
    stiffness = 10.0 ** rng.uniform(-2.0, 2.0, n)

    def residuals(parameters, active):
        return np.sinh(stiffness[active] * (parameters - true_roots[active]))

    everywhere = np.ones(n, dtype=bool)
    lower = residuals(np.zeros(n), everywhere)
    upper = residuals(np.ones(n), everywhere)
    config = module.SurfaceExtractionConfig()
    roots, unconverged = module._chandrupatla_roots(residuals, lower, upper, config)

    assert not unconverged.any()
    np.testing.assert_allclose(roots, true_roots, rtol=0.0, atol=1.0e-11)
