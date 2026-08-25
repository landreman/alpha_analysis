from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from alpha_analysis.j_connectivity.background_mesh import (
    BackgroundMeshConfig,
    StructuredPrismMeshBackend,
    signed_tetrahedron_volumes,
    tetrahedron_quality,
)
from alpha_analysis.j_connectivity.synthetic_fields import SyntheticFourierField
from alpha_analysis.j_connectivity.visualization import plot_background_mesh


def _field(nfp=3):
    return SyntheticFourierField(
        nfp=nfp,
        m=np.array([0, 1]),
        n=np.array([0, nfp]),
        cosine_coefficients=np.array([[2.0], [0.2]]),
        sine_coefficients=np.array([[0.0], [0.1]]),
        G_coefficients=np.array([1.0]),
        I_coefficients=np.array([0.0]),
        iota_coefficients=np.array([0.7]),
    )


def _mesh(config=BackgroundMeshConfig(3, 12, 3)):
    return StructuredPrismMeshBackend(config).build(_field())


def _face_counts_after_seam_identification(mesh):
    representative = np.arange(mesh.points.shape[0])
    representative[mesh.periodic_node_pairs[:, 1]] = mesh.periodic_node_pairs[:, 0]
    counts = {}
    for tetrahedron in representative[mesh.tetrahedra]:
        for face in combinations(tetrahedron, 3):
            key = tuple(sorted(face))
            counts[key] = counts.get(key, 0) + 1
    return counts


def test_structured_mesh_has_positive_orientation_volume_and_quality():
    config = BackgroundMeshConfig(n_radial=3, n_poloidal=12, n_zeta=4)
    mesh = StructuredPrismMeshBackend(config).build(_field(nfp=2))
    volumes = signed_tetrahedron_volumes(mesh)
    quality = tetrahedron_quality(mesh)

    assert np.all(volumes > 0.0)
    assert np.all(np.isfinite(quality))
    assert np.all((quality > 0.0) & (quality <= 1.0))
    polygon_area = 0.5 * config.n_poloidal * np.sin(2.0 * np.pi / config.n_poloidal)
    np.testing.assert_allclose(volumes.sum(), polygon_area * np.pi, rtol=2e-14)


def test_periodic_seam_pairing_is_exact_one_to_one_and_closes_seam():
    mesh = _mesh()
    lower, upper = mesh.periodic_node_pairs.T

    assert len(np.unique(lower)) == len(lower)
    assert len(np.unique(upper)) == len(upper)
    assert len(lower) == np.count_nonzero(mesh.boundary_tags & mesh.ZETA_MIN)
    np.testing.assert_array_equal(mesh.points[lower, :2], mesh.points[upper, :2])
    assert np.all(mesh.points[lower, 2] == 0.0)
    assert np.all(mesh.points[upper, 2] == 2.0 * np.pi / _field().nfp)

    counts = _face_counts_after_seam_identification(mesh)
    assert max(counts.values()) == 2
    boundary_faces = [face for face, count in counts.items() if count == 1]
    assert boundary_faces
    assert all(
        np.all(mesh.boundary_tags[list(face)] & mesh.OUTER) for face in boundary_faces
    )


def test_structured_mesh_is_deterministic_and_samples_field_arrays():
    first = _mesh()
    second = _mesh()

    for name in (
        "points",
        "tetrahedra",
        "periodic_node_pairs",
        "boundary_tags",
        "B",
        "D_B",
        "D2_B",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.points.dtype == np.float64
    assert first.tetrahedra.dtype == np.int64
    assert first.B.shape == (first.points.shape[0],)
    assert np.all(np.isfinite(first.B))


def test_background_mesh_visual_example_is_headless_and_saveable(tmp_path):
    mesh = _mesh(BackgroundMeshConfig(2, 8, 2))
    figure, axes = plot_background_mesh(mesh)
    output = tmp_path / "background_mesh.png"
    figure.savefig(output)

    assert output.stat().st_size > 0
    assert len(axes) == 3
    assert "periodic" in axes[1].get_title().lower()
    assert "quality" in axes[2].get_title().lower()
    plt.close(figure)


def test_pyvista_conversion_is_an_optional_view(monkeypatch):
    mesh = _mesh(BackgroundMeshConfig(1, 6, 1))

    class FakeGrid:
        def __init__(self, cells, cell_types, points):
            self.cells = cells
            self.cell_types = cell_types
            self.points = points
            self.point_data = {}

    class FakePyVista:
        class CellType:
            TETRA = 10

        UnstructuredGrid = FakeGrid

    import alpha_analysis.j_connectivity as package

    monkeypatch.setattr(
        package, "optional_import", lambda *_args, **_kwargs: FakePyVista
    )
    grid = mesh.to_pyvista()

    np.testing.assert_array_equal(grid.points, mesh.points)
    assert np.all(grid.cell_types == FakePyVista.CellType.TETRA)
    assert set(grid.point_data) == {
        "B [field units]",
        "D_parallel B [field units/rad]",
        "D_parallel^2 B [field units/rad^2]",
        "boundary tag [bit mask]",
    }

    monkeypatch.setattr(
        package,
        "optional_import",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ImportError("connectivity")),
    )
    with pytest.raises(ImportError, match="connectivity"):
        mesh.to_pyvista()
