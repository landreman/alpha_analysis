"""Baseline contracts for the j-connectivity package (DESIGN.md §23, milestone 0)."""

import importlib

import pytest


def test_j_connectivity_imports_without_optional_dependencies():
    """The base package must not import the optional connectivity stack."""
    package = importlib.import_module("alpha_analysis.j_connectivity")

    assert package.ConnectivityConfig().root_tolerance > 0.0


def test_statuses_preserve_unresolved_outcomes():
    """DESIGN.md §21.1 requires explicit, non-silent failure states."""
    from alpha_analysis.j_connectivity.types import TraceStatus

    assert TraceStatus.REGULAR.name == "REGULAR"
    assert TraceStatus.MAX_PERIODS.name == "MAX_PERIODS"
    assert TraceStatus.ROOT_FAILURE.name == "ROOT_FAILURE"
    assert TraceStatus.QUADRATURE_FAILURE.name == "QUADRATURE_FAILURE"


def test_run_metadata_records_reproducibility_context():
    from alpha_analysis.j_connectivity.types import RunMetadata

    metadata = RunMetadata(
        equilibrium_path="data/example.nc",
        equilibrium_hash="abc123",
        code_commit="deadbeef",
    )

    assert metadata.equilibrium_path == "data/example.nc"
    assert metadata.equilibrium_hash == "abc123"
    assert metadata.code_commit == "deadbeef"
    assert metadata.created_at.tzinfo is not None


def test_optional_dependency_error_names_extra(monkeypatch):
    from alpha_analysis.j_connectivity import optional_import

    def missing(_name):
        raise ModuleNotFoundError("No module named 'pyvista'", name="pyvista")

    monkeypatch.setattr(importlib, "import_module", missing)

    with pytest.raises(ImportError, match=r"alpha-analysis\[connectivity\]"):
        optional_import("pyvista", extra="connectivity")
