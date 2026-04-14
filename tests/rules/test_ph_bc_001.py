"""PH-BC-001 — relative vs absolute mode branching."""

import numpy as np
import pytest

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_bc_001


def _laplace_hd_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [32, 32],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )


def _laplace_dirichlet_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [32, 32],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )


def _zeros_on_hd_boundary() -> GridField:
    n = 32
    xg = np.linspace(0.0, 1.0, n)
    yg = np.linspace(0.0, 1.0, n)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    # u = sin(pi x) sin(pi y): boundary is zero by construction.
    u = np.sin(np.pi * mesh_x) * np.sin(np.pi * mesh_y)
    return GridField(u, h=(1.0 / (n - 1), 1.0 / (n - 1)), periodic=False)


def test_ph_bc_001_homogeneous_dirichlet_is_absolute_pass():
    spec = _laplace_hd_spec()
    field = _zeros_on_hd_boundary()
    boundary_target = np.zeros_like(field.values_on_boundary())
    result = ph_bc_001.check(field, spec, boundary_target=boundary_target)
    assert result.rule_id == "PH-BC-001"
    assert result.mode == "absolute"
    assert result.status == "PASS"


def test_ph_bc_001_homogeneous_dirichlet_violation_is_fail():
    spec = _laplace_hd_spec()
    n = 32
    u = np.ones((n, n))  # violates u=0 on boundary
    field = GridField(u, h=(1.0 / (n - 1), 1.0 / (n - 1)), periodic=False)
    boundary_target = np.zeros_like(field.values_on_boundary())
    result = ph_bc_001.check(field, spec, boundary_target=boundary_target)
    assert result.mode == "absolute"
    assert result.status == "FAIL"
    assert result.raw_value is not None and result.raw_value > 0


def test_ph_bc_001_inhomogeneous_is_relative_pass():
    spec = _laplace_dirichlet_spec()
    n = 32
    xg = np.linspace(0.0, 1.0, n)
    yg = np.linspace(0.0, 1.0, n)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    u = mesh_x**2 - mesh_y**2  # harmonic polynomial
    field = GridField(u, h=(1.0 / (n - 1), 1.0 / (n - 1)), periodic=False)
    boundary_target = field.values_on_boundary().copy()
    result = ph_bc_001.check(field, spec, boundary_target=boundary_target)
    assert result.mode == "relative"
    assert result.status == "PASS"


def test_ph_bc_001_rejects_shape_mismatch():
    spec = _laplace_hd_spec()
    field = _zeros_on_hd_boundary()
    wrong_target = np.zeros(7)  # deliberately wrong shape
    with pytest.raises(ValueError, match="does not match"):
        ph_bc_001.check(field, spec, boundary_target=wrong_target)


def test_ph_bc_001_metadata():
    assert ph_bc_001.__rule_id__ == "PH-BC-001"
    assert ph_bc_001.__default_severity__ == "error"
    assert "adapter" in ph_bc_001.__input_modes__
    assert "dump" in ph_bc_001.__input_modes__


def test_ph_bc_001_inhomogeneous_violation_is_fail():
    spec = _laplace_dirichlet_spec()
    n = 32
    xg = np.linspace(0.0, 1.0, n)
    yg = np.linspace(0.0, 1.0, n)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    u = mesh_x**2 - mesh_y**2
    field = GridField(u, h=(1.0 / (n - 1), 1.0 / (n - 1)), periodic=False)
    # Set target to a very different trace to force relative-mode FAIL.
    boundary_target = field.values_on_boundary().copy() + 100.0
    result = ph_bc_001.check(field, spec, boundary_target=boundary_target)
    assert result.mode == "relative"
    assert result.status in {"WARN", "FAIL"}
