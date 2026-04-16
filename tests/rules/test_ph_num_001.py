"""PH-NUM-001 quadrature convergence tests — V1 structural stub."""

import pytest

pytest.importorskip("skfem")

import numpy as np
from skfem import Basis, ElementTriP2, MeshTri

from physics_lint import DomainSpec
from physics_lint.field import GridField, MeshField
from physics_lint.rules import ph_num_001


def _spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [32, 32],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "mesh", "backend": "fd", "adapter_path": "x"},
        }
    )


def test_ph_num_001_polynomial_passes_as_v1_stub():
    mesh = MeshTri().refined(3)
    basis = Basis(mesh, ElementTriP2())
    x = basis.doflocs[0]
    y = basis.doflocs[1]
    u_dofs = x * (1 - x) * y * (1 - y)
    field = MeshField(basis=basis, dofs=u_dofs)
    result = ph_num_001.check(field, _spec())
    assert result.status == "PASS"
    assert result.reason == "qorder convergence check is a stub until V1.1"
    assert result.raw_value is not None


def test_ph_num_001_skip_on_non_mesh():
    result = ph_num_001.check(
        GridField(np.zeros((16, 16)), h=(1 / 15, 1 / 15), periodic=False),
        _spec(),
    )
    assert result.status == "SKIPPED"
    assert "MeshField" in (result.reason or "")
