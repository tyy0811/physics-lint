"""PH-CON-004 per-element conservation hotspot tests — MeshField only."""

import pytest

pytest.importorskip("skfem")

import numpy as np
from skfem import Basis, ElementTriP2, MeshTri

from physics_lint import DomainSpec
from physics_lint.field import GridField, MeshField
from physics_lint.rules import ph_con_004


def _spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [32, 32],  # nominal; not used by mesh rules
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "mesh", "backend": "fd", "adapter_path": "x"},
        }
    )


def test_ph_con_004_smooth_solution_has_no_hotspots():
    mesh = MeshTri().refined(4)
    basis = Basis(mesh, ElementTriP2())
    x = basis.doflocs[0]
    y = basis.doflocs[1]
    u_dofs = np.sin(np.pi * x) * np.sin(np.pi * y)
    field = MeshField(basis=basis, dofs=u_dofs)
    result = ph_con_004.check(field, _spec())
    assert result.status == "PASS"
    assert result.raw_value is not None
    # A smooth field should not produce a large max/mean ratio. Allow headroom
    # for the FE projection's element-to-element variance; 10 is the WARN cut.
    assert result.raw_value < 10.0


def test_ph_con_004_skip_on_non_mesh():
    field = GridField(np.zeros((16, 16)), h=(1 / 15, 1 / 15), periodic=False)
    result = ph_con_004.check(field, _spec())
    assert result.status == "SKIPPED"
    assert "MeshField" in (result.reason or "")


def test_ph_con_004_constant_field_skips_numerical_zero():
    """A constant field has Δu ≡ 0; the L²-projected zero-trace operator
    still produces interior values that are zero to roundoff, so the
    mean_elem is below the numerical-zero guard and the rule SKIPs.
    The check here is that the interior-element mask does not crash and
    the rule takes a well-defined path rather than producing a spurious
    WARN driven by floating-point noise."""
    mesh = MeshTri().refined(3)
    basis = Basis(mesh, ElementTriP2())
    u_dofs = np.ones(basis.N)
    field = MeshField(basis=basis, dofs=u_dofs)
    result = ph_con_004.check(field, _spec())
    # Accept PASS or SKIPPED — both are well-defined outcomes for a
    # numerically-zero residual field. A WARN would indicate spurious
    # amplification from the max/mean ratio on roundoff noise and should
    # fail the test.
    assert result.status in {"PASS", "SKIPPED"}
