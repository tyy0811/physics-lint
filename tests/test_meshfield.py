"""MeshField tests — FE path against analytical Laplacian."""

import numpy as np
import pytest

pytest.importorskip("skfem")

from physics_lint.field import GridField, MeshField  # noqa: F401


def test_meshfield_l2_projected_laplacian_matches_analytical_sin_sin():
    from skfem import Basis, ElementTriP2, MeshTri

    mesh = MeshTri().refined(4)  # ~ 300 vertices
    basis = Basis(mesh, ElementTriP2())

    # u = sin(pi x) sin(pi y), Laplacian = -2 pi^2 u
    x = basis.doflocs[0]
    y = basis.doflocs[1]
    u_dofs = np.sin(np.pi * x) * np.sin(np.pi * y)
    expected_dofs = -2 * np.pi**2 * u_dofs

    field = MeshField(basis=basis, dofs=u_dofs)
    lap_dofs = field.laplacian_l2_projected_zero_trace().values()

    # L^2 error relative to the analytical Laplacian
    err = np.linalg.norm(lap_dofs - expected_dofs) / max(np.linalg.norm(expected_dofs), 1e-12)
    assert err < 0.1, f"MeshField laplacian relative error {err:.3e} exceeds tolerance"


def test_meshfield_integrate_constant_one():
    from skfem import Basis, ElementTriP2, MeshTri

    mesh = MeshTri().refined(3)
    basis = Basis(mesh, ElementTriP2())
    u_dofs = np.ones(basis.N)
    field = MeshField(basis=basis, dofs=u_dofs)
    area = field.integrate()
    assert abs(area - 1.0) < 1e-10  # unit square


def test_meshfield_l2_projected_laplacian_refines_with_mesh():
    """At a finer mesh the relative error should drop — sanity check the Galerkin projection."""
    from skfem import Basis, ElementTriP2, MeshTri

    errs: list[float] = []
    for n_refine in (3, 4, 5):
        mesh = MeshTri().refined(n_refine)
        basis = Basis(mesh, ElementTriP2())
        x = basis.doflocs[0]
        y = basis.doflocs[1]
        u_dofs = np.sin(np.pi * x) * np.sin(np.pi * y)
        expected = -2 * np.pi**2 * u_dofs
        field = MeshField(basis=basis, dofs=u_dofs)
        lap = field.laplacian_l2_projected_zero_trace().values()
        err = float(np.linalg.norm(lap - expected) / np.linalg.norm(expected))
        errs.append(err)
    # Monotonic decrease (allowing a tiny float tolerance)
    assert errs[1] < errs[0] + 1e-12, f"refinement did not reduce error: {errs}"
    assert errs[2] < errs[1] + 1e-12, f"second refinement did not reduce error: {errs}"


def test_meshfield_laplacian_raises_with_rename_pointer():
    """MeshField.laplacian() is a stub in V1 — it must raise a clear error
    pointing at the renamed method, not silently return a wrong value."""
    from skfem import Basis, ElementTriP2, MeshTri

    mesh = MeshTri().refined(2)
    basis = Basis(mesh, ElementTriP2())
    u_dofs = np.zeros(basis.N)
    field = MeshField(basis=basis, dofs=u_dofs)

    with pytest.raises(NotImplementedError) as exc:
        field.laplacian()
    msg = str(exc.value)
    assert "pointwise" in msg.lower()
    assert "laplacian_l2_projected_zero_trace" in msg
