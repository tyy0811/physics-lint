"""PH-POS-002 — Maximum principle violation for Laplace."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_pos_002


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


def test_ph_pos_002_harmonic_passes():
    # u = x^2 - y^2, harmonic. Boundary values span [-1, 1], interior is within.
    spec = _laplace_dirichlet_spec()
    n = 32
    xg = np.linspace(0.0, 1.0, n)
    yg = np.linspace(0.0, 1.0, n)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    u = mesh_x**2 - mesh_y**2
    field = GridField(u, h=(1.0 / 31, 1.0 / 31), periodic=False)
    result = ph_pos_002.check(field, spec, boundary_values=field.values_on_boundary())
    assert result.status == "PASS"


def test_ph_pos_002_interior_overshoot_fails():
    spec = _laplace_dirichlet_spec()
    n = 32
    u = np.zeros((n, n))
    u[15, 15] = 5.0  # wild interior value above all boundary values
    field = GridField(u, h=(1.0 / 31, 1.0 / 31), periodic=False)
    boundary_vals = field.values_on_boundary()  # all zeros
    result = ph_pos_002.check(field, spec, boundary_values=boundary_vals)
    assert result.status == "FAIL"
    assert result.raw_value is not None and result.raw_value > 0


def test_ph_pos_002_metadata():
    assert ph_pos_002.__rule_id__ == "PH-POS-002"
    assert ph_pos_002.__default_severity__ == "error"


def test_ph_pos_002_non_laplace_skipped():
    spec = DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [16, 16],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    field = GridField(np.zeros((16, 16)), h=1.0 / 15, periodic=False)
    result = ph_pos_002.check(field, spec, boundary_values=np.zeros(4))
    assert result.status == "SKIPPED"
    assert "laplace only" in result.reason.lower()


def test_ph_pos_002_interior_undershoot_fails():
    # Symmetric test: interior value BELOW the minimum boundary value.
    spec = _laplace_dirichlet_spec()
    n = 16
    u = np.ones((n, n))
    u[8, 8] = -5.0
    field = GridField(u, h=1.0 / (n - 1), periodic=False)
    boundary_vals = field.values_on_boundary()
    result = ph_pos_002.check(field, spec, boundary_values=boundary_vals)
    assert result.status == "FAIL"
    # The overshoot is 1 - (-5) = 6 below the boundary min
    assert result.raw_value is not None
    assert result.raw_value > 5
