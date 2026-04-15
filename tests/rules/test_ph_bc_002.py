"""PH-BC-002 — boundary flux imbalance via divergence theorem."""

import numpy as np
import pytest

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_bc_002


def _laplace_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [64, 64],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )


def test_ph_bc_002_harmonic_has_zero_net_flux():
    # u = x^2 - y^2, harmonic. Net flux around the boundary = 0 (up to FD error).
    spec = _laplace_spec()
    n = 64
    xg = np.linspace(0.0, 1.0, n)
    yg = np.linspace(0.0, 1.0, n)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    u = mesh_x**2 - mesh_y**2
    field = GridField(u, h=(1.0 / (n - 1), 1.0 / (n - 1)), periodic=False)
    result = ph_bc_002.check(field, spec)
    assert result.rule_id == "PH-BC-002"
    assert result.status == "PASS"
    assert result.raw_value is not None
    assert abs(result.raw_value) < 0.01  # small FD edge contribution


def test_ph_bc_002_non_harmonic_has_nonzero_net_flux():
    spec = _laplace_spec()
    n = 64
    xg = np.linspace(0.0, 1.0, n)
    yg = np.linspace(0.0, 1.0, n)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    # u = x^2 + y^2 has Laplacian = 4, net flux integral = 4 (genuinely non-harmonic).
    # NOTE: The plan originally used exp(x)*sin(y), but that IS harmonic
    # (Laplacian = exp(x)*sin(y) + exp(x)*(-sin(y)) = 0), so the net flux is
    # ~0 and the test incorrectly passes. x^2 + y^2 is the fix.
    u = mesh_x**2 + mesh_y**2
    field = GridField(u, h=(1.0 / (n - 1), 1.0 / (n - 1)), periodic=False)
    result = ph_bc_002.check(field, spec)
    assert result.status in {"WARN", "FAIL"}
    assert result.raw_value is not None and abs(result.raw_value) > 0.01


def test_ph_bc_002_poisson_is_skipped_until_week_2():
    # Matching sibling of PH-RES-001: Poisson source integration lands in
    # Week 2. Until then the rule must emit SKIPPED rather than crash with
    # NotImplementedError mid-run.
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
    field = GridField(np.zeros((16, 16)), h=1.0 / 15, periodic=False, backend="fd")
    result = ph_bc_002.check(field, spec)
    assert result.status == "SKIPPED"
    assert result.reason is not None
    assert "Week 2" in result.reason


def test_ph_bc_002_heat_pde_is_skipped():
    spec = DomainSpec.model_validate(
        {
            "pde": "heat",
            "grid_shape": [16, 16, 4],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 0.1]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
            "diffusivity": 0.01,
        }
    )
    field = GridField(np.zeros((16, 16)), h=(1.0 / 15, 1.0 / 15), periodic=False)
    result = ph_bc_002.check(field, spec)
    assert result.status == "SKIPPED"
    assert "laplace/poisson only" in result.reason


def test_ph_bc_002_rejects_non_gridfield():
    import torch

    from physics_lint import CallableField

    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, 4),
            torch.linspace(0.0, 1.0, 4),
            indexing="ij",
        ),
        dim=-1,
    )
    field = CallableField(
        lambda x: (x[..., 0] ** 2 - x[..., 1] ** 2).unsqueeze(-1),
        sampling_grid=grid,
        h=(1.0 / 3, 1.0 / 3),
    )
    spec = _laplace_spec()
    with pytest.raises(TypeError, match="requires GridField"):
        ph_bc_002.check(field, spec)


def test_ph_bc_002_metadata():
    assert ph_bc_002.__rule_id__ == "PH-BC-002"
    assert ph_bc_002.__default_severity__ == "warning"
