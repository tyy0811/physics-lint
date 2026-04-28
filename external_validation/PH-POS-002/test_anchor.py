"""External-validation anchor for PH-POS-002 - Evans section 2.2.3 Theorem 4."""

from __future__ import annotations

import numpy as np

from external_validation._harness.fixtures import (
    AnalyticalField,
    harmonic_cubic,
    harmonic_xx_yy,
    harmonic_xy,
    unit_square_grid,
)
from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_pos_002

N = 64
H = 1.0 / (N - 1)


def _laplace_dirichlet_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [N, N],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )


def _run_rule(field_fixture: AnalyticalField) -> tuple[float, str]:
    mesh_x, mesh_y = unit_square_grid(N)
    u = field_fixture.u(mesh_x, mesh_y)
    field = GridField(u, h=(H, H), periodic=False)
    boundary = field.values_on_boundary()
    result = ph_pos_002.check(field, _laplace_dirichlet_spec(), boundary_values=boundary)
    return float(result.raw_value or 0.0), result.status


def _run_rule_with_injected_overshoot() -> tuple[float, str]:
    u = np.zeros((N, N))
    u[N // 2, N // 2] = 5.0
    field = GridField(u, h=(H, H), periodic=False)
    result = ph_pos_002.check(
        field, _laplace_dirichlet_spec(), boundary_values=field.values_on_boundary()
    )
    return float(result.raw_value or 0.0), result.status


def test_harmonic_xx_yy_passes():
    _, status = _run_rule(harmonic_xx_yy())
    assert status == "PASS"


def test_harmonic_xy_passes():
    _, status = _run_rule(harmonic_xy())
    assert status == "PASS"


def test_harmonic_cubic_passes():
    _, status = _run_rule(harmonic_cubic())
    assert status == "PASS"


def test_non_harmonic_injected_overshoot_fails():
    raw, status = _run_rule_with_injected_overshoot()
    assert status == "FAIL"
    assert raw > 0
