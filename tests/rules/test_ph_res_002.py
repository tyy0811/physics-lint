"""PH-RES-002 — FD-vs-AD cross-check. Adapter-only; dump mode SKIPs."""

import numpy as np
import torch

from physics_lint import CallableField, DomainSpec, GridField
from physics_lint.rules import ph_res_002


def _laplace_spec(periodic: bool = False) -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [16, 16],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": periodic,
            "boundary_condition": {"kind": "periodic" if periodic else "dirichlet"},
            "field": {
                "type": "grid" if periodic else "callable",
                "backend": "fd",
                "dump_path" if periodic else "adapter_path": "x",
            },
        }
    )


def test_ph_res_002_dump_mode_skipped():
    spec = _laplace_spec(periodic=True)
    u = np.zeros((16, 16))
    field = GridField(u, h=1.0 / 16, periodic=True)
    result = ph_res_002.check(field, spec)
    assert result.status == "SKIPPED"
    assert result.reason and "callable" in result.reason.lower()


def test_ph_res_002_adapter_mode_quadratic_zero_discrepancy():
    spec = _laplace_spec(periodic=False)
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, 16),
            torch.linspace(0.0, 1.0, 16),
            indexing="ij",
        ),
        dim=-1,
    )

    def model(x):
        return (x[..., 0] ** 2 + x[..., 1] ** 2).unsqueeze(-1)

    field = CallableField(model, sampling_grid=grid, h=(1.0 / 15, 1.0 / 15))
    result = ph_res_002.check(field, spec)
    assert result.status in {"PASS", "WARN"}  # small FD error at edges
    assert result.raw_value is not None
    assert result.raw_value < 0.1  # 10% discrepancy tolerance


def test_ph_res_002_adapter_mode_3d_quadratic():
    # Cover the 3D interior-slicing branch with a smooth quadratic field.
    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [8, 8, 8],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "field": {"type": "callable", "backend": "fd", "adapter_path": "x"},
        }
    )
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, 8),
            torch.linspace(0.0, 1.0, 8),
            torch.linspace(0.0, 1.0, 8),
            indexing="ij",
        ),
        dim=-1,
    )

    def model(x):
        return (x[..., 0] ** 2 + x[..., 1] ** 2 + x[..., 2] ** 2).unsqueeze(-1)

    field = CallableField(model, sampling_grid=grid, h=(1.0 / 7, 1.0 / 7, 1.0 / 7))
    result = ph_res_002.check(field, spec)
    assert result.status in {"PASS", "WARN"}
    assert result.raw_value is not None
    assert result.raw_value < 0.1
