"""PH-RES-003 — Spectral-vs-FD discrepancy on periodic grids."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_res_003


def _periodic_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [64, 64],
            "domain": {"x": [0.0, 2 * np.pi], "y": [0.0, 2 * np.pi]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )


def test_ph_res_003_smooth_periodic_passes():
    n = 64
    h = 2 * np.pi / n
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    xg, yg = np.meshgrid(x, x, indexing="ij")
    u = np.sin(xg) * np.sin(yg)
    field = GridField(u, h=h, periodic=True)
    result = ph_res_003.check(field, _periodic_spec())
    assert result.status == "PASS"


def test_ph_res_003_skipped_on_nonperiodic():
    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [32, 32],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    u = np.zeros((32, 32))
    field = GridField(u, h=1.0 / 31, periodic=False)
    result = ph_res_003.check(field, spec)
    assert result.status == "SKIPPED"
    assert result.reason and "periodic" in result.reason.lower()


def test_ph_res_003_accepts_callable_field_periodic():
    """Adapter-mode periodic: PH-RES-003 materializes the callable and
    runs the spectral-vs-FD cross-check on the sampled values. sin(x)
    is exact under both backends so the relative difference is below
    the rule's PASS threshold."""
    import torch

    from physics_lint import CallableField

    n = 32
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 2 * np.pi, n + 1)[:-1],
            torch.linspace(0.0, 2 * np.pi, n + 1)[:-1],
            indexing="ij",
        ),
        dim=-1,
    )
    field = CallableField(
        lambda x: torch.sin(x[..., 0]).unsqueeze(-1),
        sampling_grid=grid,
        h=(2 * np.pi / n, 2 * np.pi / n),
        periodic=True,
    )
    result = ph_res_003.check(field, _periodic_spec())
    assert result.status == "PASS"
