"""PH-CON-003 — heat energy dissipation sign violation."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.analytical import heat as heat_sols
from physics_lint.rules import ph_con_003


def _heat_hd_spec(n: int, nt: int) -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "heat",
            "grid_shape": [n, n, nt],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 0.5]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "diffusivity": 0.01,
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )


def test_ph_con_003_decaying_energy_passes():
    n, nt = 32, 16
    spec = _heat_hd_spec(n, nt)
    sol = heat_sols.eigenfunction_decay_square(kappa=0.01)
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 0.5, nt)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    pred = np.stack([sol.u(X, Y, ti) for ti in t], axis=-1)
    field = GridField(pred, h=(1 / (n - 1), 1 / (n - 1), 0.5 / (nt - 1)), periodic=False)
    result = ph_con_003.check(field, spec)
    assert result.rule_id == "PH-CON-003"
    assert result.status == "PASS"


def test_ph_con_003_energy_growth_is_warn():
    n, nt = 32, 16
    spec = _heat_hd_spec(n, nt)
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    base = np.sin(np.pi * X) * np.sin(np.pi * Y)
    pred = np.stack(
        [base * (1.0 + 0.5 * k / (nt - 1)) for k in range(nt)],
        axis=-1,
    )
    field = GridField(pred, h=(1 / (n - 1), 1 / (n - 1), 0.5 / (nt - 1)), periodic=False)
    result = ph_con_003.check(field, spec)
    assert result.status in {"WARN", "FAIL"}


def test_ph_con_003_wave_pde_is_skipped():
    n, nt = 32, 16
    spec = DomainSpec.model_validate(
        {
            "pde": "wave",
            "grid_shape": [n, n, nt],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 0.5]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "wave_speed": 1.0,
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 0.5, nt)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    pred = np.stack(
        [np.sin(np.pi * X) * np.sin(np.pi * Y) * np.cos(np.pi * np.sqrt(2) * ti) for ti in t],
        axis=-1,
    )
    field = GridField(pred, h=(1 / (n - 1), 1 / (n - 1), 0.5 / (nt - 1)), periodic=False)
    result = ph_con_003.check(field, spec)
    assert result.status == "SKIPPED"
