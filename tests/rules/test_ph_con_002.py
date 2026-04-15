"""PH-CON-002 — wave energy conservation tests."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.analytical import wave as wave_sols
from physics_lint.rules import ph_con_002


def _wave_hd_spec(n: int, nt: int) -> DomainSpec:
    return DomainSpec.model_validate(
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


def test_ph_con_002_standing_wave_conserves_energy():
    n, nt = 64, 32
    spec = _wave_hd_spec(n, nt)
    sol = wave_sols.standing_wave_square(c=1.0)
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 0.5, nt)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    pred = np.stack([sol.u(X, Y, ti) for ti in t], axis=-1)
    field = GridField(pred, h=(1 / (n - 1), 1 / (n - 1), 0.5 / (nt - 1)), periodic=False)
    result = ph_con_002.check(field, spec)
    assert result.rule_id == "PH-CON-002"
    assert result.status == "PASS"


def test_ph_con_002_growing_amplitude_fails():
    n, nt = 32, 16
    spec = _wave_hd_spec(n, nt)
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 0.5, nt)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    base = np.sin(np.pi * X) * np.sin(np.pi * Y)
    pred = np.stack(
        [base * (1.0 + 1.0 * ti) * np.cos(np.pi * np.sqrt(2.0) * ti) for ti in t],
        axis=-1,
    )
    field = GridField(pred, h=(1 / (n - 1), 1 / (n - 1), 0.5 / (nt - 1)), periodic=False)
    result = ph_con_002.check(field, spec)
    assert result.status in {"WARN", "FAIL"}
