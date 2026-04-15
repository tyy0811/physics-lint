"""PH-CON-001 — heat mass conservation: PER/hN exact, hD rate consistency."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.analytical import heat as heat_sols
from physics_lint.rules import ph_con_001


def _periodic_spec(n: int, nt: int) -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "heat",
            "grid_shape": [n, n, nt],
            "domain": {"x": [0.0, 2 * np.pi], "y": [0.0, 2 * np.pi], "t": [0.0, 0.5]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "diffusivity": 0.01,
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )


def _hd_spec(n: int, nt: int) -> DomainSpec:
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


def test_ph_con_001_periodic_conserves_mass():
    # cos(x)*cos(y) has zero mass over [0, 2 pi]^2 at all t; the rule should PASS
    n, nt = 64, 16
    spec = _periodic_spec(n, nt)
    sol = heat_sols.periodic_cos_cos(kappa=0.01)
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    t = np.linspace(0.0, 0.5, nt)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    pred = np.stack([sol.u(X, Y, ti) for ti in t], axis=-1)
    field = GridField(pred, h=(2 * np.pi / n, 2 * np.pi / n, 0.5 / (nt - 1)), periodic=True)

    result = ph_con_001.check(field, spec)
    assert result.rule_id == "PH-CON-001"
    assert result.status == "PASS"
    assert result.mode == "exact-mass"


def test_ph_con_001_periodic_mass_drift_fails():
    n, nt = 64, 16
    spec = _periodic_spec(n, nt)
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    X, _Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    pred = np.stack(
        [np.ones_like(X) * (1.0 + 0.5 * k / (nt - 1)) for k in range(nt)],
        axis=-1,
    )
    field = GridField(pred, h=(2 * np.pi / n, 2 * np.pi / n, 0.5 / (nt - 1)), periodic=True)

    result = ph_con_001.check(field, spec)
    assert result.status in {"WARN", "FAIL"}
    assert result.mode == "exact-mass"


def test_ph_con_001_hd_rate_consistency_pass():
    # Eigenfunction decay on [0,1]^2 hD: dM/dt = kappa * integral(lap u) exactly.
    n, nt = 64, 32
    spec = _hd_spec(n, nt)
    sol = heat_sols.eigenfunction_decay_square(kappa=0.01)
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 0.5, nt)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    pred = np.stack([sol.u(X, Y, ti) for ti in t], axis=-1)
    field = GridField(pred, h=(1 / (n - 1), 1 / (n - 1), 0.5 / (nt - 1)), periodic=False)

    result = ph_con_001.check(field, spec)
    assert result.status == "PASS"
    assert result.mode == "rate-consistency"
