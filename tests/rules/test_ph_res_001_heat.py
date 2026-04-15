"""PH-RES-001 heat path — Bochner L^2(H^-1) residual on periodic eigenfunction."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.analytical import heat as heat_sols
from physics_lint.rules import ph_res_001


def _heat_periodic_spec(n: int, nt: int) -> DomainSpec:
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


def test_ph_res_001_heat_periodic_analytical_is_pass():
    n, nt = 64, 16
    spec = _heat_periodic_spec(n, nt)
    sol = heat_sols.periodic_cos_cos(kappa=0.01)

    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    t = np.linspace(0.0, 0.5, nt)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    pred = np.stack([sol.u(X, Y, ti) for ti in t], axis=-1)
    h = (2 * np.pi / n, 2 * np.pi / n, 0.5 / (nt - 1))
    field = GridField(pred, h=h, periodic=True, backend="spectral")

    result = ph_res_001.check(field, spec)
    assert result.rule_id == "PH-RES-001"
    assert result.status == "PASS"
    assert result.recommended_norm.startswith("Bochner")


def test_ph_res_001_heat_hd_analytical_is_pass_or_warn():
    # Under hD the spatial Laplacian via FD picks up boundary-quantization
    # error; the residual won't be machine-precision but should still be
    # close to the shipped floor. Accept PASS or WARN here — real floor
    # calibration happens in Task 7.
    n, nt = 64, 16
    spec = _heat_hd_spec(n, nt)
    sol = heat_sols.eigenfunction_decay_square(kappa=0.01)
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 0.5, nt)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    pred = np.stack([sol.u(X, Y, ti) for ti in t], axis=-1)
    h = (1.0 / (n - 1), 1.0 / (n - 1), 0.5 / (nt - 1))
    field = GridField(pred, h=h, periodic=False, backend="fd")
    result = ph_res_001.check(field, spec)
    assert result.rule_id == "PH-RES-001"
    assert result.status in {"PASS", "WARN"}
    assert result.recommended_norm.startswith("Bochner")


def test_ph_res_001_heat_nonzero_residual_is_warn_or_fail():
    n, nt = 64, 16
    spec = _heat_periodic_spec(n, nt)
    # Use sin(x) sin(y) * (1 + 0.1 t) — NOT a heat solution: d/dt = 0.1*sin*sin,
    # while kappa*lap = 0.01 * (-2 sin sin (1 + 0.1 t)) ≈ -0.02 sin sin. Clear
    # mismatch drives residual well above the floor.
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    t = np.linspace(0.0, 0.5, nt)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    pred = np.stack([np.sin(X) * np.sin(Y) * (1 + 0.1 * ti) for ti in t], axis=-1)
    h = (2 * np.pi / n, 2 * np.pi / n, 0.5 / (nt - 1))
    field = GridField(pred, h=h, periodic=True, backend="spectral")

    result = ph_res_001.check(field, spec)
    assert result.status in {"WARN", "FAIL"}
