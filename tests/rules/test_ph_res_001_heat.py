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


def test_ph_res_001_heat_periodic_analytical_is_pass_with_bochner_h_minus_one():
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
    # Periodic+spectral path uses the variationally correct H^-1 Bochner
    # norm (exact, up to FFT roundoff + time-derivative truncation error).
    assert result.recommended_norm == "Bochner-H-1"


def test_ph_res_001_heat_hd_analytical_is_pass_or_warn_with_bochner_l2():
    # hD is non-periodic: falls back to Bochner-L2. The eigenfunction-decay
    # solution has no DC content, so the fallback residual is dominated by
    # 4th-order FD truncation in the spatial Laplacian plus 2nd-order time
    # FD truncation — small enough for PASS/WARN at the shipped floor.
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
    assert result.recommended_norm == "Bochner-L2"


def test_ph_res_001_heat_hd_constant_in_time_is_warn_or_fail():
    """Regression for Finding 1 (non-periodic DC false-pass).

    u(x, y, t) = t on hD is not a heat solution: u_t = 1 everywhere,
    kappa * lap u = 0, so the residual is a constant 1 in space and time.
    Bochner-L2 picks up the full 1 * area * T = 0.5 contribution; the
    old periodic H^-1 path dropped the DC mode and returned 0 -> PASS.
    """
    n, nt = 32, 8
    spec = _heat_hd_spec(n, nt)
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 0.5, nt)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    # Broadcast the scalar time across the spatial slice.
    pred = np.stack(
        [np.full_like(X, fill_value=ti) for ti in t],
        axis=-1,
    ).astype(float)
    del Y  # unused in the non-solution construction
    h = (1.0 / (n - 1), 1.0 / (n - 1), 0.5 / (nt - 1))
    field = GridField(pred, h=h, periodic=False, backend="fd")

    result = ph_res_001.check(field, spec)
    # Bochner-L2 of constant-1 residual over [0,1]^2 x [0,0.5] is sqrt(0.5);
    # that's ~7e4 above the shipped 1e-5 floor, so status must not be PASS.
    assert result.status in {"WARN", "FAIL"}
    assert result.recommended_norm == "Bochner-L2"
    assert result.raw_value is not None and result.raw_value > 0.1


def test_ph_res_001_heat_nt_too_small_is_skipped():
    """Finding 4: Nt=2 would crash np.gradient(edge_order=2) — now SKIPPED."""
    spec = _heat_hd_spec(16, 2)
    # Construct a valid (16, 16, 2) dump-like prediction — anything will do
    # since the rule should SKIP before reaching np.gradient.
    pred = np.zeros((16, 16, 2))
    h = (1.0 / 15, 1.0 / 15, 0.5 / 1)
    field = GridField(pred, h=h, periodic=False, backend="fd")
    result = ph_res_001.check(field, spec)
    assert result.status == "SKIPPED"
    assert "time samples" in (result.reason or "").lower()


def test_ph_res_001_wave_nt_too_small_is_skipped():
    spec = DomainSpec.model_validate(
        {
            "pde": "wave",
            "grid_shape": [16, 16, 2],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 0.1]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "wave_speed": 1.0,
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    field = GridField(np.zeros((16, 16, 2)), h=(1 / 15, 1 / 15, 0.1), periodic=False)
    result = ph_res_001.check(field, spec)
    assert result.status == "SKIPPED"
    assert "time samples" in (result.reason or "").lower()


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
