"""External-validation anchor for PH-CON-003 - Evans Section 7.1.2 Theorem 2."""

from __future__ import annotations

import math

import numpy as np

from external_validation._harness.fixtures import unit_square_grid
from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_con_003

N = 64
H = 1.0 / (N - 1)
TIMESTEPS = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
DT = float(TIMESTEPS[1] - TIMESTEPS[0])
KAPPA = 0.1  # See CITATION.md "Pinned value". kappa=1.0 makes energies decay
# 2700x over the measurement window, breaking the rule's
# np.gradient(..., edge_order=2) endpoint extrapolation even
# though the analytical eigenmode is strictly dissipative.
EPS_QUAD = 1e-4
EXPECTED_RATIO = math.exp(-4.0 * KAPPA * math.pi**2 * DT)  # approx 0.820869


def _heat_eigenmode_u(t: float) -> np.ndarray:
    mesh_x, mesh_y = unit_square_grid(N)
    return (
        math.exp(-2.0 * KAPPA * math.pi**2 * t)
        * np.sin(math.pi * mesh_x)
        * np.sin(math.pi * mesh_y)
    )


def _heat_dirichlet_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "heat",
            "grid_shape": [N, N, len(TIMESTEPS)],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 0.2]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "diffusivity": KAPPA,
            "field": {"type": "grid", "backend": "fd", "dump_path": "h.npz"},
        }
    )


def _energy(u: np.ndarray, h: float) -> float:
    return 0.5 * float(np.trapz(np.trapz(u**2, dx=h, axis=-1), dx=h, axis=-1))


def test_fixture_analytical_heat_energy_ratio_matches_exp_minus_4kappa_pi2_dt():
    energies = [_energy(_heat_eigenmode_u(t), H) for t in TIMESTEPS]
    for k in range(len(TIMESTEPS) - 1):
        ratio = energies[k + 1] / energies[k]
        assert abs(ratio - EXPECTED_RATIO) < EPS_QUAD, (
            f"step {k}: ratio={ratio:.6f} expected={EXPECTED_RATIO:.6f}"
        )


def test_analytical_heat_sequence_passes_rule():
    u_seq = np.stack([_heat_eigenmode_u(t) for t in TIMESTEPS], axis=-1)
    field = GridField(u_seq, h=(H, H, DT), periodic=False)
    result = ph_con_003.check(field, _heat_dirichlet_spec())
    assert result.status == "PASS"


def test_non_dissipative_sequence_is_warn_or_fail():
    u0 = _heat_eigenmode_u(0.0)
    u_seq = np.stack([u0 * (1.0 + 2.0 * float(t) / 0.2) for t in TIMESTEPS], axis=-1)
    field = GridField(u_seq, h=(H, H, DT), periodic=False)
    result = ph_con_003.check(field, _heat_dirichlet_spec())
    assert result.status in {"WARN", "FAIL"}
