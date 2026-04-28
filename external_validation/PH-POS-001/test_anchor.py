"""PH-POS-001 external-validation anchor - Evans positivity theorems.

Pre-execution audit: see AUDIT.md in this directory for the enumerate-the-
splits verification (discrete-predicate rule, single code path post-gate,
no semantic-compatibility concerns).

External references (full provenance in CITATION.md):
- Evans, *Partial Differential Equations*, AMS GSM 19 (2010), Section 2.2.3
  Theorem 4 + Positivity corollary (p. 27) for the Poisson case.
- Evans, same book, Section 2.3.3 Theorem 4 (strong maximum principle for
  the heat equation) for the heat case.
"""

from __future__ import annotations

import math

import numpy as np

from external_validation._harness.fixtures import unit_square_grid
from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_pos_001

N = 64
H = 1.0 / (N - 1)
TIMESTEPS_HEAT = [0.0, 0.02, 0.05, 0.1]
DT_HEAT = TIMESTEPS_HEAT[1] - TIMESTEPS_HEAT[0]


# -- Poisson fixture: u = x(1-x)y(1-y), non-negative polynomial ----------


def _poisson_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [N, N],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )


def _poisson_polynomial() -> np.ndarray:
    mesh_x, mesh_y = unit_square_grid(N)
    return mesh_x * (1 - mesh_x) * mesh_y * (1 - mesh_y)


def test_poisson_polynomial_passes_positivity():
    u = _poisson_polynomial()
    field = GridField(u, h=(H, H), periodic=False)
    result = ph_pos_001.check(field, _poisson_spec())
    assert result.status == "PASS"


def test_poisson_polynomial_has_nonneg_interior_and_zero_boundary():
    u = _poisson_polynomial()
    # Mathematical precondition: confirm analytical shape matches the
    # theorem's hypothesis before running the rule.
    assert np.all(u[1:-1, 1:-1] > 0)
    assert np.allclose(u[0, :], 0.0)
    assert np.allclose(u[-1, :], 0.0)
    assert np.allclose(u[:, 0], 0.0)
    assert np.allclose(u[:, -1], 0.0)


# -- Heat fixture: periodic eigenmode with positive offset ---------------
# Axis convention: time LAST (grid_shape=[N_x, N_y, N_t], h=(H, H, DT_HEAT)),
# matches ph_con_003.py / existing heat test conventions.
# diffusivity=1.0 satisfies d_t u = Laplacian u for the eigenmode component.


def _heat_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "heat",
            "grid_shape": [N, N, len(TIMESTEPS_HEAT)],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 0.1]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "diffusivity": 1.0,
            "field": {"type": "grid", "backend": "fd", "dump_path": "h.npz"},
        }
    )


def _heat_analytical(t: float) -> np.ndarray:
    mesh_x, mesh_y = unit_square_grid(N)
    return 1.0 + 0.5 * math.exp(-8.0 * math.pi**2 * t) * np.sin(2 * math.pi * mesh_x) * np.sin(
        2 * math.pi * mesh_y
    )


def test_heat_analytical_passes_positivity_all_timesteps():
    u_seq = np.stack([_heat_analytical(t) for t in TIMESTEPS_HEAT], axis=-1)
    assert u_seq.min() > 0
    field = GridField(u_seq, h=(H, H, DT_HEAT), periodic=True)
    result = ph_pos_001.check(field, _heat_spec())
    assert result.status == "PASS"


# -- Negative control: injected spike ------------------------------------


def test_injected_negative_spike_fails():
    u = _poisson_polynomial()
    cy, cx = N // 2, N // 2
    u[cy - 2 : cy + 3, cx - 2 : cx + 3] -= 0.8
    assert u.min() < -0.7
    field = GridField(u, h=(H, H), periodic=False)
    result = ph_pos_001.check(field, _poisson_spec())
    assert result.status == "FAIL"
    assert result.raw_value is not None and result.raw_value < 0
