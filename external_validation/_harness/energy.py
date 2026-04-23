"""Conservation-law energy helpers for Tasks 8 and 9.

Stub populated across Tasks 8 and 9 (shared primitives per complete-v1.0 plan
§3 `energy.py [new, Tasks 8, 9]`).

Planned surface area:
- **Task 8 (PH-CON-001, heat mass conservation):**
  - `mass_integral(u, dx) -> float`: `∫_Ω u dV` via trapezoidal quadrature,
    periodic or Dirichlet-aware.
  - `mass_conservation_error(u_sequence, dx) -> np.ndarray`: per-step deviation
    of `∫ u(t_k, x) dx` from `∫ u(t_0, x) dx`.
  - Evans §2.3 + Dafermos Ch I balance-law framing.
- **Task 9 (PH-CON-002, wave energy conservation):**
  - `wave_energy(u, u_t, dx, c) -> float`: `½ ∫_Ω (u_t² + c² |∇u|²) dV` via
    trapezoidal quadrature on periodic domain.
  - `energy_conservation_ratio(u_sequence, u_t_sequence, dx, c) -> np.ndarray`:
    per-step `E(t_k) / E(t_0)`.
  - Evans §2.4.3 + Strauss §2.2 + Hairer-Lubich-Wanner Ch IX (symplectic
    leapfrog conservation).

No implementation yet -- Tasks 8 and 9 populate this module in their execution
phase.
"""

from __future__ import annotations
