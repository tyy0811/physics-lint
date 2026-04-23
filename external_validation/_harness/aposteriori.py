"""A-posteriori residual-based error indicators for Task 10 (PH-CON-004).

Stub populated in Task 10. Consumers: `external_validation/PH-CON-004/test_anchor.py`.

Planned surface area (complete-v1.0 plan §18 Task 10):
- `residual_indicator(u_h, mesh, f, a=1.0) -> np.ndarray`: per-element residual-
  based η(τ, u_h) indicator for Poisson `-∇·(a∇u) = f` on a triangulated mesh.
  `η(τ, u_T)² = ||h f||²_{0,τ} + Σ_e ||h^{1/2} [∇u_T·n_e]||²_{0,e}` --
  Verfürth 2013 Thm 1.12 / Chs 1-4.
- Hotspot-concentration helpers: given the indicator array, verify the top-k
  elements concentrate at known singularity locations (L-corner).

No implementation yet -- scikit-fem Example 22 adaptive-Poisson fixture is
the Task 10 execution target.
"""

from __future__ import annotations
