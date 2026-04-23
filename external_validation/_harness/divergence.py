"""Gauss-Green / divergence-theorem helpers for Task 5 (PH-BC-002).

Stub populated in Task 5. Consumers: `external_validation/PH-BC-002/test_anchor.py`.

Planned surface area (complete-v1.0 plan §13 Task 5):
- `divergence_integral_on_triangulation(field_fn, mesh) -> float`: `∫_Ω ∇·F dV`
  computed on a triangulation.
- `flux_integral_on_boundary(field_fn, mesh) -> float`: `∫_{∂Ω} F·n dS` on the
  same mesh's boundary edges.
- The two should agree exactly (up to triangulation-dependent quadrature
  error) for a C¹ vector field on a Lipschitz-boundary convex polygon --
  Evans Appendix C.2 Gauss-Green Theorem 1 + Gilbarg-Trudinger §2.4.

No implementation yet -- see Task 5 execution plan §13 for the `F=(x,y)` on
unit-square fixture design (analytical integral = 2).
"""

from __future__ import annotations
