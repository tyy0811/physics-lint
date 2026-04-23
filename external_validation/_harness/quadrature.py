"""FEM quadrature-order convergence helpers for Task 11 (PH-NUM-001).

Stub populated in Task 11. Consumers: `external_validation/PH-NUM-001/test_anchor.py`.

Planned surface area (complete-v1.0 plan §19 Task 11):
- `observed_rate(hs, errs) -> float`: log-log regression slope, requires
  positive `hs` and `errs` (Category 5 precondition).
- `mms_intorder_sweep(p, intorder, N_values, mms_solution) -> np.ndarray`:
  error array over an h-refinement sweep at fixed FE polynomial order `p` and
  quadrature order `intorder`, using a smooth manufactured solution.
- Predicted-rate accessor: `ciarlet_predicted_rate(p) = p + 1` for conforming
  FE with sufficient quadrature (Ciarlet §4.1 Thms 4.1.2-4.1.6).

No implementation yet -- Task 11 execution populates this module.
"""

from __future__ import annotations
