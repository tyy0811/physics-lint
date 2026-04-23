"""Trace-norm helpers for boundary-condition anchor fixtures (Task 4).

Stub populated in Task 4 (PH-BC-001). Consumers: `external_validation/PH-BC-001/test_anchor.py`.

Planned surface area (complete-v1.0 plan §12 Task 4):
- `trace_l2_norm(u, boundary_mask) -> float`: L² norm of `u` restricted to
  discrete boundary points -- the H^{1/2} trace-space proxy used by PH-BC-001's
  emitted bRMSE proxy.
- `bc_violation_rmse(u_pred, u_bc_prescribed, boundary_mask) -> float`:
  RMS deviation at boundary points. Compatible with the Evans §5.5 Thm 1
  trace-theorem precondition (H¹ domain function on Lipschitz boundary).

No implementation yet -- see Task 4 execution plan §12 for the fixture design.
"""

from __future__ import annotations
