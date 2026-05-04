"""Mesh-side rollout adapter for `_rollout_anchors/_harness`.

Per spec §3.1, this module materializes one timestep of a mesh rollout
into a Field-API-compatible object so the existing public `physics-lint
check` CLI can consume it without rule modification:

- Gate A PASS (preferred): `MeshField(basis=reconstructed_basis,
  dofs=node_values_at_t)` if the DGL graph can be coerced to a
  scikit-fem `Basis`.

- Gate A PARTIAL (fallback): `GridField(values=resampled, h=spacing,
  periodic=False)` after a documented regular-grid resampling pass.

- Gate A FAIL: this module emits SARIF directly via the harness path
  (no public-API materialization); the cross-stack-via-public-API
  claim is dropped per spec §6 / §1.4.

Day-0 scope: a small public surface (`materialize_grid_field`) for the
fixture validation tests, which already operate on regular grids and
therefore exercise the GridField path of this adapter. The DGL-side
materialization (`materialize_mesh_field`) lands on Day 2 once the
NGC checkpoint Q1 audit produces a real PhysicsNeMo sample to test
against; Day 0 deliberately does not write speculative DGL code per
the executing agent's "no speculative stubs" rule.
"""

from __future__ import annotations

import numpy as np

from physics_lint import GridField


def materialize_grid_field(
    values: np.ndarray,
    *,
    h: float | tuple[float, ...],
    periodic: bool = False,
    backend: str = "fd",
) -> GridField:
    """Wrap a numpy array of per-timestep node values as a GridField.

    Used by the FNO-on-Darcy fallback (Gate D), where the model output
    is already on a regular grid, and by the GridField PARTIAL fallback
    of Gate A after a resampling pass.

    The function is intentionally a thin wrapper — its job is to make
    the materialization step explicit in the call graph so a code
    reviewer can see "this is the public-API entry point for the mesh
    case study", not to add behaviour over `GridField.__init__`.
    """
    return GridField(values, h=h, periodic=periodic, backend=backend)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DGL → MeshField materialization
# ---------------------------------------------------------------------------
#
# Deliberately not implemented in the Day 0 scaffold. This path requires:
#
#   1. A real PhysicsNeMo NGC sample timestep (Audit Q1 / Gate A) so the
#      DGL-graph-to-scikit-fem-basis coercion is exercised against actual
#      output, not a synthetic graph.
#   2. The `nvidia-physicsnemo` and `dgl` dependencies, which live behind
#      the `[validation-rollout]` extra and are not installed on Day 0.
#   3. A confirmed Gate A verdict (PASS / PARTIAL) — under FAIL, this
#      function is never called.
#
# Per the executing agent's "no speculative stubs" rule, the function lands
# in a separate Day 2 commit once Gate A returns a verdict. Until then,
# callers requesting it will receive an `AttributeError` from this module
# — that is the intended behaviour.
