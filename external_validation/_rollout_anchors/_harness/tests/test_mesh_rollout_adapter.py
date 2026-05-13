"""Tests for the mesh rollout adapter's loader-contract helpers.

Lands the Phase-1D code-absorption coverage for case study 02:

- Task 10 (D0-23 v8): ``_expect_velocity`` resolves the NGC vortex-
  shedding velocity-field key. Pattern-B P0 single-instance enumeration:
  the audit (preflight/loader_contract_audit.json V3) confirms the NGC
  key is ``"velocity"`` — same as the legacy LB / synthetic path — so
  no pre-generalization (no helper key list, no metadata pivot). A
  second NGC naming (amendment 1's Ahmed Body) triggers a refactor;
  one instance does not.

- Task 11 (D0-23 v9): ``MGN_DATASET_SYSTEM_CLASS`` and the substrate-
  class dispatch on ``energy_drift_on_mesh`` /
  ``dissipation_sign_violation_on_mesh``. Mirrors the particle-side
  D0-22 dispatch (parallel route, duplicate-logic-drift risk *named*
  in round-codex-4 — not eliminated; stack-agnostic refactor gated
  on case study 03 evidence).

- Task 12 (D0-23 v10): ``_assert_loader_contract_mgn`` defensive
  validation per preflight V1-V18 + the 5 secondary known-unknowns.
  Fires on incoming MGN rollouts before rule kernels consume them.
"""

from __future__ import annotations

import numpy as np

from external_validation._rollout_anchors._harness.mesh_rollout_adapter import (
    MeshRollout,
    _expect_velocity,
)

# ---------------------------------------------------------------------------
# Task 10 — _expect_velocity NGC key resolution (D0-23 verdict 8)
# ---------------------------------------------------------------------------


def test_expect_velocity_resolves_ngc_velocity_key_for_vortex_shedding() -> None:
    """D0-23 verdict 8: the NGC cylinder_flow dataset emits node-resolved
    velocity under the literal key ``"velocity"`` (preflight
    loader_contract_audit.json V3_field_names: cells, mesh_pos, node_type,
    velocity, pressure). Pattern-B P0 single-instance enumeration —
    confirms the legacy default is the correct NGC key; no helper
    branching required.
    """
    rollout = MeshRollout(
        node_positions=np.zeros((10, 2), dtype=np.float32),
        node_type=np.zeros(10, dtype=np.int64),
        node_values={"velocity": np.ones((5, 10, 2), dtype=np.float32)},
        dt=0.01,
        metadata={
            "framework": "pytorch+dgl",
            "model": "modulus_ns_meshgraphnet",
            "dataset": "vortex_shedding_2d",
        },
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )

    velocity = _expect_velocity(rollout)
    assert isinstance(velocity, np.ndarray), (
        f"_expect_velocity must resolve the NGC velocity key 'velocity' "
        f"(per D0-23 v8); got {type(velocity)}"
    )
    assert velocity.shape == (5, 10, 2)


def test_expect_velocity_still_accepts_legacy_synthetic_path() -> None:
    """Pattern-B P0 discipline: the helper accepts the literal
    ``"velocity"`` key regardless of framework metadata. Single-instance
    enumeration — amendment 1's Ahmed Body is the multi-instance trigger
    that would force a key-list / metadata-pivot refactor.
    """
    rollout = MeshRollout(
        node_positions=np.zeros((10, 2), dtype=np.float32),
        node_type=np.zeros(10, dtype=np.int64),
        node_values={"velocity": np.ones((5, 10, 2), dtype=np.float32)},
        dt=0.01,
        metadata={"framework": "synthetic"},
    )
    velocity = _expect_velocity(rollout)
    assert isinstance(velocity, np.ndarray)
    assert velocity.shape == (5, 10, 2)
