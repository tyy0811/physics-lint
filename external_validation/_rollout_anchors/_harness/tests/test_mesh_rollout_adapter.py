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
    MGN_DATASET_SYSTEM_CLASS,
    MeshRollout,
    _expect_velocity,
    dissipation_sign_violation_on_mesh,
    energy_drift_on_mesh,
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


# ---------------------------------------------------------------------------
# Task 11 — MGN_DATASET_SYSTEM_CLASS + substrate-class dispatch (D0-23 v9)
# ---------------------------------------------------------------------------


def test_mgn_dataset_system_class_pins_vortex_shedding_2d() -> None:
    """D0-23 verdict 6: vortex_shedding_2d is pinned as open-driven-
    dissipative (boundary-driven sub-class) per the substrate-class
    smoke (preflight/substrate_class_smoke.json). Empirical-only
    classification per the design §2.2 "classify when you exercise"
    rule.
    """
    assert "vortex_shedding_2d" in MGN_DATASET_SYSTEM_CLASS
    assert MGN_DATASET_SYSTEM_CLASS["vortex_shedding_2d"] == "open-driven-dissipative"


def test_energy_drift_on_mesh_skips_when_open_driven_dissipative() -> None:
    """D0-23 verdict 9 (parallel to D0-22 amendment 1):
    energy_drift_on_mesh SKIPs with reason on open-driven-dissipative
    substrates. The strictly-dissipative-or-conservative assumption
    underpinning energy_drift fails on boundary-driven flows.
    """
    rollout = MeshRollout(
        node_positions=np.zeros((10, 2), dtype=np.float32),
        node_type=np.zeros(10, dtype=np.int64),
        # KE clears KE_REST_THRESHOLD so D0-08 does NOT fire; the
        # D0-23 substrate-class dispatch must fire instead.
        node_values={"velocity": 10 * np.ones((5, 10, 2), dtype=np.float32)},
        dt=0.01,
        metadata={
            "framework": "pytorch+dgl",
            "model": "modulus_ns_meshgraphnet",
            "dataset": "vortex_shedding_2d",
            # Force is_regular_grid=True so the graph-mesh SKIP doesn't
            # short-circuit before the substrate-class dispatch (which
            # is gated on Phase-2's DGL→MeshField materialization in
            # production code paths).
            "regular_grid": True,
        },
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )

    result = energy_drift_on_mesh(rollout)
    assert result.value is None, (
        "energy_drift_on_mesh must SKIP on open-driven-dissipative; "
        f"got value={result.value}, skip_reason={result.skip_reason}"
    )
    skip_reason = result.skip_reason or ""
    assert "open-driven-dissipative" in skip_reason, (
        f"SKIP reason must cite the substrate class; got: {skip_reason!r}"
    )
    assert "D0-22" in skip_reason or "D0-23" in skip_reason


def test_dissipation_sign_violation_on_mesh_skips_when_open_driven_dissipative() -> None:
    """D0-23 verdict 9 (parallel to D0-22 base gate):
    dissipation_sign_violation_on_mesh SKIPs with reason on open-driven-
    dissipative substrates. dE/dt > 0 over a stretch is physics, not a
    model violation.
    """
    rollout = MeshRollout(
        node_positions=np.zeros((10, 2), dtype=np.float32),
        node_type=np.zeros(10, dtype=np.int64),
        node_values={"velocity": 10 * np.ones((5, 10, 2), dtype=np.float32)},
        dt=0.01,
        metadata={
            "framework": "pytorch+dgl",
            "model": "modulus_ns_meshgraphnet",
            "dataset": "vortex_shedding_2d",
            "regular_grid": True,
        },
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )

    result = dissipation_sign_violation_on_mesh(rollout)
    assert result.value is None, (
        "dissipation_sign_violation_on_mesh must SKIP on open-driven-dissipative; "
        f"got value={result.value}, skip_reason={result.skip_reason}"
    )
    skip_reason = result.skip_reason or ""
    assert "open-driven-dissipative" in skip_reason
    assert "D0-22" in skip_reason or "D0-23" in skip_reason


def test_energy_drift_on_mesh_does_not_skip_when_no_substrate_class_match() -> None:
    """Regression guard: dispatch is dataset-name-conditioned. An unknown
    dataset name (or absence) must NOT trigger the substrate-class SKIP —
    the function proceeds to the KE-rest / compute path as before.
    """
    # Use a regular-grid synthetic fixture so we reach compute. Positions
    # must form a valid 2x5 grid so grid_shape inference succeeds.
    xs, ys = np.meshgrid(np.arange(2), np.arange(5), indexing="ij")
    positions = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    # Constant velocity → KE constant → drift = 0; passes through dispatch
    # path without firing any SKIP.
    velocity_t = np.ones((3, 10, 2), dtype=np.float32)

    rollout = MeshRollout(
        node_positions=positions,
        node_type=np.zeros(10, dtype=np.int64),
        node_values={"velocity": velocity_t},
        dt=0.01,
        metadata={
            "framework": "synthetic",
            "regular_grid": True,
            # Note: NO "dataset" key → MGN_DATASET_SYSTEM_CLASS.get("") is None
            # → dispatch does NOT fire.
        },
    )
    result = energy_drift_on_mesh(rollout)
    assert result.value is not None, (
        f"unknown dataset must not trigger substrate-class SKIP; "
        f"got skip_reason={result.skip_reason}"
    )
    # Constant velocity → drift = 0.
    assert result.value == 0.0
