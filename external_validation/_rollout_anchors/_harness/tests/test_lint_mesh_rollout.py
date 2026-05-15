"""Unit tests for :func:`_harness.lint_mesh_rollout.lint_mesh_rollout`.

Smoke coverage: three HarnessResult rows per call, rule_id naming
matches the LB-side particle lint convention, ke_initial/ke_final
attaches to harness:energy_drift on SKIP, substrate-class dispatch
fires through the helper.
"""

from __future__ import annotations

import numpy as np

from external_validation._rollout_anchors._harness.lint_mesh_rollout import (
    lint_mesh_rollout,
)
from external_validation._rollout_anchors._harness.mesh_rollout_adapter import (
    MeshRollout,
)


def _make_synthetic_grid_rollout(*, dataset: str = "synthetic") -> MeshRollout:
    """3-frame 4x4 uniform-grid rollout with constant velocity = (1, 0).
    Mass-conservation defect on the FD path = ~machine epsilon (constant
    field is exactly divergence-free); KE constant -> drift = 0,
    dissipation_sign_violation = 0.
    """
    n = 4
    xs, ys = np.meshgrid(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n), indexing="ij")
    positions = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    n_nodes = positions.shape[0]
    v0 = np.stack([np.ones(n_nodes, dtype=np.float32), np.zeros(n_nodes, dtype=np.float32)], axis=1)
    velocity = np.tile(v0[None, ...], (3, 1, 1))
    return MeshRollout(
        node_positions=positions,
        node_type=np.zeros(n_nodes, dtype=np.int64),
        node_values={"velocity": velocity},
        dt=0.01,
        metadata={
            "framework": "synthetic",
            "model": "synthetic",
            "dataset": dataset,
            "regular_grid": True,  # use the FD path (no FE needed)
        },
        edge_index=None,
    )


def test_lint_mesh_rollout_emits_three_rows_one_per_rule() -> None:
    """Three rule mirrors -> three HarnessResult rows in fixed order."""
    rollout = _make_synthetic_grid_rollout()
    results = lint_mesh_rollout(
        rollout,
        case_study="02-physicsnemo-mgn",
        dataset="synthetic",
        model="synthetic",
        ckpt_hash="n/a",
    )
    assert [r.rule_id for r in results] == [
        "harness:mass_conservation_defect",
        "harness:energy_drift",
        "harness:dissipation_sign_violation",
    ]
    # Constant velocity -> all three values are numeric and close to 0.
    for r in results:
        assert r.raw_value is not None
        assert abs(r.raw_value) < 1e-6


def test_lint_mesh_rollout_propagates_extra_properties() -> None:
    """Caller-supplied ``extra_properties`` lands on every row."""
    rollout = _make_synthetic_grid_rollout()
    results = lint_mesh_rollout(
        rollout,
        case_study="02-physicsnemo-mgn",
        dataset="vortex_shedding_2d",
        model="deepmind-cylinder-flow-gt",
        ckpt_hash="n/a_gt",
        extra_properties={"trajectory_index": 44, "arm": "gt-control"},
    )
    for r in results:
        assert r.extra_properties["trajectory_index"] == 44
        assert r.extra_properties["arm"] == "gt-control"


def test_lint_mesh_rollout_emits_skip_reason_on_substrate_class_dispatch() -> None:
    """When energy_drift / dissipation_sign_violation SKIP via D0-23 v9,
    the row's ``skip_reason`` property carries the dispatch reason; the
    row's ``raw_value`` is None and the message starts with ``"SKIP:"``.
    """
    rollout = _make_synthetic_grid_rollout(dataset="vortex_shedding_2d")
    results = lint_mesh_rollout(
        rollout,
        case_study="02-physicsnemo-mgn",
        dataset="vortex_shedding_2d",
        model="deepmind-cylinder-flow-gt",
        ckpt_hash="n/a_gt",
    )
    by_rule = {r.rule_id: r for r in results}
    energy_drift = by_rule["harness:energy_drift"]
    dissipation = by_rule["harness:dissipation_sign_violation"]
    for skip_row in (energy_drift, dissipation):
        assert skip_row.raw_value is None
        assert skip_row.message.startswith("SKIP:")
        assert "open-driven-dissipative" in skip_row.extra_properties["skip_reason"]
    # ke_initial / ke_final attached on energy_drift row.
    assert "ke_initial" in energy_drift.extra_properties
    assert "ke_final" in energy_drift.extra_properties
    # Constant velocity -> non-NaN KE; the SKIP fires AFTER KE is computable
    # but the helper still records the series for transparency.
    assert energy_drift.extra_properties["ke_initial"] is not None
    assert energy_drift.extra_properties["ke_final"] is not None
